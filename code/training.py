import argparse
import datetime
import hashlib
import os
import shutil
import socket
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim

from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from .dsets import ADAM2dSegmentationDataset, getMRA
from util.logconf import logging
from .model import UNetWrapper, SegmentationAugmentation
from .loss import GeneralizedDiceLoss
from torch.optim.lr_scheduler import ExponentialLR

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


METRICS_LOSS_NDX = 1
METRICS_DICE_NDX = 2
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_SIZE = 10

class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=5,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )

        parser.add_argument('--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-noise',
            help="Augment the training data by randomly adding noise to the data.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--tb-prefix',
            default='p2ch13',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='none',
        )


        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.val_writer = None
        self.augmentation = True

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.segmentation_model, self.augmentation_model = self.initModel()
        self.optimizer = self.initOptimizer()
        self.criterion = GeneralizedDiceLoss().to(self.device)

    def initModel(self):
        segmentation_model = UNetWrapper(
            in_channels=1, 
            out_channels=1,
        )
        
        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model

    def initOptimizer(self):

        return Adam(self.segmentation_model.parameters(),weight_decay=1e-5,lr=0.00001)
    
    def initTrainDl(self):
        train_ds = ADAM2dSegmentationDataset(
            config=4,
            isValSet_bool=False,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        val_ds = ADAM2dSegmentationDataset(
            config=4,
            isValSet_bool=True,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '_trn_seg_' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '_val_seg_' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_score = 0.0
        self.validation_cadence = 1
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                # if validation is wanted
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                self.saveModel('seg', epoch_ndx, score == best_score)

                self.logImages(epoch_ndx, 'trn', train_dl)
                self.logImages(epoch_ndx, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()
        train_dl.dataset.shuffleSamples()

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         classificationThreshold=0.5):
        input_t, label_t = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)
       
        
        prediction_g = self.segmentation_model(input_g)
        loss = self.criterion(prediction_g, label_g).mean()
      

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():

            predictionBool_g = (prediction_g[:, 0:1] 
                                > classificationThreshold).to(torch.float32)
            
            label_g = label_g.to(torch.float32)

            tp = (predictionBool_g *  label_g).sum(dim=[1,2,3,4])
            fn = ((1-predictionBool_g) *  label_g).sum(dim=[1,2,3,4])
            fp = (predictionBool_g *  (1-label_g)).sum(dim=[1,2,3,4])

            intersection = (predictionBool_g * label_g).sum(dim=[1,2,3,4])
            dice = (2. * intersection )/(predictionBool_g.sum(dim=[1,2,3,4]) + label_g.sum(dim=[1,2,3,4]))
            dice = torch.where(torch.isnan(dice), torch.tensor(1.), dice)
                    
            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss
            metrics_g[METRICS_DICE_NDX, start_ndx:end_ndx] = dice.mean()
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return loss
        

    def logImages(self, epoch_ndx, mode_str, dl):
        self.segmentation_model.eval()
        
        images = sorted(dl.dataset.series_list)[:12]
        for series_ndx, tup in enumerate(images):
            mra_t, label_t=dl.dataset.getitem_trainingCrop(tup)

            input_g = mra_t.to(self.device).unsqueeze(0)

            prediction_g = self.segmentation_model(input_g)[0]

            prediction_a = (prediction_g.to('cpu').detach().numpy() > 0.5)
            prediction_a = prediction_a[0,8,:,:]
            label_a = label_t.numpy() > 0.5
            label_a = label_a[0,8,:,:]


            mra_t[0,8,:,:] /= torch.max(mra_t[0,8,:,:])
            mra_t[0,8,:,:] += 0.5

            mraSlice_a = mra_t[0,8,:,:].numpy()

            image_a = np.zeros((64, 64, 3), dtype=np.float32)
            image_a[:,:,:] = mraSlice_a.reshape((64,64,1))
            image_a[:,:,0] += prediction_a & (1 - label_a)

            image_a[:,:,1] += prediction_a & label_a
            image_a *= 0.5
            image_a.clip(0, 1, image_a)

            writer = getattr(self, mode_str + '_writer')
            writer.add_image(
                f'{mode_str}/{series_ndx}_prediction',
                image_a,
                self.totalTrainingSamples_count,
                dataformats='HWC',
            )

            if epoch_ndx == 1:
                image_a = np.zeros((64, 64, 3), dtype=np.float32)
                image_a[:,:,:] = mraSlice_a.reshape((64, 64,1))
                image_a[:,:,1] += label_a  # Green

                image_a *= 0.5
                image_a[image_a < 0] = 0
                image_a[image_a > 1] = 1
                writer.add_image(
                    '{}/{}_label'.format(
                        mode_str,
                        series_ndx
                    ),
                    image_a,
                    self.totalTrainingSamples_count,
                    dataformats='HWC',
                )
                # This flush prevents TB from getting confused about which
                # data item belongs where.
            writer.flush()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()
        metrics_dict['dice/all'] = metrics_a[METRICS_DICE_NDX].mean()

        metrics_dict['percent_all/tp'] = \
            sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = \
            sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = \
            sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100


        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
            / ((precision + recall) or 1)

        log.info(("E{} {:8} "
                 + "{loss/all:.4f} loss, "
                 + "{dice/all:.4f} dice, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{dice/all:.4f} dice, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
        ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        writer.flush()

        score = metrics_dict['pr/recall']

        return score


    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'data-unversioned',
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment,
                self.totalTrainingSamples_count,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.info("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'data-unversioned', 'models',
                self.cli_args.tb_prefix,
                '{}_{}_{}.best.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment)
            )
            torch.save(state, file_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())


if __name__ == '__main__':

    SegmentationTrainingApp().main()
