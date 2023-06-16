import csv
import functools
import glob
import os
import random

from collections import namedtuple


import numpy as np

import nibabel as nib
import torch
import torch.cuda
from torch.utils.data import Dataset


from util.disk import getCache
from util.util import IrcTuple


from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = getCache('code_raw')


CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'diameter_mm, series_name, center_xyz'
)
def xyz2zxy(X):
    return np.transpose(X, (2, 0, 1))
@functools.lru_cache(1)
def getCandidateInfoList():
    path='C:\\Users\\Usuari\\Desktop\\TFM\\DATA'
   
    candidateInfo_list = []
    for file in os.listdir(path):
        for files in os.listdir(f"{path}\{file}"):
            if files.endswith("location.txt"):
                if os.path.getsize(f"{path}\{file}\{files}") != 0:
                    series_name = file
                    with open(f"{path}\{file}\{files}", "r") as f:
                        for row in list(csv.reader(f)):
                            candidateCenter_xyz = tuple([float(x) for x in row[0:3]])
                            candidateDiameter_mm = float(row[3])
                                
                            candidateInfo_list.append(CandidateInfoTuple(
                                candidateDiameter_mm,
                                series_name,
                                candidateCenter_xyz,
                            ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

@functools.lru_cache(1)
def getCandidateInfoDict():
    candidateInfo_list = getCandidateInfoList()
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_name,
                                      []).append(candidateInfo_tup)

    return candidateInfo_dict

class MRA:
    def __init__(self, series_name):
        nii_path = glob.glob(
            'C:\\Users\\Usuari\\Desktop\\TFM\\DATA\\{}\\TOF.nii.gz'.format(series_name)
        )
        mra_nii = nib.load(nii_path[0])
        mra_a = np.array(mra_nii.get_fdata(), dtype=np.float32) 
        mra_a=xyz2zxy(mra_a)

        
        self.series_name = series_name
        self.a = mra_a

        #get scpacing in each direction
        self.vxSize_xyz = IrcTuple(*mra_nii.header.get_zooms())

        self.positive_mask = self.buildAnnotationMask() ## returns anotation mask

    def buildAnnotationMask(self):
        nii_mask_path = glob.glob(
            'C:\\Users\\Usuari\\Desktop\\TFM\\DATA\\{}\\aneurysms.nii.gz'.format(self.series_name)
        ) 
        mask_nii = nib.load(nii_mask_path[0])
        mask_a = np.array(mask_nii.get_fdata(), dtype=np.int_)
        mask_a=xyz2zxy(mask_a)

        return mask_a
    
    def getRawCandidate(self, center_xyz, width_xyz):

        slice_list = []
        for axis, center_val in enumerate(center_xyz):
            start_ndx = int(round(center_val - width_xyz[axis]/2))
            end_ndx = int(start_ndx + width_xyz[axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_xyz[axis])

            if end_ndx > self.a.shape[axis]:
                end_ndx = self.a.shape[axis]
                start_ndx = int(self.a.shape[axis] - width_xyz[axis])

            slice_list.append(slice(start_ndx, end_ndx))
        
        pos_chunk = self.positive_mask[tuple(slice_list)]
        mra_chunk = self.a[tuple(slice_list)]

        return mra_chunk,pos_chunk

    

@functools.lru_cache(1, typed=True)
def getMRA(series_name):
    return MRA(series_name)

@raw_cache.memoize(typed=True)
def getMRARawCandidate(series_name, center_xyz, width_xyz):
    mra = getMRA(series_name)
    mra_chunk, pos_chunk= mra.getRawCandidate(center_xyz,width_xyz)
    return mra_chunk, pos_chunk

class ADAM2dSegmentationDataset(Dataset):
    def __init__(self,
                 isValSet_bool=None,
                 config=0
            ):
        self.isValSet_bool = isValSet_bool

        self.series_list = sorted(getCandidateInfoList(),reverse=True)

        self.config = config
        self.lists = self.dividir_lista()

        if isValSet_bool:
            self.series_list = self.lists[self.config]
            assert self.series_list
        else:
            self.series_list = [e for ls in self.lists for e in ls if ls != self.lists[self.config]]
            assert self.series_list
            

        log.info("{!r}: {} {} series".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool]
        ))

    def dividir_lista(self):
        k_1 = []
        k_2 = []
        k_3 = []
        k_4 = []
        k_5 = []
        
        for i, elemento in enumerate(self.series_list, start=1):
            if i % 5 == 1:
                k_1.append(elemento)
            elif i % 5 == 2:
                k_2.append(elemento)
            elif i % 5 == 3:
                k_3.append(elemento)
            elif i % 5 == 4:
                k_4.append(elemento)
            elif i % 5 == 0:
                k_5.append(elemento)
    
        return k_1,k_2,k_3,k_4,k_5
    
    def shuffleSamples(self):
        random.shuffle(self.series_list)

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.series_list[ndx % len(self.series_list)]
        return self.getitem_trainingCrop(candidateInfo_tup)

    def getitem_trainingCrop(self, candidateInfo_tup):
        
        center_zxy = candidateInfo_tup.center_xyz[-1:] + candidateInfo_tup.center_xyz[:-1]
        mra_a, pos_a= getMRARawCandidate(
            candidateInfo_tup.series_name,
            center_zxy,
            (16, 64, 64)
        )

        mra_t = torch.from_numpy(mra_a).to(torch.float32)
        pos_t = torch.from_numpy(pos_a).to(torch.long)

        return mra_t.unsqueeze(0), pos_t.unsqueeze(0)

    

class PrepcacheADAMDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.candidateInfo_list = getCandidateInfoList()

        self.seen_set = set()
        self.candidateInfo_list.sort(key=lambda x: x.series_name)

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):

        candidateInfo_tup = self.candidateInfo_list[ndx]
        center_zxy = candidateInfo_tup.center_xyz[-1:] + candidateInfo_tup.center_xyz[:-1]
        getMRARawCandidate(candidateInfo_tup.series_name, center_zxy, (16, 64, 64))

        series_name = candidateInfo_tup.series_name
        if series_name not in self.seen_set:
            self.seen_set.add(series_name)

        return 0, 1 
