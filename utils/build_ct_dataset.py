# – coding: utf-8 --
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv


import json
# Interface for accessing the COCO-Text dataset.

# COCO-Text is a large dataset designed for text detection and recognition.
# This is a Python API that assists in loading, parsing and visualizing the
# annotations. The format of the COCO-Text annotations is also described on
# the project website http://vision.cornell.edu/se3/coco-text/. In addition to this API, please download both
# the COCO images and annotations.
# This dataset is based on Microsoft COCO. Please visit http://mscoco.org/
# for more information on COCO, including for the image data, object annotatins
# and caption annotations.

# An alternative to using the API is to load the annotations directly
# into Python dictionary:
# with open(annotation_filename) as json_file:
#     coco_text = json.load(json_file)
# Using the API provides additional utility functions.

# The following API functions are defined:
#  COCO_Text  - COCO-Text api class that loads COCO annotations and prepare data structures.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.

# COCO-Text Toolbox.        Version 1.1
# Data and  paper available at:  http://vision.cornell.edu/se3/coco-text/
# Code based on Microsoft COCO Toolbox Version 1.0 by Piotr Dollar and Tsung-Yi Lin
# extended and adapted by Andreas Veit, 2016.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
import numpy as np
import copy
import os
import random
random.seed(9726)

class COCO_Text:
    def __init__(self, coco_file,lmdb_file):
        """
        Constructor of COCO-Text helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = {}
        self.imgToAnns = {}
        self.catToImgs = {}
        self.imgs = {}
        self.cats = {}
        self.val = []
        self.test = []
        self.train = []
        
        self.coco = {}
        self.lmdb = {}
        with open(coco_file, 'r') as f:
            self.coco = json.loads(f.read())
        with open(lmdb_file, encoding='utf-8') as f:
            for row in csv.reader(f, skipinitialspace=True):
                self.lmdb[row[0]] = row[1]



    def init_imgs(self):
        ct_imgs = {} #coco text
        co_imgs = self.coco['images'] #coco images list {'height': 926, 'width': 1333, 'id': 1, 'file_name': 'image_0.jpg'},
        for co_img in co_imgs:
            ct_imgs[co_img['id']] = co_img
            ri = random.randint(0,10)
            ct_imgs[co_img['id']]['set']  = 'train' if ri<=7 else 'val' if ri==8 else 'test'
        self.imgs = ct_imgs
    def get_name_by_imageid(self,id:int):
        return CT.imgs[id]['file_name'].split('.')[0] #不带后缀
    def init_anns(self):
        ct_anns = {} # coco text
        co_anns = self.coco['annotations']
        ct_imgToAnns = {}
        gid = 0
        last_img_id = ''
        for co_ann in co_anns:
            ct_anns[str(co_ann['id'])] = {}
            ct_anns[str(co_ann['id'])]['polygon'] = co_ann['segmentation'][0] #要有[0]
            ct_anns[str(co_ann['id'])]['language'] = 'non-english'
            ct_anns[str(co_ann['id'])]['area'] = co_ann['area']
            ct_anns[str(co_ann['id'])]['id'] = co_ann['id']
            ct_anns[str(co_ann['id'])]['image_id'] = str(co_ann['image_id'])
            if last_img_id != str(co_ann['image_id']):
                gid = 0
            last_img_id = str(co_ann['image_id'])
            print(f'{self.get_name_by_imageid(int(co_ann["image_id"]))}_{gid}.jpg')
            ct_anns[str(co_ann['id'])]['utf8_string'] = self.lmdb[f'{self.get_name_by_imageid(int(co_ann["image_id"]))}_{gid}.jpg']#'----'
            gid += 1
            ct_anns[str(co_ann['id'])]['bbox'] = co_ann['bbox']
            ct_anns[str(co_ann['id'])]['legibility'] = 'legible' #全部可读
            ct_anns[str(co_ann['id'])]['class'] = 'handwritten' #全部手写
            if str(co_ann['image_id']) in ct_imgToAnns:
                ct_imgToAnns[str(co_ann['image_id'])].append(co_ann['id'])
            else:
                ct_imgToAnns[str(co_ann['image_id'])] = [co_ann['id']]
            #ct_imgToAnns[str(co_ann['image_id'])] = [co_ann['id']] if not str(co_ann['id']) in ct_imgToAnns else ct_imgToAnns[str(co_ann['id'])].append(co_ann['id'])
        self.anns = ct_anns
        self.imgToAnns = ct_imgToAnns
    # def init_imgToAnns(self):
    #     assert self.imgs == {} or self.anns == {},'Make Sure init_imgs() and init_anns() was called before'
    #     ct_imgToAnns = {} # coco text
    #     for img_id in self.imgs.keys():
    #         ct_imgToAnns[str(img_id)] = []
        
    def init_cats_info(self):
        self.cats = {'legibility': {'1': {'id': 1, 'name': 'legible'},
        '2': {'id': 2, 'name': 'illegible'}},
        'class': {'1': {'id': 1, 'name': 'machine printed'},
        '3': {'id': 3, 'name': 'others'},
        '2': {'id': 2, 'name': 'handwritten'}},
        'script': {'1': {'id': 1, 'name': 'english'},
        '3': {'id': 3, 'name': 'na'},
        '2': {'id': 2, 'name': 'not english'}}}
        self.info = {'url': 'https://www.cvmart.net/race/10340/base',
        'date_created': '2022-07-20',
        'version': '0.1',
        'description': 'Dataset of the DAR contest',
        'author': 'Built by Penghai Zhao'}

    def dump_to_json(self,json_dir):
        self.dataset['imgs'] = self.imgs
        self.dataset['imgToAnns'] = self.imgToAnns
        self.dataset['cats'] = self.cats
        self.dataset['anns'] = self.anns
        self.dataset['info'] = self.info
        with open(os.path.join(json_dir,'label-CT.json'), 'w',encoding='utf8') as file_obj:
            json.dump(self.dataset, file_obj,ensure_ascii=False)
            
        pass
        
CT = COCO_Text(coco_file=r'F:\Data\GJJS-dataset\dataset\train\coco\annotations-origin.json\dataset.json',lmdb_file=r'F:\Data\GJJS-dataset\dataset\train\text_label_ct_gid-utf8.csv')
CT.init_imgs()
CT.init_anns()
CT.init_cats_info()
# print(CT.imgToAnns)
# # print(CT.imgs)
# print(len(CT.imgToAnns['1000']))

CT.dump_to_json(r'F:\Data\GJJS-dataset\dataset\train')