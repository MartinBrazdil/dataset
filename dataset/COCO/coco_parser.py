import os
from io import BytesIO
from typing import Set

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO

from dataset.utils.dataset_parser import DataParser


class COCOParser(DataParser):
    def __init__(self, split: Set[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_root('/host_home/projects/data/coco')
        self.search_path(['annotations'])
        self.search_path(['instances_val2017',
                          'instances_train2017',
                          'captions_val2017',
                          'captions_train2017',
                          'image_info_test2017',
                          'image_info_test-dev2017',
                          'image_info_unlabeled2017',
                          'person_keypoints_train2017',
                          'person_keypoints_val2017'],
                         extension='.json',
                         subpath=self.args.annotations)
        self.search_path(['train2017',
                          'test2017',
                          'val2017'])
        self.split = split
        if 'TRAIN' in split:
            self.log.info('Loading ' + self.args.instances_train2017)
            self.instances_train2017 = COCO(self.args.instances_train2017)
            self.log.info('Loading ' + self.args.captions_train2017)
            self.captions_train2017 = COCO(self.args.captions_train2017)
            self.log.info('Loading ' + self.args.person_keypoints_train2017)
            self.person_keypoints_train2017 = COCO(self.args.person_keypoints_train2017)
        if 'VAL' in split:
            self.log.info('Loading ' + self.args.instances_val2017)
            self.instances_val2017 = COCO(self.args.instances_val2017)
            self.log.info('Loading ' + self.args.captions_val2017)
            self.captions_val2017 = COCO(self.args.captions_val2017)
            self.log.info('Loading ' + self.args.person_keypoints_val2017)
            self.person_keypoints_val2017 = COCO(self.args.person_keypoints_val2017)
        if 'TEST' in split:
            self.log.info('Loading ' + self.args.image_info_test2017)
            self.image_info_test2017 = COCO(self.args.image_info_test2017)
            self.log.info('Loading ' + self.args.image_info_test_dev2017)
            self.image_info_test_dev2017 = COCO(self.args.image_info_test_dev2017)
            self.log.info('Loading ' + self.args.image_info_unlabeled2017)
            self.image_info_unlabeled2017 = COCO(self.args.image_info_unlabeled2017)

    def deallocate(self):
        if 'TRAIN' in self.split:
            self.log.info('Deallocating ' + self.args.instances_train2017)
            del self.instances_train2017
            self.log.info('Deallocating ' + self.args.captions_train2017)
            del self.captions_train2017
            self.log.info('Deallocating ' + self.args.person_keypoints_train2017)
            del self.person_keypoints_train2017
        if 'VAL' in self.split:
            self.log.info('Deallocating ' + self.args.instances_val2017)
            del self.instances_val2017
            self.log.info('Deallocating ' + self.args.captions_val2017)
            del self.captions_val2017
            self.log.info('Deallocating ' + self.args.person_keypoints_val2017)
            del self.person_keypoints_val2017
        if 'TEST' in self.split:
            self.log.info('Deallocating ' + self.args.image_info_test2017)
            del self.image_info_test2017
            self.log.info('Deallocating ' + self.args.image_info_test_dev2017)
            del self.image_info_test_dev2017
            self.log.info('Deallocating ' + self.args.image_info_unlabeled2017)
            del self.image_info_unlabeled2017

    def img_path(self, img_ann: dict):
        img_dir = os.path.split(os.path.dirname(img_ann['coco_url']))[1]
        return os.path.join(self.args.__dict__[img_dir], img_ann['file_name'])

    def load_img(self, img_ann: dict):
        img_path = self.img_path(img_ann)
        with open(img_path, 'rb') as img_file:
            img = Image.open(img_file)
            # img.load()
        return img

    def load_img_binary(self, img_ann: dict):
        img_path = self.img_path(img_ann)
        with open(img_path, 'rb') as img_file:
            return img_file.read()

    def load_PIL_pixels(self, img_bin):
        img = Image.open(BytesIO(img_bin))
        img.load()
        return img

    def load_torch_pixels(self, img_bin):
        pil_img = self.load_PIL_pixels(img_bin)
        return torch.tensor(np.array(pil_img), dtype=torch.int8)
