import json
import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

data4v_root = '/home/akoukounas/ShareGPT4V/data/'
json_name = 'share-captioner_coco_lcs_sam_1246k_1107.json'
image_root = '/home/akoukounas/ShareGPT4V/data/'


class share4v_val_dataset(data.Dataset):
    def __init__(self, preprocess, tokenizer):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[: self.total_len]
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace('\n', ' ')
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption


class share4v_train_dataset(data.Dataset):
    def __init__(self, preprocess, tokenizer):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[self.total_len :]
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        try:  # try except is used in case any image is missing (we have not tested sam dataset)
            caption = self.json_data[index]['conversations'][1]['value']
            caption = caption.replace('\n', ' ')

            caption_short = caption.split('. ')[0]

            image_name = self.image_root + self.json_data[index]['image']
            image = Image.open(image_name)
            image_tensor = self.preprocess(image)
            return image_tensor, caption, caption_short
        except:
            print(image_name)
            with open('./image_names.txt', 'w') as file:
                file.write(image_name + '\n')

            index += 570486  # first 570486 images are from sam, after that index all images are all okay
            caption = self.json_data[index]['conversations'][1]['value']
            caption = caption.replace('\n', ' ')

            caption_short = caption.split('. ')[0]

            image_name = self.image_root + self.json_data[index]['image']
            image = Image.open(image_name)
            image_tensor = self.preprocess(image)
            return image_tensor, caption, caption_short
