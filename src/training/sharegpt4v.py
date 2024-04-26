import json
import cv2
from PIL import Image

import torch
import torch.utils.data as data
import os
import numpy as np
import random

data4v_root = '/home/akoukounas/ShareGPT4V/data/'
json_name = 'share-captioner_coco_lcs_sam_1246k_1107.json'
image_root = '/home/akoukounas/ShareGPT4V/data/'

class share4v_val_dataset(data.Dataset):
    def __init__(self,preprocess, tokenizer):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[:self.total_len]
        self.preprocess = preprocess
        self.tokenizer = tokenizer
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption


class share4v_train_dataset(data.Dataset):
    def __init__(self,preprocess,tokenizer):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[self.total_len:]
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        if index < 570486:
            index += 570486
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        

        caption_short = caption.split(". ")[0]
        tokenized_caption = self.tokenizer(caption)[0]
        tokenized_caption_short = self.tokenizer(caption_short)[0]

        image_name = self.image_root + self.json_data[index]['image']
        try:
            image = Image.open(image_name)
            image_tensor = self.preprocess(image)
            return image_tensor, tokenized_caption, tokenized_caption_short
        except:
            print(image_name)
            image_tensor = torch.zeros(3, 224, 224)
            with open("/home/akoukounas/image_names.txt", "w") as file:
                file.write(image_name + "\n")
            return image_tensor, tokenized_caption, tokenized_caption_short