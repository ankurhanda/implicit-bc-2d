import os.path as osp
import numpy as np
from torch.utils.data import Dataset
# from dataclasses import dataclass

import torch
from torchvision import transforms

import cv2 
from typing import List, Tuple

class ImplicitBCDataset_2D(Dataset):

    #TODO: move to dataclass or attrs 
    def __init__(self, 
                 dataset_size: int, 
                 img_size: Tuple[int, int]):
        
        self.dataset_size = dataset_size
        self.img_size     = img_size 

        self.height       = self.img_size[0]
        self.width        = self.img_size[1]

        '''
        Generate possible random keypoint locations 
        '''
        self.keypts_xy = []
        offset = 20
        for i in range(self.dataset_size):

            x = np.random.randint(self.img_size[1]/2-offset, self.img_size[1]/2+offset)
            y = np.random.randint(self.img_size[0]/2-offset, self.img_size[0]/2+offset)

            self.keypts_xy.append([x, y])

        print(self.keypts_xy)

        self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        data = self.prepare_train_img(idx)

        return data 

    def prepare_train_img(self, idx):

        results = dict()

        img = np.ones((self.height, self.width, 3), dtype=np.float)
        x, y = self.keypts_xy[idx]
        img[y, x, :] = 0 

        normalised_x = float(x)/self.width
        normalised_y = float(y)/self.height

        results['images'] = self.transform(img) 

        results['annotations'] = [x, y]
        results['normalised_annotations'] = [2*normalised_x-1, 2*normalised_y-1]

        return results

if __name__ == "__main__":

    from PIL import Image

    implicit_bc_dataset_2d = ImplicitBCDataset_2D(dataset_size=10,
                                                  img_size=(128, 128))

    for i in range(len(implicit_bc_dataset_2d)):
        y = implicit_bc_dataset_2d[i]

        input_img     = y['images']
        target_coords = y['normalised_annotations']
        coords        = y['annotations']
    
        img = np.transpose(input_img.numpy(), (1, 2, 0))
        img = np.array(np.uint8(img*255.0))

        img = cv2.circle(np.ascontiguousarray(img), (int(coords[0]), int(coords[1])), radius=2, color=(255,0,0), thickness=1)

        cv2.imshow('image',img)
        import time
        time.sleep(2.01)
        k = cv2.waitKey(10)
        # Press q to break
        if k == ord('q'):
            break