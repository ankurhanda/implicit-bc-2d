import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass, field 

import torch
from torchvision import transforms

import cv2 
from typing import List, Tuple

@dataclass 
class ImplicitBCDataset_2D(Dataset):
    dataset_size: int
    img_size: Tuple[int, int] 
    # height: int = field(init=False)
    # width: int = field(init=False)

    #TODO: move to dataclass or attrs 
    def __post_init__(self):
        super(ImplicitBCDataset_2D).__init__()
        
        self.height       = self.img_size[0]
        self.width        = self.img_size[1]

        print(self.height, self.width)

        '''
        Generate possible random keypoint locations 
        '''
        self.keypts_xy = []
        offset = 20
        for i in range(self.dataset_size):

            x = np.random.randint(self.width/2-offset, self.width/2+offset)
            y = np.random.randint(self.height/2-offset, self.height/2+offset)

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

        img = np.ones((self.height, self.width, 3), dtype=np.uint8)*255
        x, y = self.keypts_xy[idx]
        img = cv2.circle(np.ascontiguousarray(img), (int(x), int(y)), radius=4, color=(255,0,0), thickness=-1)

        normalised_x = float(x)/self.width
        normalised_y = float(y)/self.height

        results['images'] = self.transform(img) 

        results['annotations'] = np.array([x, y], dtype=np.float32)
        results['normalised_annotations'] = np.array([2*normalised_x-1, 2*normalised_y-1], dtype=np.float32)

        return results

if __name__ == "__main__":

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