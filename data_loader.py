import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass, field 

import torch
from torchvision import transforms

import cv2 
from typing import List, Tuple

def set_seed(seed=91):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass 
class ImplicitBCDataset_2D(Dataset):

    dataset_size: int
    img_size: Tuple[int, int] 
    fixed_seed:bool = field(default=False)
    mode: str = field(default='train')
    neg_pairs: float = field(default=99) # it will be 9 + 1 positive pair 
 
    def __post_init__(self):
        super(ImplicitBCDataset_2D).__init__()
        
        self.height       = self.img_size[0]
        self.width        = self.img_size[1]

        if self.fixed_seed:
            set_seed(seed=91)

        '''
        Generate possible random keypoint locations 
        '''
        self.keypts_xy = []
        offset = 20
        for i in range(self.dataset_size):

            x = np.random.rand(1)
            y = np.random.rand(1)
            
            # import ipdb; ipdb.set_trace();

            x = int(x * self.width -1)
            y = int(y * self.height-1)

            # x = np.random.randint(self.width/2-offset, self.width/2+offset)
            # y = np.random.randint(self.height/2-offset, self.height/2+offset)

            self.keypts_xy.append([x, y])

        np.savetxt('{}_dataset.txt'.format(self.mode), np.array(self.keypts_xy))
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
        img = cv2.circle(np.ascontiguousarray(img), (int(x), int(y)), radius=8, color=(255,0,0), thickness=-1)

        normalised_x = float(x)/self.width
        normalised_y = float(y)/self.height

        results['images'] = self.transform(img) 

        results['annotations'] = np.array([x, y], dtype=np.float32)
        results['normalised_annotations'] = np.array([2*normalised_x-1, 2*normalised_y-1], dtype=np.float32)

        xy_neg = np.random.rand(self.neg_pairs, 2)
        results['normalised_negatives'] = np.array(2*xy_neg-1.0, dtype=np.float32)

        return results

if __name__ == "__main__":

    set_seed(seed=91)
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