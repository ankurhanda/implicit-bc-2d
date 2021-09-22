import os
import multiprocessing
from PIL import Image
import numpy as np 

import torch
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
import csv
import fire 


import torch.nn as nn 
import torch.nn.functional as F

LR = 3e-4 #1e-3

from networks import BaseCNN
import cv2 

# pytorch lightning module
class ImplicitBC_2d_Learner(pl.LightningModule):
    def __init__(self, **kwargs):
        super(ImplicitBC_2d_Learner, self).__init__()
        # self.save_hyperparameters()

        self.batch_iter = 0 
        self.learner = BaseCNN(in_channels=3)

        self.loss = torch.nn.MSELoss()
    
    def forward(self, images):
        return self.learner(images)
    
    # def plot_ground_truth(self, heatmaps, annotated_images):

    #     import cv2 
    #     while True:

    #         myheatmaps = heatmaps.cpu().numpy()
    #         myheatmaps = myheatmaps.reshape(2, 3, self.num_keypoints, 30, 40) #(b, num_cam, num_heatmaps, 30, 40)
            
    #         for b in range(0, 2):
    #             for cam in range(0, 3):
    #                 img = (annotated_images[b, cam].permute(1, 2, 0).detach() * 255.0).to('cpu').numpy()
    #                 img = img.copy()
    #                 img = cv2.resize(img, (40, 30), cv2.INTER_LINEAR)

    #                 for count in range(0, self.num_keypoints):

    #                     curr_heatmap = (myheatmaps[b][cam][count].copy()*255.0)

    #                     my_img = img + curr_heatmap.reshape(30, 40, 1)
    #                     my_img = np.clip(my_img, 0, 255)
    #                     my_img = my_img.astype('uint8').copy()

    #                     cv2.imshow('image',my_img)
    #                     import time
    #                     time.sleep(2.01)
    #                     k = cv2.waitKey(10)
    #                     # Press q to break
    #                     if k == ord('q'):
    #                         break
        
    #     return


    def training_step(self, batch, batch_idx):    
        
        source_views = batch['images']
        xy_ground_truth_normalised = batch['normalised_annotations']
       
        xy_pred_normalised = self.forward(source_views)

        loss = self.loss(xy_pred_normalised, xy_ground_truth_normalised)

        self.log('loss_train', loss)
        self.log('epoch', self.current_epoch)

        canvases = []
        for i in range(0, xy_pred_normalised.shape[0]):
            # import ipdb; ipdb.set_trace();
            input_img = source_views[i].clone()
            img = np.transpose(input_img.cpu().detach().numpy(), (1, 2, 0))
            img = np.array(np.uint8(img*255.0))

            xy = xy_pred_normalised[i].clone().cpu().detach().numpy()

            x  = (xy[0]+1)/2.0 * input_img.shape[-1]
            y  = (xy[1]+1)/2.0 * input_img.shape[-2]
        
            img = cv2.circle(np.ascontiguousarray(img), (int(x), int(y)), radius=2, color=(255,0,0), thickness=-1)

            canvases.append(img)

        canvases = np.array(canvases)
        canvases = torch.from_numpy(canvases / 255.0).permute(0, 3, 1, 2)

        grid_keypoints = torchvision.utils.make_grid(canvases)
        self.logger.experiment.add_image('predicted_no_anno', grid_keypoints, self.batch_iter)

        self.batch_iter += 1

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)


def run(mode):

    from torch.utils.data import DataLoader, Dataset
    from data_loader import ImplicitBCDataset_2D

    from pytorch_lightning.callbacks import ModelCheckpoint
    import cv2

    implicit_bc_dataset_2d = ImplicitBCDataset_2D(dataset_size=10,
                                                  img_size=(128, 128))

    EPOCHS = 100 

    if mode == 'train':
            
        train_loader = DataLoader(implicit_bc_dataset_2d, batch_size=1, 
                                            num_workers=12, shuffle=True, 
                                            pin_memory=True, drop_last=False) 

        model = ImplicitBC_2d_Learner()

        # import ipdb; ipdb.set_trace();                                        

        # checkpoint_callback = ModelCheckpoint(
        #     monitor='loss_train',
        #     dirpath='trained_keypoint3d_models/views_{}/skip_frames_{}'.format(NUM_VIEWS, SKIP_FRAMES),
        #     filename="sample-loss-{epoch:03d}",
        #     save_top_k=5,
        #     save_weights_only=True
        #     )

        from pytorch_lightning.plugins import DDPPlugin

        trainer = pl.Trainer(
                gpus = [0], 
                max_epochs = EPOCHS,
                accumulate_grad_batches = 1,
                stochastic_weight_avg=True,
                plugins=DDPPlugin(find_unused_parameters=True),
                #distributed_backend='ddp',
                # callbacks=[checkpoint_callback]
            )

        trainer.fit(model, train_loader) #, val_loader)
    
    # elif mode == 'test':

    #     model_path = 'trained_keypoint3d_models/views_{}/skip_frames_{}/129/sample-loss-epoch=129.ckpt'.format(NUM_VIEWS, SKIP_FRAMES)
    #     print('****** testing model ', model_path)


    #     model = KeyPointDexPilot3DLearner.load_from_checkpoint(model_path,**input_arguments)   
    #     test_loader = DataLoader(dex_learn_dataset, batch_size=1, 
    #                                     num_workers=12, shuffle=False, 
    #                                     pin_memory=True, drop_last=False)


    #     model = model.cuda()
    #     model.eval()

    #     # which_demos = dex_learn_dataset.get_demo_ids()
    #     # which_demos = which_demos[0] if len(which_demos) == 1 else which_demos[0]
    #     # print('held-out trajectories: ', which_demos)

    #     entered = False 

    #     #TODO: Fix this for multiple held out trajectories
    #     for count, elem in enumerate(test_loader):
            
    #         demo_id = elem['img_info']['demo_id'].cpu().detach().numpy()
    #         img_id  = elem['img_info']['image_id'].cpu().detach().numpy()

    #         with torch.no_grad():

    #             source_views = elem['images']
    #             # source_views = elem['annotated_images']

    #             imgs0 = source_views["view_0"].clone().detach().unsqueeze(1)
    #             imgs1 = source_views["view_1"].clone().detach().unsqueeze(1)
    #             imgs2 = source_views["view_2"].clone().detach().unsqueeze(1)

    #             # import ipdb; ipdb.set_trace();

    #             # '''
    #             # get ground truth keypoints 
    #             # '''
    #             # keypoint_annotations = elem['keypoint_annotations']

    #             # keypts0 = keypoint_annotations['keypoints_view_0'].detach()
    #             # keypts1 = keypoint_annotations['keypoints_view_1'].detach()
    #             # keypts2 = keypoint_annotations['keypoints_view_2'].detach()

    #             # keypoints = torch.cat([keypts0, keypts1, keypts2], dim=1)
    #             # keypoints = keypoints.cpu().numpy()
                

    #             sources = torch.cat([imgs0, imgs1, imgs2], dim=1).cuda()

    #             xyzs_world, loss_dict, heatmaps_hat, xy_normalized = model.forward(sources)

    #             batch_size, num_cam, _, h, w = sources.shape 

    #             heatmaps_hat = heatmaps_hat.clone()
    #             heatmaps_hat = heatmaps_hat.reshape(-1, 30, 40)
    #             xy = model.learner.heatmap_to_xy(heatmaps_hat)
    #             xy = xy.reshape(batch_size, num_cam, input_arguments['num_keypoints'], 2)
    #             xy = (xy+1)/2 
    #             xy = xy.to('cpu').detach().numpy()
    #             img = sources.clone().permute(0, 1, 3, 4, 2).cpu().detach().numpy()
        
    #             canvases = []

    #             for b in range(0, batch_size):
    #                 curr_canvas = []
    #                 for c in range(0, num_cam):
    #                     curr_img = (img[b, c].copy()*255.0).astype('uint8')
                        
    #                     for (x, y) in xy[b][c]:
    #                         colour=(255.0, 0, 0)
    #                         cv2.circle(curr_img, (int(x*320),int(y*240)), 3, colour, -1)
                        
    #                     # for (kx, ky) in keypoints[b][c]:
    #                     #     if kx > 0 and ky > 0:
    #                     #         colour=(0.0, 255.0, 0)
    #                     #         cv2.circle(curr_img, (int(kx),int(ky)), 2, colour, -1)

    #                     curr_canvas.append(curr_img)

    #                 curr_canvas = np.array(curr_canvas)
    #                 curr_canvas = np.concatenate(curr_canvas, axis=1)

    #                 canvases.append(curr_canvas.copy())

    #             canvases = np.array(canvases)
    #             # import ipdb; ipdb.set_trace();
    #             cv2.imwrite('keypoint_results/keypoint_results_{:04d}.png'.format(count), canvases[0][...,::-1])
    #             print(count, '  files written out of ...   ', dex_learn_dataset.__len__())
    #             # import ipdb; ipdb.set_trace();



    #             for b in range(0, batch_size):
    #                 for c in range(0, num_cam):
    #                     dirName = 'keypoints2d/sugar_box_lift' + str(demo_id[0]) + '/color{}'.format(c+1)
    #                     if not os.path.exists(dirName):
    #                         os.makedirs(dirName)

    #                     keypts =[]
    #                     for (x, y) in xy[b][c]:
    #                         kpt_x, kpt_y = int(x*320),int(y*240)
    #                         keypts.append([kpt_x, kpt_y])
                        
    #                     fileName = dirName + '/keypoints2d_{:04d}.txt'.format(img_id[0])
    #                     np.savetxt(fileName, np.array(keypts))

if __name__ == "__main__":
    fire.Fire(run)


