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


# pytorch lightning module
class ImplicitBC_2d(pl.LightningModule):
    def __init__(self, **kwargs):
        super(ImplicitBC_2d, self).__init__()
        # self.save_hyperparameters()

        self.learner = ImplicitBC_2d(**kwargs)
        self.batch_iter = 0 

        self.loss = torch.nn.MSELoss()
    
    def forward(self, images):
        return self.learner(images=images)
    
    def plot_ground_truth(self, heatmaps, annotated_images):

        import cv2 
        while True:

            myheatmaps = heatmaps.cpu().numpy()
            myheatmaps = myheatmaps.reshape(2, 3, self.num_keypoints, 30, 40) #(b, num_cam, num_heatmaps, 30, 40)
            
            for b in range(0, 2):
                for cam in range(0, 3):
                    img = (annotated_images[b, cam].permute(1, 2, 0).detach() * 255.0).to('cpu').numpy()
                    img = img.copy()
                    img = cv2.resize(img, (40, 30), cv2.INTER_LINEAR)

                    for count in range(0, self.num_keypoints):

                        curr_heatmap = (myheatmaps[b][cam][count].copy()*255.0)

                        my_img = img + curr_heatmap.reshape(30, 40, 1)
                        my_img = np.clip(my_img, 0, 255)
                        my_img = my_img.astype('uint8').copy()

                        cv2.imshow('image',my_img)
                        import time
                        time.sleep(2.01)
                        k = cv2.waitKey(10)
                        # Press q to break
                        if k == ord('q'):
                            break
        
        return


    def training_step(self, batch, batch_idx):    
        
        source_views = batch['images']
       
        # heatmaps_hat = self.learner.encode_heatmaps(sources)
        xy_normalised = self.forward(source_views)

        loss = self.loss(xy_pred_normalised, xy_ground_truth_normalised)

        self.log('loss_train', loss)
        self.log('epoch', self.current_epoch)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)


def run(mode):

    import sys 
    sys.path.insert(1, '/home/ankur/workspace/DexLearn/dexpilot-dataloader')

    from torch.utils.data import DataLoader, Dataset
    from dataset_reader import DexLearnDataset

    from pytorch_lightning.callbacks import ModelCheckpoint
    import cv2

    # vision_backbone = 'resnet'


    # BATCH_SIZE = 40 if vision_backbone == 'resnet' else 25 #if vision_backbone == 'transformer'
    BATCH_SIZE = 10
    EPOCHS = 200
    NUM_VIEWS = 3
    SKIP_FRAMES=100
    PAST_N_FRAMES=1

    # ann_file = "/home/ankur/workspace/DexLearn/dexpilot_demos/anno/anno_coco_card_slide_{}.json".format(mode)
    # data_root = "/home/ankur/workspace/DexLearn/dexpilot_demos"

    data_root = "/home/ankur/workspace/DexLearn/dexpilot_demos/sugar_box_lift"
    #ann_file = data_root + "/anno/anno_coco_sugar_box_lift_{}.json".format(mode)
    # Let's do the testing on the training set 
    ann_file = data_root + "/anno/anno_coco_sugar_box_lift_train.json"
    # ann_file = data_root + "/anno/anno_coco_sugar_box_lift_train_traj31-69.json"

    dex_learn_dataset = DexLearnDataset(ann_file=ann_file,
                                                data_root=data_root,
                                                mode=mode,
                                                num_views=NUM_VIEWS,
                                                skip_frames=SKIP_FRAMES,
                                                past_n_frames=0,
                                                unsup_learning=True)


    view_matrices4x4 = get_view_matrices()
    projection_matrices4x4 = prepare_intrinsic_matrices(heatmap_width=40, heatmap_height=30)
    observation_space = dict()
    observation_space['images'] = np.random.rand(1, 3, 240, 320, 3)


    input_arguments =  dict(observation_space=observation_space,
                crop_size=None,
                projection_matrix=projection_matrices4x4,
                view_matrices=view_matrices4x4,
                num_keypoints=10,
                encoder_cls=CustomeEncoder,
                decoder_cls=CustomeDecoder,
                latent_stack=False,
                n_filters=32,
                separation_margin=0.05,
                mean_depth=0.5,
                noise=0.01,
                independent_pred=True,
                decode_first_frame=False, #change this to yes 
                decode_attention=False)


    if mode == 'train':
            
        train_loader = DataLoader(dex_learn_dataset, batch_size=BATCH_SIZE, 
                                            num_workers=12, shuffle=True, 
                                            pin_memory=True, drop_last=True) 

        # for count, elem in enumerate(train_loader):
        #     batch = elem['seg_mask_images']['view_seg_image_0']
        #     import ipdb; ipdb.set_trace();

        model = KeyPointDexPilot3DLearner(**input_arguments)

        # import ipdb; ipdb.set_trace();                                        

        checkpoint_callback = ModelCheckpoint(
            monitor='loss_train',
            dirpath='trained_keypoint3d_models/views_{}/skip_frames_{}'.format(NUM_VIEWS, SKIP_FRAMES),
            filename="sample-loss-{epoch:03d}",
            save_top_k=5,
            save_weights_only=True
            )

        from pytorch_lightning.plugins import DDPPlugin

        trainer = pl.Trainer(
                gpus = [0], 
                max_epochs = EPOCHS,
                accumulate_grad_batches = 1,
                stochastic_weight_avg=True,
                plugins=DDPPlugin(find_unused_parameters=True),
                #distributed_backend='ddp',
                callbacks=[checkpoint_callback]
            )

        trainer.fit(model, train_loader) #, val_loader)
    
    elif mode == 'test':

        model_path = 'trained_keypoint3d_models/views_{}/skip_frames_{}/129/sample-loss-epoch=129.ckpt'.format(NUM_VIEWS, SKIP_FRAMES)
        print('****** testing model ', model_path)


        model = KeyPointDexPilot3DLearner.load_from_checkpoint(model_path,**input_arguments)   
        test_loader = DataLoader(dex_learn_dataset, batch_size=1, 
                                        num_workers=12, shuffle=False, 
                                        pin_memory=True, drop_last=False)


        model = model.cuda()
        model.eval()

        # which_demos = dex_learn_dataset.get_demo_ids()
        # which_demos = which_demos[0] if len(which_demos) == 1 else which_demos[0]
        # print('held-out trajectories: ', which_demos)

        entered = False 

        #TODO: Fix this for multiple held out trajectories
        for count, elem in enumerate(test_loader):
            
            demo_id = elem['img_info']['demo_id'].cpu().detach().numpy()
            img_id  = elem['img_info']['image_id'].cpu().detach().numpy()

            with torch.no_grad():

                source_views = elem['images']
                # source_views = elem['annotated_images']

                imgs0 = source_views["view_0"].clone().detach().unsqueeze(1)
                imgs1 = source_views["view_1"].clone().detach().unsqueeze(1)
                imgs2 = source_views["view_2"].clone().detach().unsqueeze(1)

                # import ipdb; ipdb.set_trace();

                # '''
                # get ground truth keypoints 
                # '''
                # keypoint_annotations = elem['keypoint_annotations']

                # keypts0 = keypoint_annotations['keypoints_view_0'].detach()
                # keypts1 = keypoint_annotations['keypoints_view_1'].detach()
                # keypts2 = keypoint_annotations['keypoints_view_2'].detach()

                # keypoints = torch.cat([keypts0, keypts1, keypts2], dim=1)
                # keypoints = keypoints.cpu().numpy()
                

                sources = torch.cat([imgs0, imgs1, imgs2], dim=1).cuda()

                xyzs_world, loss_dict, heatmaps_hat, xy_normalized = model.forward(sources)

                batch_size, num_cam, _, h, w = sources.shape 

                heatmaps_hat = heatmaps_hat.clone()
                heatmaps_hat = heatmaps_hat.reshape(-1, 30, 40)
                xy = model.learner.heatmap_to_xy(heatmaps_hat)
                xy = xy.reshape(batch_size, num_cam, input_arguments['num_keypoints'], 2)
                xy = (xy+1)/2 
                xy = xy.to('cpu').detach().numpy()
                img = sources.clone().permute(0, 1, 3, 4, 2).cpu().detach().numpy()
        
                canvases = []

                for b in range(0, batch_size):
                    curr_canvas = []
                    for c in range(0, num_cam):
                        curr_img = (img[b, c].copy()*255.0).astype('uint8')
                        
                        for (x, y) in xy[b][c]:
                            colour=(255.0, 0, 0)
                            cv2.circle(curr_img, (int(x*320),int(y*240)), 3, colour, -1)
                        
                        # for (kx, ky) in keypoints[b][c]:
                        #     if kx > 0 and ky > 0:
                        #         colour=(0.0, 255.0, 0)
                        #         cv2.circle(curr_img, (int(kx),int(ky)), 2, colour, -1)

                        curr_canvas.append(curr_img)

                    curr_canvas = np.array(curr_canvas)
                    curr_canvas = np.concatenate(curr_canvas, axis=1)

                    canvases.append(curr_canvas.copy())

                canvases = np.array(canvases)
                # import ipdb; ipdb.set_trace();
                cv2.imwrite('keypoint_results/keypoint_results_{:04d}.png'.format(count), canvases[0][...,::-1])
                print(count, '  files written out of ...   ', dex_learn_dataset.__len__())
                # import ipdb; ipdb.set_trace();



                for b in range(0, batch_size):
                    for c in range(0, num_cam):
                        dirName = 'keypoints2d/sugar_box_lift' + str(demo_id[0]) + '/color{}'.format(c+1)
                        if not os.path.exists(dirName):
                            os.makedirs(dirName)

                        keypts =[]
                        for (x, y) in xy[b][c]:
                            kpt_x, kpt_y = int(x*320),int(y*240)
                            keypts.append([kpt_x, kpt_y])
                        
                        fileName = dirName + '/keypoints2d_{:04d}.txt'.format(img_id[0])
                        np.savetxt(fileName, np.array(keypts))

if __name__ == "__main__":
    fire.Fire(run)


