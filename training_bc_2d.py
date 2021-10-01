from PIL import Image
import numpy as np 
import time 

import torch
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from einops import rearrange
from einops.layers.torch import Rearrange

import fire 

import torch.nn as nn 
import torch.nn.functional as F

LR = 3e-4 #1e-3
softmax_temp = 1

from networks import BaseCNN, ImplicitCNN
import cv2 

# pytorch lightning module
class ImplicitBC_2d_Learner(pl.LightningModule):
    def __init__(self, **kwargs):
        super(ImplicitBC_2d_Learner, self).__init__()
        # self.save_hyperparameters()
        # self.network_type = 'explicit'
        self.network_type = 'implicit'

        self.lr = LR

        self.batch_iter = 0 
        if self.network_type == 'explicit':
            self.learner = BaseCNN(in_channels=3)
            self.loss = torch.nn.MSELoss()
        else:
            self.learner = ImplicitCNN(in_channels=3)
            self.loss = torch.nn.CrossEntropyLoss()

    
    def forward(self, images, coords=None):
        if self.network_type == 'explicit':
            return self.learner(images)
        else:
            return self.learner(images, coords=coords)

    def training_step(self, batch, batch_idx):    
        
        source_views = batch['images']
        xy_ground_truth_normalised = batch['normalised_annotations']
        
        if self.network_type == 'explicit':
            xy_pred_normalised = self.forward(source_views)
            loss = self.loss(xy_pred_normalised, xy_ground_truth_normalised)
        else:
            
            xy_negative_samples_normalised = batch['normalised_negatives'][0]
            xy_pos_neg = torch.cat((xy_ground_truth_normalised, xy_negative_samples_normalised), dim=0)
            perm = torch.randperm(xy_pos_neg.shape[0]).cuda()
            xy_pos_neg = xy_pos_neg[perm]

            idx = (perm==0).nonzero()[0]
            # import ipdb; ipdb.set_trace();

            energy = self.forward(source_views, xy_pos_neg)
            energy = energy * -1.0 / softmax_temp 
            target = torch.zeros(1, dtype=torch.long, device=energy.device)
            loss = self.loss(energy, idx) #target)
        
        self.log('loss_train', loss)
        self.log('epoch', self.current_epoch)

        if self.network_type == 'explicit':
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu=False,
            using_native_amp=False,
            using_lbfgs=False,
        ):

        for pg in optimizer.param_groups:
            # pg['lr'] = 0.5**( np.floor(epoch/200.0) )* self.lr #decrease the learning rate from 1e-4 to 1e-5 over the course of 1000 epochs
            pg['lr'] = 0.7**( np.floor(epoch/200.0) )* self.lr #decrease the learning rate from 1e-4 to 1e-5 over the course of 1000 epochs
            # print(pg['lr'], 'learning rate')
        # update params
        optimizer.step(closure=optimizer_closure)



from omegaconf import DictConfig, OmegaConf
import hydra

# @hydra.main(config_path=".", config_name="hydra_config")
# def run(cfg):
def run(mode):

    # mode = cfg.mode

    from torch.utils.data import DataLoader, Dataset
    from data_loader import ImplicitBCDataset_2D

    from pytorch_lightning.callbacks import ModelCheckpoint
    import cv2

    EPOCHS = 2000

    if mode == 'train':

        implicit_bc_dataset_2d = ImplicitBCDataset_2D(dataset_size=10,
                                                  img_size=(128, 128))
            
        train_loader = DataLoader(implicit_bc_dataset_2d, batch_size=1, 
                                            num_workers=12, shuffle=True, 
                                            pin_memory=True, drop_last=False) 

        model = ImplicitBC_2d_Learner()

        checkpoint_callback = ModelCheckpoint(
            monitor='loss_train',
            dirpath='trained_models',
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
                # distributed_backend='ddp',
                callbacks=[checkpoint_callback]
            )

        trainer.fit(model, train_loader) #, val_loader)
    
    elif mode == 'test':

        model_path = 'trained_models/sample-loss-epoch=1597.ckpt'
        print('****** testing model ', model_path)

        implicit_bc_dataset_2d = ImplicitBCDataset_2D(dataset_size=1000,
                                                      img_size=(128, 128), 
                                                      fixed_seed=True, 
                                                      mode=mode)


        model = ImplicitBC_2d_Learner.load_from_checkpoint(model_path)   
        test_loader = DataLoader(implicit_bc_dataset_2d, batch_size=1, 
                                        num_workers=12, shuffle=False, 
                                        pin_memory=True, drop_last=False)


        model = model.cuda()
        model.eval()

        preds = []

        if model.network_type == 'implicit':

            good_preds = 0 

            for count, elem in enumerate(test_loader):

                with torch.no_grad():

                    images = elem['images'].cuda()

                    x = np.random.rand(16384)
                    y = np.random.rand(16384)
                    
                    cv_np_img = images.clone().cpu().detach().numpy()
                    cv_np_img = rearrange(cv_np_img, 'b c h w -> b h w c')
                    cv_np_img = cv_np_img[0]
                    
                    for i in range(0, 10):

                        coords = np.array([2*x-1, 2*y-1], dtype=np.float32)
                        coords = Rearrange('c b -> b c')(coords)
                        coords = torch.from_numpy(coords).cuda()

                        energy = model(images, coords)
                        energy = energy * -1.0 / softmax_temp
                        probs  = torch.nn.Softmax(dim=1)(energy)
                        
                        sample_ind = torch.multinomial(probs.squeeze(0), 16384, replacement=True)
                        new_coords = coords[sample_ind] 
                        new_coords = new_coords.cpu().detach().numpy()
                        
                        x = (new_coords[:, 0]+1)/2 * 128 + np.random.normal(0, 0.33 * 0.9 ** i)
                        y = (new_coords[:, 1]+1)/2 * 128 + np.random.normal(0, 0.33 * 0.9 ** i)

                        x = x / 128.0
                        y = y / 128.0

                        x = np.minimum(np.maximum(x, 0.0),1.0)
                        y = np.minimum(np.maximum(y, 0.0),1.0)

                        top_k = torch.topk(probs, 10)

                        indices = top_k.indices
                        values  = top_k.values
                        values  = values / torch.sum(values)
                        values  = values.reshape(-1, 1)
                        top_k_coords = coords[indices][0]
                        
                        prediction = torch.sum(values * top_k_coords, dim=0) #top_k_coords[0] #torch.sum(values * top_k_coords, dim=0)

                        # cv_np_img = images.clone().cpu().detach().numpy()
                        # cv_np_img = rearrange(cv_np_img, 'b c h w -> b h w c')
                        # cv_np_img = cv_np_img[0]

                        # cpu_coords = new_coords #new_coords.clone().cpu().detach().numpy()
                        # for xx, yy in cpu_coords:
                        #     yy = ( yy + 1 ) / 2 * cv_np_img.shape[0]# + np.random.normal(0, 0.33)
                        #     xx = ( xx + 1 ) / 2 * cv_np_img.shape[1]# + np.random.normal(0, 0.33)
                        #     cv_np_img = cv2.circle(np.ascontiguousarray(cv_np_img), (int(xx), int(yy)), radius=1, color=(0,255,0), thickness=-1)

                        # cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
                        # cv2.imshow('Align Example', cv_np_img[...,::-1])
                        # key = cv2.waitKey(1)

                        # time.sleep(3)

                        # # Press esc or 'q' to close the image window
                        # if key & 0xFF == ord('q') or key == 27:
                        #     cv2.destroyAllWindows()
                        #     break

                        
                        

                    # prediction = torch.sum(values * top_k_coords, dim=0)
                    xy_pred = prediction.cpu().detach().numpy()
                    xy_pred = (xy_pred+1)/2.0 * 128 
                    xy_pred = (xy_pred+0.5).astype(int)

                    xy_gt = elem['annotations'].cpu().detach().numpy()

                    err = np.linalg.norm(xy_pred - xy_gt)
                    # err = err * err / 2.0 

                    if err <= 1.0:
                        good_preds += 1 
                        print(err, count, good_preds)

                    preds.append([xy_gt[0][0], xy_gt[0][1], err])


            preds = np.array(preds)
            np.savetxt('vanilla_coords_pred.txt', preds)

        else:

            for count, elem in enumerate(test_loader):

                with torch.no_grad():

                    images = elem['images'].cuda()
                    xy_pred_normalised = model(images)
                    xy_pred = xy_pred_normalised.cpu().detach().numpy()
                    xy_pred = (xy_pred+1)/2.0 * 128 

                    xy_gt = elem['annotations'].cpu().detach().numpy()

                    err = np.linalg.norm(xy_pred - xy_gt)

                    if err <= 1.0:
                        print(err)

                    preds.append([xy_gt[0][0], xy_gt[0][1], err])


            preds = np.array(preds)
            np.savetxt('vanilla_coords_pred.txt', preds)


if __name__ == "__main__":
    fire.Fire(run)
    # run()


