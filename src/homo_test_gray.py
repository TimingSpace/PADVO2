import torch
import torch.autograd as autograd
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import argparse

from utils import *
from data import transformation as tf
import data.data_loader
import loss.loss_functions
import models.VONet
import visualization.my_visualizer as visualizer
torch.manual_seed(100) # random seed generate random number

parser = argparse.ArgumentParser(description='')
parser.add_argument('--optimizer', dest='optimization_method', default='sgd', help='optimization method')
parser.add_argument('--result', dest='result_path', default='result/00_predict_posang.txt', help='predict result path')
parser.add_argument('--imagelist', dest='image_list_path', default='dataset/kitti_image.txt', help='image list path')
parser.add_argument('--motion', dest='motion_path', default='dataset/kitti_pose.txt', help='motion path')
parser.add_argument('--model', dest='model_name', default='adam_20', help='model name')
parser.add_argument('--batch', dest='batch_size',type=int, default=200, help='batch size')
parser.add_argument('--motion_test', dest='motion_path_test', default='dataset/kitti_pose_test.txt', help='test motion path')
parser.add_argument('--imagelist_test', dest='image_list_path_test', default='dataset/kitti_image_test.txt', help='test image list path')
parser.add_argument('--port', dest='visdom_port', default='8098', help='visdom port')
parser.add_argument('--ip', dest='visdom_ip', default='http://128.237.217.183', help='visdom port')
parser.add_argument('--mean_std_path', dest='mean_std_path', default='my_utils/mean_std.txt', help='visdom port')
args = parser.parse_args()
use_gpu_flag = True
if __name__ == '__main__':
    #with torch.cuda.device(0):
    valid_period = 5
    visualize_training_period = 5
    save_visualize_training_period = 5
    input_batch_size = args.batch_size
    finetune_flag = True
    ################## init model###########################
    model = models.VONet.VONet()
    model = model.float()
    #weights = autograd.Variable(torch.FloatTensor([1,1,1,1,1,1]))
    model.load_state_dict(torch.load('saved_model/model_adam_20_299.pt'))
    # normalization parameter
    # model and optimization
    if use_gpu_flag:
        model = nn.DataParallel(model.cuda())
        #weights = nn.DataParallel(weights.cuda())
        print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(300, 0,50).step)
    print(optimizer)

    ################### load data####################
# training data
    motion_files_path = args.motion_path
    path_files_path = args.image_list_path
    print(motion_files_path)
    print(path_files_path)
    # transform
    transforms_ = [
                transforms.Resize((188,620)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    kitti_dataset = data.data_loader.SepeDataset(path_to_poses_files=motion_files_path,path_to_image_lists=path_files_path,transform_=transforms_)


    # testing data
    motion_files_path_test = args.motion_path_test
    path_files_path_test = args.image_list_path_test
    print(motion_files_path_test)
    print(path_files_path_test)
    # transform
    transforms_ = [
                transforms.Resize((188,620)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]


    kitti_dataset_test = data.data_loader.SepeDataset(path_to_poses_files=motion_files_path_test,path_to_image_lists=path_files_path_test,transform_=transforms_)

    dataloader_vid = DataLoader(kitti_dataset_test, batch_size=input_batch_size,shuffle=False ,num_workers=4,drop_last=True)

    vis = visualizer.Visualizer(args.visdom_ip,args.visdom_port)
    ####Vilidation Path###############################################################
    model.eval()
    forward_visual_result = []
    ground_truth = []
    sum_loss_epoch = 0
    for i_batch, sample_batched in enumerate(dataloader_vid):
        #print('************** i_batch',i_batch,'******************')
        model.zero_grad()
        input_batch_images_f_12 = autograd.Variable(sample_batched['image_f_01'])
        input_batch_images_f_23 = autograd.Variable(sample_batched['image_f_12'])
        input_batch_images_f_13 = autograd.Variable(sample_batched['image_f_02'])
        if use_gpu_flag:
            input_batch_images_f_12 = input_batch_images_f_12.cuda()
            input_batch_images_f_13 = input_batch_images_f_13.cuda()
            input_batch_images_f_23= input_batch_images_f_23.cuda()
        predict_f_12 = model(input_batch_images_f_12)
        predict_f_13 = model(input_batch_images_f_13)
        predict_f_23 = model(input_batch_images_f_23)
        temp_f = predict_f_12.cpu().data.numpy()

        gt_f_12 = sample_batched['motion_f_01'].numpy()
        forward_visual_result = np.append(forward_visual_result,temp_f)
        ground_truth = np.append(ground_truth,gt_f_12)
    data_length = len(kitti_dataset_test)//input_batch_size*input_batch_size
    forward_visual_result = forward_visual_result.reshape(data_length,6)*kitti_dataset.motion_stds+kitti_dataset.motion_means
    ground_truth = ground_truth.reshape(data_length,6)*kitti_dataset_test.motion_stds+kitti_dataset_test.motion_means
   # forward_visual_result = forward_visual_result*motion_stds_np+motion_means_np
    forward_visual_result_m = tf.ses2poses(forward_visual_result)
    ground_truth_m          = tf.ses2poses(ground_truth)
    vis.plot_path_with_gt(ground_truth_m,forward_visual_result_m,6,'testing set forward')




