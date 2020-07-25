import torch
import torch.autograd as autograd
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#import egomotionprediction as ep
from utils import *
from data import transformation as tf
import data.data_loader
import loss.loss_functions
import models.VONet
import visualization.my_visualizer as visualizer
import evaluate
from options import parse as parse

torch.manual_seed(100) # random seed generate random number


def main():


    # parameters and flags
    args = parse()
    valid_period = 5
    visualize_training_period = 5
    save_visualize_training_period = 5
    input_batch_size = args.batch_size
    finetune_flag = False
    coor_layer_flag   = False
    pad_flag = False
    with_attention_flag = False
    use_gpu_flag = True
    motion_flag = [2,4]
    #motion_flag = [0,1,2,3,4,5]
    data_balance_flag = False
    color_flag = False
    vis_flag   = False
    no_motion_flag =[d not in motion_flag for d in range(0,6)]
    print(motion_flag,no_motion_flag)
    #camera_parameter=[450,180,225,225,225,90]
    #camera_parameter=[651,262,651,651,320,130]
    camera_parameter=[640,180,640,640,320,90]
    image_size = (camera_parameter[1],camera_parameter[0])

    ################## init model###########################
    model = models.VONet.SPADVONet(coor_layer_flag = coor_layer_flag,color_flag = color_flag)
    model = model.float()
    print(model)
    if use_gpu_flag:
        #model     = nn.DataParallel(model.cuda())
        model     = model.cuda()
    if finetune_flag:
        model.load_state_dict(torch.load(args.model_load))
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(200, 0,50).step)
    print(optimizer)
    #ego_pre = ep.EgomotionPrediction()
    ################### load data####################
    # training data
    motion_files_path = args.motion_path
    path_files_path = args.image_list_path
    print(motion_files_path)
    print(path_files_path)
    # transform
    if color_flag:
        transforms_ = [
                    transforms.Resize(image_size),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    else:
        transforms_ = [
                    transforms.Resize(image_size),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor()]
    kitti_dataset = data.data_loader.SepeDataset(path_to_poses_files=motion_files_path,path_to_image_lists=path_files_path,transform_=transforms_,camera_parameter = camera_parameter,coor_layer_flag = coor_layer_flag,color_flag = color_flag)

    #dataloader = DataLoader(kitti_dataset, batch_size=input_batch_size,shuffle=False ,num_workers=4,drop_last=True,sampler=kitti_dataset.sampler)
    dataloader = DataLoader(kitti_dataset, batch_size=input_batch_size,shuffle=True ,num_workers=4,drop_last=True)
    if data_balance_flag:
        print('data balance by prob')
        dataloader = DataLoader(kitti_dataset, batch_size=input_batch_size,shuffle=False ,num_workers=4,drop_last=True,sampler=kitti_dataset.sampler)
    else:
        print('no data balance')
    dataloader_vis = DataLoader(kitti_dataset, batch_size=input_batch_size,shuffle=False ,num_workers=4,drop_last=True)
    # testing data
    motion_files_path_test = args.motion_path_test
    path_files_path_test = args.image_list_path_test
    print(motion_files_path_test)
    print(path_files_path_test)
    # transform
    if color_flag:
        transforms_ = [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    else:
        transforms_ = [
                    transforms.Resize(image_size),
                    transforms.ToTensor()]



    kitti_dataset_test = data.data_loader.SepeDataset(path_to_poses_files=motion_files_path_test,path_to_image_lists=path_files_path_test,transform_=transforms_,camera_parameter = camera_parameter,norm_flag=1,coor_layer_flag = coor_layer_flag,color_flag=color_flag)

    dataloader_vid = DataLoader(kitti_dataset_test, batch_size=input_batch_size,shuffle=False ,num_workers=4,drop_last=True)
    print(len(kitti_dataset),len(kitti_dataset_test))


    epoch_loss_visu_mean=0
    epoch_loss_eval_mean=0
    if vis_flag:
        vis = visualizer.Visualizer(args.visdom_ip,args.visdom_port)
        print('vis',args.visdom_ip,args.visdom_port)
    training_loss_data = open('../saved_new/'+args.model_name+'_training.loss','a')
    testing_loss_data = open('../saved_new/'+args.model_name+'_testing.loss','a')
    training_ate_data = open('../saved_new/'+args.model_name+'_training.ate','a')
    testing_ate_data = open('../saved_new/'+args.model_name+'_testing.ate','a')
     ################## training   #######################
    for epoch in range(101):
        epoch_loss = 0
        result = []
        result = np.array(result)
        model.train()
        for i_batch, sample_batched in enumerate(dataloader):
            batch_loss,result = pad_update(model,sample_batched,with_attention_flag=with_attention_flag,pad_flag=pad_flag,motion_flag=motion_flag)
            temp_f = weighted_mean_motion(result)
            epoch_loss += batch_loss
            if vis_flag:
                vis.plot_current_errors(epoch,i_batch*input_batch_size
                        /len(kitti_dataset),batch_loss.data)
            print(epoch,'******',i_batch,'/',len(dataloader),'*******',batch_loss.item())
            batch_loss.backward()
            optimizer.step()
        data_length = len(kitti_dataset)//input_batch_size*input_batch_size
        epoch_loss_mean = epoch_loss*input_batch_size/data_length
        if vis_flag:
            vis.plot_epoch_current_errors(epoch,epoch_loss_mean.data)
        lr_scheduler.step()

    ####Visualization Path###############################################################
        with torch.no_grad():
            if epoch%valid_period==0:
                model.eval()
                forward_visual_result = []
                ground_truth = []
                epoch_loss_visu = 0
                for i_batch, sample_batched in enumerate(dataloader_vis):
                    print('visu************** i_batch',i_batch,'******************')
                    model.zero_grad()
                    batch_loss,result = pad_update(model,sample_batched,with_attention_flag = with_attention_flag,pad_flag = pad_flag,motion_flag=motion_flag)
                    #batch_loss.backward()
                    batch_loss.detach_()
                    training_loss_data.write(str(batch_loss.cpu().data.tolist())+'\n')
                    training_loss_data.flush()
                    epoch_loss_visu+=batch_loss
                    temp_f = weighted_mean_motion(result)
                    gt_f_12 = sample_batched['motion_f_01'].numpy()
                    forward_visual_result = np.append(forward_visual_result,temp_f)
                    ground_truth = np.append(ground_truth,gt_f_12)
                    if i_batch> 100:
                        break
                data_length = len(kitti_dataset)//input_batch_size*input_batch_size
                epoch_loss_visu_mean = epoch_loss_visu*input_batch_size/data_length
                forward_visual_result = forward_visual_result.reshape(-1,6)*kitti_dataset.motion_stds
                forward_visual_result[:,no_motion_flag]=0
                #ground_truth = ground_truth.reshape(data_length,6)*kitti_dataset.motion_stds+kitti_dataset.motion_means
                ground_truth = ground_truth.reshape(-1,6)*kitti_dataset.motion_stds

                forward_visual_result_m = tf.eular2pose(forward_visual_result)
                ground_truth_m          = tf.eular2pose(ground_truth)
                #forward_visual_result_m = tf.ses2poses(forward_visual_result)
                #ground_truth_m          = tf.ses2poses(ground_truth)
                rot_train,tra_train   = evaluate.evaluate(ground_truth_m,forward_visual_result_m)
                training_ate_data.write(str(np.mean(tra_train))+' '+ str(np.mean(rot_train))+'\n')
                training_ate_data.flush()
                if vis_flag:
                    vis.plot_path_with_gt(forward_visual_result_m,ground_truth_m,5,'training set forward')
                torch.save(model.state_dict(), '../saved_model/model_'+args.model_name+'_'+str(epoch).zfill(3)+'.pt')
        ####Vilidation Path###############################################################
                model.eval()
                forward_visual_result = []
                backward_visual_result = []
                ground_truth = []
                epoch_loss_eval = 0
                forward_visual_opti = []
                for i_batch, sample_batched in enumerate(dataloader_vid):
                    #opti vis
                    print('test************** i_batch',i_batch,'******************')
                    batch_loss_eval,predicted_result_eval = pad_update(model,sample_batched,with_attention_flag=with_attention_flag,pad_flag = pad_flag,motion_flag=motion_flag)
                    batch_loss_eval.detach_()
                    testing_loss_data.write(str(batch_loss_eval.cpu().data.tolist())+'\n')
                    testing_loss_data.flush()
                    epoch_loss_eval+=batch_loss_eval
                    temp_f = weighted_mean_motion(predicted_result_eval)
                    gt_f_12 = sample_batched['motion_f_01'].numpy()
                    forward_visual_result = np.append(forward_visual_result,temp_f)
                    ground_truth = np.append(ground_truth,gt_f_12)
                data_length = len(kitti_dataset_test)//input_batch_size*input_batch_size
                epoch_loss_eval_mean=epoch_loss_eval*input_batch_size/data_length
                forward_visual_result = forward_visual_result.reshape(data_length,6)*kitti_dataset_test.motion_stds
                forward_visual_result[:,no_motion_flag]=0
                #ground_truth = ground_truth.reshape(data_length,6)*kitti_dataset_test.motion_stds+kitti_dataset_test.motion_means
                ground_truth = ground_truth.reshape(data_length,6)*kitti_dataset_test.motion_stds

                forward_visual_result_m = tf.eular2pose(forward_visual_result)
                ground_truth_m          = tf.eular2pose(ground_truth)
                #forward_visual_result_m = tf.ses2poses(forward_visual_result)
                #ground_truth_m          = tf.ses2poses(ground_truth)
                rot_eval,tra_eval   = evaluate.evaluate(ground_truth_m,forward_visual_result_m)

                testing_ate_data.write(str(np.mean(tra_eval))+' '+ str(np.mean(rot_eval))+'\n')
                testing_ate_data.flush()
                if vis_flag:
                    vis.plot_two_path_with_gt(forward_visual_result_m,forward_visual_result_m,ground_truth_m,10,'testing set forward')
                    vis.plot_epoch_training_validing(epoch,epoch_loss_visu_mean.detach().cpu().numpy(),epoch_loss_eval_mean.detach().cpu().numpy())
                    vis.plot_epoch_training_validing_2(epoch,np.mean(tra_train),np.mean(tra_eval),22)



def weighted_mean_motion(predicted_result):
    predict_f_12 = predicted_result[0]
    result = np.zeros((predict_f_12.shape[0],6))
    if True:
        predict_b_21 = predicted_result[1]
        predict_f_12 = torch.stack((predict_f_12,-predict_b_21))
        predict_f_12 = predict_f_12.mean(0)
    temp_f = predict_f_12.cpu().data.numpy()
    result[:,[2,4]] = temp_f
    return result


def pad_update(model,sample_batched,with_attention_flag=False,use_gpu_flag=True,pad_flag=False,motion_flag=[0,1,2,3,4,5]):
    model.zero_grad()
    input_batch_images_f_12 = autograd.Variable(sample_batched['image_f_01'])
    input_batch_motions_f_12 = autograd.Variable(sample_batched['motion_f_01'])
    input_batch_images_b_21 = autograd.Variable(sample_batched['image_b_10'])
    input_batch_motions_b_21 = autograd.Variable(sample_batched['motion_b_10'])
    if use_gpu_flag:
        input_batch_images_f_12 = input_batch_images_f_12.cuda()
        input_batch_motions_f_12 = input_batch_motions_f_12.cuda()
        input_batch_images_b_21 = input_batch_images_b_21.cuda()
        input_batch_motions_b_21 = input_batch_motions_b_21.cuda()
    predict_f_12 = model(input_batch_images_f_12)
    predict_b_21 = model(input_batch_images_b_21)
    result=[predict_f_12,predict_b_21]#,predict_f_13,att_f_13,predict_b_31,att_b_31,predict_f_23,att_f_23,predict_b_32,att_b_32]
    batch_loss = loss.loss_functions.GroupGVLoss(predict_f_12, input_batch_motions_f_12, \
        predict_b_21,input_batch_motions_b_21)
    return batch_loss,result

if __name__ == '__main__':
    main()





'''
def optimized_motion(predicted_result,kitti_dataset,ego_pre):
    predict_f_12 = predicted_result[0]
    predict_b_21 = predicted_result[2]
    temp_f = predict_f_12.cpu().data.numpy()
    temp_b = predict_b_21.cpu().data.numpy()
    temp_f = np.transpose(temp_f,(0,2,3,1))
    temp_f = temp_f*kitti_dataset_test.motion_stds+kitti_dataset_test.motion_means
    temp_f = np.transpose(temp_f,(0,3,1,2))

    temp_b = np.transpose(temp_b,(0,2,3,1))
    temp_b = temp_b*kitti_dataset_test.motion_stds+kitti_dataset_test.motion_means
    temp_b = np.transpose(temp_b,(0,3,1,2))

    quat_f = tf.sos2quats(temp_f[:,3:6,:,:])
    quat_b = tf.sos2quats(temp_b[:,3:6,:,:])
    quats = np.concatenate((quat_f,quat_b),axis=1)
    #print(quats.shape)

    trans_f = np.transpose(temp_f[:,0:3,:,:].reshape(temp_f.shape[0],3,temp_f.shape[2]*temp_f.shape[3]),(0,2,1))
    trans_b = np.transpose( temp_b[:,0:3,:,:].reshape(temp_b.shape[0],3,temp_b.shape[2]*temp_b.shape[3]),(0,2,1))
    trans = np.concatenate((trans_f,trans_b),axis=1)
    optimized_motion =[]
    for i_q in range(0,quats.shape[0]):
        ego = ego_pre.predict_patch(quats[i_q,:,:],trans[i_q,:,:])
        optimized_motion = np.append(forward_visual_opti,tf.SE2se(np.array(ego)))
    return optimized_motion
'''


