#!/bin/bash
#cd /home/david/Program/SingleShotVO/src # path to current source 
cd $PWD/src # path to current source 
traindata=kitti
testdata=kitti
vis=8520
tag=0726_001

python3 ssvo_train.py --imagelist ../dataset/$traindata/$traindata.image.train --motion ../dataset/$traindata/$traindata.pose.train --imagelist_test ../dataset/$testdata/$testdata.image.test --motion_test ../dataset/$testdata/$testdata.pose.test --ip http://127.0.0.1 --port $vis --model $tag --batch 10 --model_load test_save_model_2.pt  --no_pad --no_att --coor

#cd /home/wangxiangwei/Program/SingleShotVO/src
#traindata=kitti
#testdata=kitti

#python3 ssvo_train.py --imagelist ../dataset/$traindata/$traindata.train.image --motion ../dataset/$traindata/$traindata.train.pose --imagelist_test ../dataset/$testdata/$testdata.test.image --motion_test ../dataset/$testdata/$testdata.test.pose --ip http://128.237.141.242 --port 8524 --model 0913_001 --batch 60 --model_load ../saved_model//model_0905_002_030.pt
