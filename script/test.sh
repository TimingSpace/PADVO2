#!/bin/bash
cd $PWD/src
traindata=kitti
testdata=kitti

python3 ssvo_test.py --imagelist ../dataset/$traindata/$traindata.image.train --motion ../dataset/$traindata/$traindata.pose.train --imagelist_test ../dataset/$testdata/$testdata.image.test --motion_test ../dataset/$testdata/$testdata.pose.test --ip http://127.0.0.1 --port 8528 --model_load ../saved_model//model_0611_001_035.pt --batch 1 --model 0913_002_091_09
