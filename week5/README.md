# M6 Project: Video Surveillance for Road Traffic Monitoring

## Week 5: Multi-camera tracking

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2019-team3/blob/master/week5/images/output.gif">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2019-team3/blob/master/week5/images/output1.gif">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2019-team3/blob/master/week5/images/output2.gif">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2019-team3/blob/master/week5/images/output3.gif">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2019-team3/blob/master/week5/images/output4.gif">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2019-team3/blob/master/week5/images/output5.gif">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2019-team3/blob/master/week5/images/output5_rect.gif">
</div>

To run the single camera experiments:
```
python main_single_camera.py
```
To run the multi camera experiments:
```
python main_multi_camera.py
```

## Directory structure

```
.
├── L1Stabilizar/             # Matlab implementation of Video Stabilization
├── evaluation/               # IoU, mAP
├── images/                   # Example images and GIFs
├── main_multi_camera.py
├── main_single_camera.py
├── notebooks/                # Necessary notebooks to do some specific tasks for CNNs
├── object_tracking/          # Tracking algorithms
├── pickle/                   
├── processing/                   
├── siamese/                   
└── util/                     # Readers, plots 
```
