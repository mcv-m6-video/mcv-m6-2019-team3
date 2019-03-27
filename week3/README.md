# M6 Project: Video Surveillance for Road Traffic Monitoring

## Week 3: CNN and tracking

<div align="center">
  Mask R-CNN:
  <img src="https://github.com/mcv-m6-video/mcv-m6-2019-team3/blob/master/week3/images/mask-rcnn_off_the_shelf.gif">
  YOLO:
  <img src="https://github.com/mcv-m6-video/mcv-m6-2019-team3/blob/master/week3/images/yolo.gif">
  Tracking by Overlap:
  <img src="https://github.com/mcv-m6-video/mcv-m6-2019-team3/blob/master/week3/images/tracking_overlap.gif">
</div>

To run the project
```
python main.py
```

We set some flags to change the behaviour of the program:
```
use_pkl = True                  # Load .pkl instead of compute the intermediate results
display_frames = False          # Show plots
export_frames = False           # Save .png images to disk
```

## Directory structure

```
.
├── evaluation/               # IoU, mAP
├── images/                   # Example images and GIFs
├── main.py
├── notebooks/                # Necessary notebooks to do some specific tasks for CNNs
├── object_tracking/          # Tracking algorithms
└── util/                     # Readers, plots 
```
