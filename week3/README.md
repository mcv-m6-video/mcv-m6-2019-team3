# M6 Project: Video Surveillance for Road Traffic Monitoring

## Week 3: Mask-RCNN and tracking

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2019-team3/blob/master/week3/images/mask-rcnn.gif">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2019-team3/blob/master/week3/images/tracking.gif">
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
├── object_tracking/          # Tracking algorithms
├── main.py
├── Mask_R-CNN_demo.ipynb     # Predict frames using Mask R-CNN
└── util/                     # Readers, plots 
```
