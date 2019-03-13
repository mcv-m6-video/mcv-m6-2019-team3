# M6 Project: Video Surveillance for Road Traffic Monitoring

## Week 2

To run the project
```
python main.py
```

We set some flags to change the behaviour of the program:
```
colorspace = None               # [None, 'HSV']
adaptive = True                # Use adaptive or non-adaptive algorithm
use_detections_pkl = True      # Load .pkl instead of compute the intermediate results
export_frames = False           # Save .png images to disk

# Find best alpha and rho
find_best_pairs = False         # Search best alpha for non-adaptive and then best rho for adaptive
grid_search = False       
no_hyperparameter = False
```

## Directory structure

```
.
├── evaluation/     # IoU, mAP
├── main.py
├── processing/     # Gaussian, MOG, GMG
└── util/           # Readers, connected_components, morphological filters, plot 
```
