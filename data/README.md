# Dataset Download Instructions

## Important
Do NOT commit raw video data to this repository.
The `data/raw/` folder is in `.gitignore`.

## ShanghaiTech Campus Dataset
1. Visit: https://sviplab.github.io/dataset/campus_dataset.html
2. Download the full dataset (training + testing splits)
3. Extract to: `data/raw/shanghaitech/`

Expected structure:
data/raw/shanghaitech/
├── training/
│   └── videos/   (437 videos)
└── testing/
    ├── videos/   (172 videos)
    └── test_frame_mask/

## UCSD Pedestrian Dataset
1. Visit: http://www.svcl.ucsd.edu/projects/anomaly/dataset.html
2. Download both UCSDped1 and UCSDped2
3. Extract to: `data/raw/ucsd/`