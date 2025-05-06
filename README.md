# DepthFusion: Depth-Aware Hybrid Feature Fusion for LiDAR-Camera 3D Object Detection

<div align="center">

[Mingqian Ji](https://github.com/Mingqj) <sup>1, </sup>,
Jian Yang <sup>2</sup>,
Shanshan Zhang <sup>1,✉</sup>

This repository represents the official implementation of the paper titled "DepthFusion: Depth-Aware Hybrid Feature Fusion for LiDAR-Camera 3D Object Detection".

![](./resources/pipeline.png)

## News
- **2025.05.06** Support DepthFusion for 3D object detection with LiDAR-camera fusion.

## Main Results
### Nuscenes Detection
| Config                                                                    | mAP        | NDS        | Latency(ms) | FPS  | Model                                                                                          |
| ------------------------------------------------------------------------- | ---------- | ---------- | ---- | ---- | ---------------------------------------------------------------------------------------------- |
| [**DepthFusion-Light**](configs/depthfusion/depthfusion-tiny.py) | 69.8 | 73.3 | 72.4  |13.8 | - | 
| [**DepthFusion-Base**](configs/depthfusion/depthfusion-base.py) | 71.2 | 74.0 | 114.9  |8.7 | - |
| [**DepthFusion-Large**](configs/depthfusion/depthfusion-large.py) | 72.3 | 74.4 | 175.4  |5.7 | - |


## Get Started

#### Installation and Data Preparation

step 1. Please prepare environment as that in [Docker](docker/Dockerfile).

step 2. Prepare bevdet repo by.
```shell script
git clone https://github.com/Mingqj/DepthFusion.git
cd DepthFusion
pip install -v -e .
```

step 3. Prepare nuScenes dataset as introduced in [nuscenes_det.md](docs/en/datasets/nuscenes_det.md) and create the pkl for DepthFusion by running:
```shell
python tools/create_data_bevdet.py
```
step 4. Arrange the folder as:
```shell script
└── nuscenes
    ├── v1.0-trainval (existing)
    ├── sweeps  (existing)
    ├── samples (existing)
    └── gts (new)
```

#### Train model
```shell
# single gpu
python tools/train.py $config
# multiple gpu
./tools/dist_train.sh $config num_gpu
```

#### Test model
```shell
# single gpu
python tools/test.py $config $checkpoint --eval mAP
# multiple gpu
./tools/dist_test.sh $config $checkpoint num_gpu --eval mAP
```

#### Visualize the predicted result.

- Private implementation. (Visualization remotely/locally)

```shell
python tools/test.py $config $checkpoint --format-only --eval-options jsonfile_prefix=$savepath
python tools/analysis_tools/vis.py $savepath/pts_bbox/results_nusc.json
```

## Acknowledgement

This project is not possible without multiple great open-sourced code bases. We list some notable examples below.

- [open-mmlab](https://github.com/open-mmlab)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [BEVerse](https://github.com/zhangyp15/BEVerse)
- [BEVStereo](https://github.com/Megvii-BaseDetection/BEVStereo)
