# DepthFusion


![](./resources/pipeline.png)


## News
- **2025.05.06** Support DepthFusion for 3D object detection with LiDAR-camera fusion.

## Main Results
### Nuscenes Detection
| Config                                                                    | mAP        | NDS        | Latency(ms) | FPS  | Model                                                                                          |
| ------------------------------------------------------------------------- | ---------- | ---------- | ---- | ---- | ---------------------------------------------------------------------------------------------- |
| [**DepthFusion_tiny**](configs/depthfusion/depthfusion-tiny.py) | 69.8 | 73.3 | 72.4  |13.8 | - | 
| [**DepthFusion_Base**](configs/depthfusion/depthfusion-base.py) | 71.2 | 74.0 | 114.9  |8.7 | - |
| [**DepthFusion-Large**](configs/depthfusion/depthfusion-large.py) | 72.3 | 74.4 | 175.4  |5.7 | - |

\# align previous frame bev feature during the view transformation.

Depth: Depth supervised from Lidar as BEVDepth.

Longterm: cat 8 history frame in temporal modeling. 1 by default. 

Stereo: A private implementation that concat cost-volumn with image feature before executing model.view_transformer.depth_net.

The latency includes Network/Post-Processing/Total. Training without CBGS is deprecated.

## Get Started

#### Installation and Data Preparation

step 1. Please prepare environment as that in [Docker](docker/Dockerfile).

step 2. Prepare bevdet repo by.
```shell script
git clone https://github.com/HuangJunJie2017/BEVDet.git
cd BEVDet
pip install -v -e .
```

step 3. Prepare nuScenes dataset as introduced in [nuscenes_det.md](docs/en/datasets/nuscenes_det.md) and create the pkl for BEVDet by running:
```shell
python tools/create_data_bevdet.py
```
step 4. For Occupancy Prediction task, download (only) the 'gts' from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and arrange the folder as:
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

#### Estimate the inference speed of BEVDet

```shell
# with pre-computation acceleration
python tools/analysis_tools/benchmark.py $config $checkpoint --fuse-conv-bn
# 4D with pre-computation acceleration
python tools/analysis_tools/benchmark_sequential.py $config $checkpoint --fuse-conv-bn
# view transformer only
python tools/analysis_tools/benchmark_view_transformer.py $config $checkpoint
```

#### Estimate the flops of BEVDet

```shell
python tools/analysis_tools/get_flops.py configs/bevdet/bevdet-r50.py --shape 256 704
```

#### Visualize the predicted result.

- Private implementation. (Visualization remotely/locally)

```shell
python tools/test.py $config $checkpoint --format-only --eval-options jsonfile_prefix=$savepath
python tools/analysis_tools/vis.py $savepath/pts_bbox/results_nusc.json
```

#### Convert to TensorRT and test inference speed.

```shell
1. install mmdeploy from https://github.com/HuangJunJie2017/mmdeploy
2. convert to TensorRT
python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --fp16 --int8
3. test inference speed
python tools/analysis_tools/benchmark_trt.py $config $engine
```

## Acknowledgement

This project is not possible without multiple great open-sourced code bases. We list some notable examples below.

- [open-mmlab](https://github.com/open-mmlab)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)

Beside, there are some other attractive works extend the boundary of BEVDet.

- [BEVerse](https://github.com/zhangyp15/BEVerse)  for multi-task learning.
- [BEVStereo](https://github.com/Megvii-BaseDetection/BEVStereo)  for stero depth estimation.

## Bibtex

If this work is helpful for your research, please consider citing the following BibTeX entries.

```
@article{huang2023dal,
  title={Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection},
  author={Huang, Junjie and Ye, Yun and Liang, Zhujin and Shan, Yi and Du, Dalong},
  journal={arXiv preprint arXiv:2311.07152},
  year={2023}
}

@article{huang2022bevpoolv2,
  title={BEVPoolv2: A Cutting-edge Implementation of BEVDet Toward Deployment},
  author={Huang, Junjie and Huang, Guan},
  journal={arXiv preprint arXiv:2211.17111},
  year={2022}
}

@article{huang2022bevdet4d,
  title={BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection},
  author={Huang, Junjie and Huang, Guan},
  journal={arXiv preprint arXiv:2203.17054},
  year={2022}
}

@article{huang2021bevdet,
  title={BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View},
  author={Huang, Junjie and Huang, Guan and Zhu, Zheng and Yun, Ye and Du, Dalong},
  journal={arXiv preprint arXiv:2112.11790},
  year={2021}
}
```
