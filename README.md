# [IJCV Article] *AdaFuse*: Adaptive Multiview Fusion for Accurate Human Pose Estimation in the Wild

## Resources

Paper: [(arXiv:2010.13302)](https://arxiv.org/abs/2010.13302)

*Occlusion-Person* Dataset: [(GitHub)](https://github.com/zhezh/occlusion_person)

## Install & Data Preparation

Clone this repo, and install the dependencies.

```
git clone https://github.com/zhezh/adafuse-3d-human-pose.git adafuse
cd adafuse
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
pip install -r requirements.txt
```
We use Pytorch 1.2.0 with Ubuntu 16.04 LTS (CUDA 10.1). Other versions, e.g. Pytorch > 1.0, CUDA > 9.0 and Ubuntu 18, should also be applicable.

The `adafuse` directory will be referred as {POSE_ROOT}.

### Pretrained Models
Download pytorch pretrained models. Please download them under ${POSE_ROOT}/models, and make them look like this:

```
${POSE_ROOT}/models
└── pytorch
    ├── adafuse
    │   ├── h36m_4view.pth.tar
    │   └── occlusion_person_8view.pth.tar
    ├── pose_backbone
    │   ├── h36m_4view_d87025.pth.tar
    │   └── occlusion_person_8view_c20e11.tar
    ├── pose_coco (Optional)
    │   ├── pose_resnet_152_384x288.pth.tar
    │   ├── pose_resnet_50_256x192.pth.tar
```
They can be downloaded from the this [link](https://dllabml-my.sharepoint.com/:f:/g/personal/research_dllabml_onmicrosoft_com/EmTUxlEP0XROlkUFetdcQSYBhbsBbwk0JLDhaMH39UqVbw?e=ei4RfJ)



### Human3.6M
Please follow [CHUNYUWANG/H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox) to prepare the data.

> Note that we have **NO** permission to redistribute the Human3.6M data. Please do NOT ask us for a copy of Human3.6M dataset.



### Occlusion-Person

Please follow [zhezh/occlusion_person](https://github.com/zhezh/occlusion_person) to prepare the data.



### MPII (Optional)
**For MPII data**, please refer to [microsoft/multiview-human-pose-estimation-pytorch](https://github.com/microsoft/multiview-human-pose-estimation-pytorch/blob/master/INSTALL.md#data-preparation).

## Evaluate
Make sure you are in the {POSE_ROOT} directory.

**Human3.6M**

```bash
python run/adafuse/adafuse_main.py --cfg experiments/h36m/h36m_4view.yaml --evaluate true
```

**Occlusion-Person**

```bash
python run/adafuse/adafuse_main.py --cfg experiments/occlusion_person/occlusion_person_8view.yaml --evaluate true
```

## Results
**Human3.6M**
```
MPJPE summary: j3d_NoFuse 22.94
MPJPE summary: j3d_HeuristicFuse 21.02
MPJPE summary: j3d_ScoreFuse 20.14
MPJPE summary: j3d_ransac 21.77
MPJPE summary: j3d_AdaFuse 19.54
```

**Occlusion-Person**

```
MPJPE summary: j3d_NoFuse 48.16
MPJPE summary: j3d_HeuristicFuse 18.02
MPJPE summary: j3d_ScoreFuse 14.97
MPJPE summary: j3d_ransac 15.40
MPJPE summary: j3d_AdaFuse 12.56
```

## Train
**Human3.6M**

```bash
python run/adafuse/adafuse_main.py --cfg experiments/h36m/h36m_4view.yaml --runMode train
```

**Occlusion-Person**

```bash
python run/adafuse/adafuse_main.py --cfg experiments/occlusion_person/occlusion_person_8view.yaml --runMode train
```

## Train 2D pose backbone
We provide pre-trained 2D pose backbone parameters in the `models/pytorch/pose_backbone` directory, if you would like to train them by yourself, please follow below instructions.

Firstly, follow previous instructions to prepare the *Pretrained Models* and *MPII* dataset which are labeled as "*Optional*".

Then,

For **Human3.6M** (MPII is needed to augment Human3.6M training data),

```bash
python run/pose2d/train.py --cfg experiments/pose2d/h36m.yaml
```

For **Occlusion-Person**,

```bash
python run/pose2d/train.py --cfg experiments/pose2d/occlusion_person.yaml
```

Finally, replace the attribute `NETWORK/PRETRAINED` in AdaFuse yaml config (e.g. in `experiments/h36m/h36m_4view.yaml`) with newly obtained model path (e.g. `output/h36m_res50.pth.tar`).

## Citation
```
@article{zhang2020adafuse,
      title={AdaFuse: Adaptive Multiview Fusion for Accurate Human Pose Estimation in the Wild}, 
      author={Zhe Zhang and Chunyu Wang and Weichao Qiu and Wenhu Qin and Wenjun Zeng},
      year={2020},
      journal={IJCV},
      publisher={Springer},
      pages={1--16},
}
```

