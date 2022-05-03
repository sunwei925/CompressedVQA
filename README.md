# CompressedVQA
Deep Learning based Full-reference and No-reference Quality Assessment Models for Compressed UGC Videos

## Description

This is a repository for the models proposed in the paper "Deep Learning based Full-reference and No-reference Quality Assessment Models for Compressed UGC Videos".

The proposed models won first place on the FR track and second place on the NR track in Challenge on Quality Assessment of Compressed UGC Videos hold on ICME 2021.

## Usage

### Install Requirements

```
opencv-python==4.5.1.48
scipy==1.5.2
torch==1.5.0
torchvision==0.6.0a0+82fd1c8
```
###Download Compressed UGC VQA database

The compressed UGC VQA database can be downloaded from http://ugcvqa.com/.

### Models

You can download the trained model via:

FR VQA model: [Google Drive](https://drive.google.com/file/d/1ohKNe_r0bXBg7qp4vQj0mDT3CwJPHVMM/view?usp=sharing)

NR VQA model: [Google Drive](https://drive.google.com/file/d/1K73padYMgq70zVWVVLIODs9SyIhdgqkT/view?usp=sharing)

### Train

We will release the training code later.

### Test
FR VQA model
```shell
CUDA_VISIBLE_DEVICES=0 python -u test_FR.py \
 --database UGCCompressed \
 --model_name UGCVQA_FR_model \
 --num_workers 16 \
 --multi_gpu False \
 --datainfo_test json_files/ugcset_dmos.json \
 --videos_dir_test UGCCompressedVideo/ \
 --trained_model ckpts/UGCVQA_FR_model.pth \
```
NR VQA model
```shell
CUDA_VISIBLE_DEVICES=0 python -u test_NR.py \
 --database UGCCompressed \
 --model_name UGCVQA_NR_model \
 --num_workers 16 \
 --multi_gpu False \
 --datainfo_test json_files/ugcset_mos.json \
 --videos_dir_test UGCCompressedVideo/ \
 --trained_model ckpts/UGCVQA_NR_model.pth \
```

## Citation

If you find this code is useful for your research, please cite:
```
@inproceedings{sun2021deep,
  title={Deep learning based full-reference and no-reference quality assessment models for compressed ugc videos},
  author={Sun, Wei and Wang, Tao and Min, Xiongkuo and Yi, Fuwang and Zhai, Guangtao},
  booktitle={2021 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```
