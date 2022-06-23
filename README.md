# CompressedVQA
Deep Learning based Full-reference and No-reference Quality Assessment Models for Compressed UGC Videos

## Description

This is a repository for the models proposed in the paper "Deep Learning based Full-reference and No-reference Quality Assessment Models for Compressed UGC Videos" [arxiv](https://arxiv.org/abs/2106.01111).

The proposed models won **first place on the FR track and second place on the NR track** in [Challenge on Quality Assessment of Compressed UGC Videos hold on IEEE ICME 2021](http://ugcvqa.com/).

## Usage

### Install Requirements

```
opencv-python==4.5.1.48
scipy==1.5.2
torch==1.5.0
torchvision==0.6.0a0+82fd1c8
```
### Download Compressed UGC VQA database

The compressed UGC VQA database can be downloaded from http://ugcvqa.com/.

### Models

You can download the trained model via:

FR VQA model: [Google Drive](https://drive.google.com/file/d/1ohKNe_r0bXBg7qp4vQj0mDT3CwJPHVMM/view?usp=sharing)

NR VQA model: [Google Drive](https://drive.google.com/file/d/1K73padYMgq70zVWVVLIODs9SyIhdgqkT/view?usp=sharing)

### Train

FR VQA model
```shell
CUDA_VISIBLE_DEVICES=0 python -u train_FR.py \
 --database UGCCompressed \
 --model_name UGCVQA_FR_model \
 --conv_base_lr 0.0001 \
 --datainfo json_files/ugcset_dmos.json \
 --videos_dir UGCCompressedVideo/ \
 --epochs 100 \
 --train_batch_size 6 \
 --print_samples 1000 \
 --num_workers 8 \
 --ckpt_path ckpts \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --reults_path results \
 --exp_version 0
```

NR VQA model
```shell
CUDA_VISIBLE_DEVICES=0 python -u train_NR.py \
 --database UGCCompressed \
 --model_name UGCVQA_NR_model \
 --conv_base_lr 0.0001 \
 --datainfo json_files/ugcset_mos.json \
 --videos_dir UGCCompressedVideo/ \
 --epochs 100 \
 --train_batch_size 6 \
 --print_samples 1000 \
 --num_workers 8 \
 --ckpt_path ckpts \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --reults_path results \
 --exp_version 0
```

### Test on the Compressed UGC VQA database
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

### Test on the demo video
#### FR VQA
single scale + test on gpu
```shell
CUDA_VISIBLE_DEVICES=0 python test_FR_demo.py --method_name=single-scale --ref videos/UGC0034_1280x720_30_crf_00.mp4 --dist videos/UGC0034_1280x720_30_crf_22.mp4 --output result.txt --is_gpu
```

using cpu (only work for windows)
```shell
python test_FR_demo.py --method_name=single-scale --ref videos/UGC0034_1280x720_30_crf_00.mp4 --dist videos/UGC0034_1280x720_30_crf_22.mp4 --output result.txt
```

using multi scale strategy
```shell
CUDA_VISIBLE_DEVICES=0 python test_FR_demo.py --method_name=multi-scale --ref videos/UGC0034_1280x720_30_crf_00.mp4 --dist videos/UGC0034_1280x720_30_crf_22.mp4 --output result.txt --is_gpu
```


#### NR VQA
single scale 
```shell
CUDA_VISIBLE_DEVICES=0 python test_NR_demo.py --method_name=single-scale --dist videos/UGC0034_1280x720_30_crf_22.mp4 --output result.txt --is_gpu
```
using cpu
```shell
python test_NR_demo.py --method_name=single-scale --dist videos/UGC0034_1280x720_30_crf_22.mp4 --output result.txt
```
using multi scale strategy
```shell
CUDA_VISIBLE_DEVICES=0 python test_NR_demo.py --method_name=multi-scale --dist videos/UGC0034_1280x720_30_crf_22.mp4 --output result.txt --is_gpu
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