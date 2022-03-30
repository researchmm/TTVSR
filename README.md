# TTVSR (CVPR2022, Oral)
This is the official PyTorch implementation of the paper [Learning Trajectory-Aware Transformer for Video Super-Resolution](Arxiv).

## Contents
- [Introduction](#introduction)
  - [Contribution](#contribution)
  - [Overview](#overview)
  - [Visual](#Visual)
- [Requirements and dependencies](#requirements-and-dependencies)
- [Model and Results](#model-and-results)
- [Dataset](#dataset)
- [Test](#test)
- [Train](#train)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)
- [Contact](#contact)


## Introduction
We proposed an approach named TTVSR to study video super-resolution by leveraging long-range frame dependencies. TTVSR introduces Transformer architectures in video super-resolution tasks and formulates video frames into pre-aligned trajectories of visual tokens to calculate attention along trajectories.
<img src="./fig/teaser_TTVSR.png" width=70%>

### Contribution
We propose a novel trajectory-aware Transformer, which is one of the first works to introduce Transformer into video super-resolution tasks. TTVSR reduces computational costs and enables long-range modeling in videos. TTVSR can outperform existing SOTA methods in four widely-used VSR benchmarks.


### Overview
<img src="./fig/framework_TTVSR.png" width=100%>

### Visual
<img src="./fig/case_TTVSR.png" width=90%>

## Requirements and dependencies
* python 3.7 (recommend to use [Anaconda](https://www.anaconda.com/))
* pytorch == 1.9.0
* torchvision == 0.10.0
* mmcv-full == 1.3.9
* scikit-image == 1.7.3
* lmdb == 1.2.1
* yapf == 0.31.0
* tensorboard == 2.6.0

## Model and Results
Pre-trained models can be downloaded from [onedrive](), [baidu cloud]()(xxxx), [google drive]().
* *TTVSR_REDS.pth*: trained on REDS dataset with BI degradation.
* *TTVSR_Vimeo90K.pth*: trained on Vimeo-90K dataset with BD degradation.

The output results on REDS4, Vid4 and UMD10 can be downloaded from [onedrive](), [baidu cloud]()(xxxx), [google drive]().


## Dataset

1. Training set
	* [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset. We regroup the training and validation dataset into one folder. The original training dataset has 240 clips from 000 to 239. The original validation dataset were renamed from 240 to 269.
	    ```
		- Make REDS structure be:
			- REDS
				- train
					- train_sharp
						- 000
						- ...
						- 269
					- train_sharp_bicubic
						- X4
							- 000
							- ...
							- 269
        ```

	* [Viemo-90K](https://github.com/anchen1011/toflow) dataset. Download the [original training + test set](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip) and use the script 'degradation/BD_degradation.m' (run in MATLAB) to generate the low-resolution images. The `sep_trainlist.txt` file listing the training samples in the download zip file.
		```
		- Make Vimeo-90K structure be:
			- vimeo_septuplet
				- sequences
					- 00001
					- ...
					- 00096
				- sequences_BD
					- 00001
					- ...
					- 00096
				- sep_trainlist.txt
				- sep_testlist.txt
        ```

2. Testing set
	* [REDS4](https://seungjunnah.github.io/Datasets/reds.html) dataset. The 000, 011, 015, 020 clips from the original training dataset of REDS.
    * [Viemo-90K](https://github.com/anchen1011/toflow) dataset. The `sep_testlist.txt` file listing the testing samples in the download zip file.
    * [Vid4 and UDM10](https://www.terabox.com/web/share/link?surl=LMuQCVntRegfZSxn7s3hXw&path=%2Fproject%2Fpfnl) dataset. Use the script 'degradation/BD_degradation.m' (run in MATLAB) to generate the low-resolution images.
		```
		- Make Vid4 structure be:
			- VID4
				- BD
					- calendar
					- ...
					- walk
				- HR
					- calendar
					- ...
					- walk
	    ```
	    ```
		- Make UDM10 structure be:
			- UDM10
				- BD
					- archpeople
					- ...
					- polyflow
				- HR
					- archpeople
					- ...
					- polyflow
        ```
## Test
<!-- 1. Clone this github repo
```
git clone https://github.com/FuzhiYang/TTSR.git
cd TTSR
```
2. Download pre-trained models and modify "model_path" in test.sh
3. Run test
```
sh test.sh
```
4. The results are in "save_dir" (default: `./test/demo/output`)
 -->

<!-- ## Evaluation
1. Prepare CUFED dataset and modify "dataset_dir" in eval.sh
2. Download pre-trained models and modify "model_path" in eval.sh
3. Run evaluation
```
sh eval.sh
```
4. The results are in "save_dir" (default: `./eval/CUFED/TTSR`) -->

## Train
<!-- 1. Prepare CUFED dataset and modify "dataset_dir" in train.sh
2. Run training
```
sh train.sh
```
3. The training results are in "save_dir" (default: `./train/CUFED/TTSR`) -->

## Citation
```
@InProceedings{liu2022learning,
author = {Liu, Chengxu and Yang, Huan and Fu, Jianlong and Qian, Xueming},
title = {Learning Trajectory-Aware Transformer for Video Super-Resolution},
booktitle = {CVPR},
year = {2022},
month = {June}
}
```

## Acknowledgment
This code is built on [mmediting](https://github.com/open-mmlab/mmediting). We thank the authors of [BasicVSR](https://github.com/ckkelvinchan/BasicVSR-IconVSR) for sharing their code.

## Contact
If you meet any problems, please describe them in issues or contact:
* Chengxu Liu: <liuchx97@gmail.com>

