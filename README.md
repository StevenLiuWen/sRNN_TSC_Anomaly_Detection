# A revisit of sparse coding based anomaly detection in stacked rnn framework
This repo is the official open source of [A revisit of sparse coding based anomaly detection in stacked rnn framework, ICCV 2017]

It is implemented on tensorflow. Please follow the instructions to run the code.

## 1. Installation (Anaconda with python3.6 installation is recommended)
* Install 3rd-package dependencies of python (listed in requirements.txt)
```
numpy==1.15.4
matplotlib==2.2.2
scikit_image==0.13.1
six==1.11.0
opencv_python==3.4.3.18
h5py==2.7.1
scipy==1.1.0
tensorflow_gpu==1.11.0
seaborn==0.8.1
skimage==0.0
scikit_learn==0.20.2
tensorflow==1.12.0
PyYAML==3.13
```

```shell
pip install -r requirements.txt

pip install tensorflow-gpu==1.11.0
```
* Other libraries
```code
CUDA 8.0
Cudnn 6.0
Ubuntu 14.04 or 16.04, Centos 7 and other distributions.
```
## 2. Download datasets
cd into Data folder of project and run the shell scripts (**ped1.sh, ped2.sh, avenue.sh, shanghaitech.sh**) under the Data folder.
```shell
cd Data
./ped2.sh
./ped1.sh
./avenue.sh
./shanghaitech.sh
```

If the download shell does not work (I guess the download link in shell script is not permanent), please manually download all datasets from [ped1.tar.gz, ped2.tar.gz, avenue.tar.gz and shanghaitech.tar.gz](https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F)
and tar each tar.gz file, and move them in to **Data** folder.

## 3. Extracting feature
 Download [twostream action recognition model](https://github.com/feichtenhofer/twostreamfusion), then put it into extract_feature folder. Please infer the instructions of this model and download pretrained models.
 Modidy the root in extract_feature/extract_feature_twostream/extract_feature.m
 run extract_feature.m using MATLAB

## 4. Training 

## Citation
If you find this useful, please cite our work as follows:
```code
@article{luo2017revisit,
  title={A revisit of sparse coding based anomaly detection in stacked rnn framework},
  author={Luo, Weixin and Liu, Wen and Gao, Shenghua},
  journal={ICCV, Oct},
  volume={1},
  number={2},
  pages={3},
  year={2017}
}
```


