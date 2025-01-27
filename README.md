# CA-TFHN
## About
This project is the implementation of the paper "MOOCs Dropout Prediction via Classmates Augmented Time-Flow Hybrid Network"
![all_method](paperpic/all_method.png)

## Abstract
Massive Open Online Courses (MOOCs) provides learners
a platform for free learning. However, MOOCs have been criticized for
the high dropout rates in recent years. For the purpose of predicting
users’ potential dropout risk in advance, a novel framework named as
Classmates Augmented Time-Flow Hybrid Network (CA-TFHN) is proposed in this paper. TFHN, absorbed the advantages of LSTM and self-attention mechanism, is designed to generate the activity features of
users by using users’ learning records. At the same time, an effective
correlation calculation is defined based on users’ potential interests on
courses with link prediction, bringing in relationships of classmates. A
user graph is reconstructed with the influences among classmates. User
features generated from this graph are fused to the activity features
of user, resulting in accurate dropout prediction. Experiments on the
XuetangX dataset demonstrate the effectiveness of CA-TFHN in predicting dropout of MOOCs.

## Dependencies
```torch==1.11.0```<p>
```torch-geometric==2.1.0.post1```<p>
```torch-cluster==1.6.0```<p>
```torch-scatter==2.0.9```<p>
```torch-sparse==0.6.14```<p>
```torch-spline-conv==1.2.1```<p>
```numpy==1.22.3```<p>
```pandas==1.4.2```<p>
```scikit-learn==1.1.2```<p>


## Dataset
XuetangX dataset: [Downloads](https://github.com/wzfhaha/dropout_prediction) <p>
KDD Cup 2015 dataset: [Downloads](http://lfs.aminer.cn/misc/moocdata/data/kddcup15.zip)

## Usage
```shell
usage: train.py [-h] [-indir INDIR] [-outdir OUTDIR] [-e E] [-r R] [-lr LR]

optional parameters

optional arguments:
  -h, --help      show this help message and exit
  -indir INDIR    input dir (default: current dir)
  -outdir OUTDIR  output dir (default: current dir)
  -e E            epoch (default: 15)
  -r R            random seed (default: 0)
  -lr LR          learning rate (default: 1e-4)
```

## Demo
All the codes need to run in the linux environment or others that
support the linux shell file
```shell
# download data from www.moocdata.org
sh dump_data.sh

# extract feature from raw data
sh feat_extract.sh

# run CA-TFHN model
python train.py
```

## Reference
:clap:Congratulations! Our paper has been accepted by the CCF Conference ICONIP(2023 International Conference on Neural Information Processing).
```
@inproceedings{liang2023moocs,
  title={MOOCs Dropout Prediction via Classmates Augmented Time-Flow Hybrid Network},
  author={Liang, Guanbao and Qian, Zhaojie and Wang, Shuang and Hao, Pengyi},
  booktitle={International Conference on Neural Information Processing},
  pages={405--416},
  year={2023},
  organization={Springer}
}
```
