# A Wide & Deep Learning Sharing Input Data for Regression Analysis

A PyTorch implementation of WDSI proposed in our paper:<br>
[*A Wide & Deep Learning Sharing Input Data for Regression Analysis*](https://doi.org/10.1109/BigComp48618.2020.0-108)

## Requirements
* python 3.8.12
* pytorch 1.10.2
* scikit-learn 1.0.2
* numpy 1.21.2
* pandas 1.2.3
* absl-py 0.10.0

## Directory Structure
```
data_dir
L train.csv
L test.csv
```

## Data Format
```
field1,field2,...,fieldN,target
...
```

## Usage

### train
```
python train.py
```

## Citation
```
@inproceedings{
    kim2020wide,
    title={A Wide & Deep Learning Sharing Input Data for Regression Analysis},
    author={Kim, Minkyu, Lee, Suan and Kim, Jinho},
    booktitle={2020 IEEE International Conference on Big Data and Smart Computing (BigComp)},
    year={2020}
}
```
