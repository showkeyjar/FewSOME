# FewSOME

This repository contains a Pytorch implementation of FewSOME: Onec-Class Few Shot Anomaly Detection with Siamese Networks.


## Citation 

## Abstract


## Installation 
This code is written in Python 3.8 and requires the packages listed in requirements.txt.

Use the following command to clone the repository to your local machine:


```
git clone https://github.com/pakdd2023anon1/FewSOME.git
```

To run the code, set up a virtual environment:

```
pip install virtualenv
cd <path-to-FewSOME-directory>
virtualenv myenv
source activate myenv
source myenv/bin/activate
pip install -r requirements.txt
```


## Running Experiments


### MNIST Example

```
cd <path-to-FewSOME-directory>

# activate virtual environment
source myenv/bin/activate

# change to source directory
cd src

# run experiment
python3 train.py -m model --num_ref_eval 30 --lr 1e-6 --batch_size 1 --weight_init_seed 1001 --dataset 'mnist' --normal_class 6 -N 30 --seed 1001 --eval_epoch 0 --epochs 100 --data_path ~/data/ --download_data True --smart_samp 0 --k 1 --alpha 0.8 --vector_size 1024 --task test --pretrain 1 --model_type 'MNIST_VGG3'

```

### CIFAR-10 Example

```
cd <path-to-FewSOME-directory>

# activate virtual environment
source myenv/bin/activate

# change to source directory
cd src

# run experiment
python3 train.py -m model --num_ref_eval 30 --lr 1e-5 --batch_size 1 --weight_init_seed 1001 --dataset 'cifar10' --normal_class 0 -N 30 --seed 1001 --eval_epoch 0 --epochs 100 --data_path ~/data/ --download_data True --smart_samp 0 --k 1 --alpha 0.5 --vector_size 2048 --task test --pretrain 1 --model_type 'CIFAR_VGG3'

```

### Fashion-MNIST Example

```
cd <path-to-FewSOME-directory>

# activate virtual environment
source myenv/bin/activate

# change to source directory
cd src

# run experiment
python3 train.py -m model --num_ref_eval 30 --lr 1e-4 --batch_size 16 --weight_init_seed 1001 --dataset 'fashion' --normal_class 9 -N 30 --seed 1001 --eval_epoch 0 --epochs 100 --data_path ~/data/ --download_data True --smart_samp 0 --k 1 --alpha 0.1 --vector_size 2048 --task test --pretrain 1 --model_type 'FASHION_VGG3'

```


### MVTec AD Example
This experiment requires the data to be downloaded from https://www.mvtec.com/company/research/datasets/mvtec-ad
```
cd <path-to-FewSOME-directory>

# activate virtual environment
source myenv/bin/activate

# change to source directory
cd src

# run experiment
python3 train.py -m model --num_ref_eval 60 --lr 1e-4 --batch_size 1 --weight_init_seed 1001 --dataset 'mvtec' --normal_class 13 -N 60 --seed 1001 --eval_epoch 0 --epochs 100 --data_path <path_to_data> --download_data True --smart_samp 0 --k 1 --alpha 1 --task test --pretrain 1 --model_type 'RESNET'

```


## Desription of arguments for train.py

```
parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--model_type', choices = ['CIFAR_VGG3','CIFAR_VGG4','MVTEC_VGG3','MNIST_VGG3', 'MNIST_LENET', 'CIFAR_LENET', 'RESNET', 'FASHION_VGG3'], required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--normal_class', type=int, default = 0)
    parser.add_argument('-N', '--num_ref', type=int, default = 30)
    parser.add_argument('--num_ref_eval', type=int, default = None)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--vector_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default = 100)
    parser.add_argument('--weight_init_seed', type=int, default = 100)
    parser.add_argument('--alpha', type=float, default = 0)
    parser.add_argument('--freeze', default = True)
    parser.add_argument('--smart_samp', type = int, choices = [0,1], default = 0)
    parser.add_argument('--k', type = int, default = 0)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('--download_data',  default=True)
    parser.add_argument('--contamination',  type=float, default=0)
    parser.add_argument('--v',  type=float, default=0.0)
    parser.add_argument('--task',  default='train', choices = ['test', 'train'])
    parser.add_argument('--eval_epoch', type=int, default=0)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--get_visual', choices = [0,1,2,3], type=int, default=0)
    parser.add_argument('--augment_no', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--biases', type=int, default=1)
    parser.add_argument('--stop_gradient', type=int, default=1)
    parser.add_argument('--num_ref_dist', type=int, default=None)
    parser.add_argument('--anchor_dist', type=int, default=0)
    parser.add_argument('--mean_dist', type=int, default=0)
    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)

```

'-m' - this argument allows the user to enter a custom model name

'--model_type' - specify the architecture 

'--dataset' - specify the dataset 

'--normal_class' - specify the index of the class that is considered normal, all other classes are anomalies 

'-N' - specify the number of reference images 

'--num_ref_eval' - during training, the distance is calculated between the reference image in question and num_ref_eval reference images in order to find the reference images that are furthest away from the reference image in question. By default x=N. However, training can be sped up by setting x < N.

'--lr' - specify the learning rate

'--vector_size' - specify the dimensions of the feature embeddings.

'--weight_decay' - specify the weight decay

'--seed' - specify the seed to select reference images

'--weight_init_seed' - specify the model seed

'--alpha' - specify the value of alpha between 0 and 1.

'--freeze' -- specify whether to freeze the values of the feature embeddings of one of the reference images known as the 'anchor'

'--smart_samp' - specify whether to pair a reference image with reference images that have the largest euclidean distance.

'--k' - specify the number of reference images to calculate the distance from 

'--epochs' - specify the number of epochs 

'--data_path' - specify where to save the data

'--download_data' - specify whether to download data

'--contamination' - specify level of pollution i.e. the percentage of anomalies present in the training data 

'--v' - soft boundary parameter 

'--task' - value 'train' trains the model and tests on a validation set, 'test' trains the model and tests on the test set 

'--eval_epoch' - specifies whether to evaluate the model after each epoch 

'--pretrain' - specifies whether to use pretrained weights 

'--get_visual' - a value of 1 saves the feature vectures for the reference images and 2000 validation images at the beginning of each epoch, a value of 2 saves the feature vectors after each pass through the model, a value of 3 save the feature vectors for each pass in the first epoch and then at the start of each epoch.

'--augment_no' - specify number of reference images to augment beyong recognition so that they are labelled as anomalies (not used)

'--batch_size' - specify the batch size 

'--biases' - specify whether to turn off or on biases. 

'-i' - specify indexes of training set to have as a reference set 




