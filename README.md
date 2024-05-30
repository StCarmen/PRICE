# PRICE: A Pretrained Model for Cross-Database Cardinality Estimation

Cardinality estimation (CardEst) plays a crucial role in the optimization of query execution plans. Over the past decade, a number of methods have been proposed to address this issue, but limitations such as the inability to transfer across different databases have hindered their widespread deployment. 

In response, we present **PRICE**, a <u>PR</u>etrained mult<u>I</u>-table <u>C</u>ard<u>E</u>st model designed to overcome these limitations. PRICE leverages low-level and transferable features, as well as self-attention models, to learn meta-knowledge for computing cardinality in any database. It is generally applicable to any unseen new database to attain high estimation accuracy, while its preparation cost is as little as the basic one-dimensional histogram-based CardEst methods.

This repository includes the implementation for **PRICE** and the procedures for evaluation, pretraining, and finetuning. If you have any questions about our methodology or this repository, please contact us.

## Code Structure

The project directory structure is organized as follows:

```powershell
.
├── benchmark
├── datas
│   ├── statistics
│   │   ├── finetune
│   │   ├── pretrain
│   │   └── test
│   └── workloads
│       ├── finetune
│       ├── pretrain
│       └── test
├── model
├── results
├── setup
└── utils
    ├── model
    └── statistics
```

## Requirements

### Environment

python 3.10.13, pytorch 2.1.0, numpy 1.26.1, pandas 2.1.3, sqlglot 18.17.0

### Evaluation Tools

We utilize [Pilotscope](https://github.com/alibaba/pilotscope) to generate subqueries, calculate p-error, and evaluate end-to-end time. Refer to the [documentation](https://woodybryant.github.io/PilotScopeDoc.io/) for installation instructions and usage guidelines. Additional details about evaluation codes can be found in the `benchmark/` directory.

## Setup

### Datasets

The datasets utilized for model evaluation, pretraining, and finetuning are expected to be stored in the `datas/datasets` directory. Example datasets (30 datasets in total) are provided in the [data repository](https://drive.google.com/file/d/1-sSihVzjgrFbO_LoiwbjI2sVOPxO7N2G/view?usp=sharing), and the storage space required for these datasets is approximately **17GB**.

### Statistics

Statistics should be generated for datasets and placed in the `datas/statistics` directory. Please note that database statistics have already been provided in the `datas/statistics` directory for the datasets accessible in the <u>data repository</u>. If you have your own datasets within the `datas/datasets` directory, it is necessary to generate statistics for them. Further information regarding this process can be found in [utils/statistics/README.md](./utils/statistics/README.md).

### Features

Once statistics are available, features are generated for model training and execution. A bash command is provided for generating features for the datasets in the <u>data repository</u>:

```shell
sh setup/generate.sh
```

This script will require a period of time to complete, and logs will be accessible in `setup/features_log`. The resulting features will be stored in `setup/features` and will occupy approximately **7GB** of disk space. If you intend to generate features for your own datasets, please modify the script accordingly.

## Evaluation

We provide our pretrained model, which was pretrained on 26 datasets (the same as the paper states), in the `results/model_params.pth` file. To evaluate the estimation accuracy of the pretrained model on unseen datasets (e.g., IMDB, STATS, ErgastF1, VisualGenome), run the following command:

```shell
python evaluate.py
```

During the evaluation process, certain metrics related to estimation accuracy (e.g., q-error) will be displayed.

```shell
imdb loss: 1.6293717622756958
imdb q-error: 30%: 1.3949   50%: 1.7771   80%: 4.0716   90%: 8.3952   95%: 15.4516   99%: 70.8889
stats loss: 2.834393262863159
stats q-error: 30%: 1.3827   50%: 1.8709   80%: 5.4934   90%: 12.4606   95%: 35.5513   99%: 579.6716
ergastf1 loss: 1.3214871883392334
ergastf1 q-error: 30%: 1.2311   50%: 1.4315   80%: 3.1244   90%: 6.4412   95%: 14.3196   99%: 67.9744
genome loss: 1.5354844331741333
genome q-error: 30%: 1.3125   50%: 1.6545   80%: 3.5951   90%: 5.1724   95%: 15.6697   99%: 114.4227
```

## Pretraining

We also offer the option to pretrain the model from scratch. Below, we provide a list of recommended parameters for pretraining, which necessitates a minimum of **160GB** of GPU memory:

```shell
python pretrain.py --query_hidden_dim 512 --final_hidden_dim 1024 --n_embd 256 --n_layers 6 --n_heads 8 --dropout_rate 0.2 --batch_size 15000 --lr 2.85e-5
```

## Finetuning

We provide code for finetuning our pretrained model based on specific datasets as well. Examples of finetuning on datasets such as IMDB, STATS, ErgastF1, and VisualGenome are given. Below is an example:

```shell
python finetune.py --dataset genome --query_hidden_dim 512 --final_hidden_dim 1024 --n_embd 256 --n_layers 6 --n_heads 8 --dropout_rate 0.2 --batch_size 15000 --lr 1e-5
```

You can finetune the pretrained model with preferred parameters on your specific finetune datasets. It is important to note that features of the finetune datasets need to be available in the `setup/features` directory. For more details, please refer to the **Setup** section.
