# MHCL

This is the code for paper:
> Multi-Channel Hypergraph Contrastive Learning for Matrix Completion

## Dependencies
Recent versions of the following packages for Python 3 are required:
* networkx==2.8.4
* numpy==1.22.3
* Requests==2.31.0
* scikit_learn==1.2.2
* scipy==1.10.1
* setuptools==60.2.0
* sphinx_gallery==0.15.0
* torch==1.10.1
* torch_cluster==1.6.0
* torch_geometric==2.2.0
* torch_sparse==0.6.13

## Datasets
All of the datasets we use are publicly available datasets.
### Link
The used datasets are available at:
* Amazon 
* Yelp 
* ML-100K 
* ML-1M
* YahooMusic
* Douban

## Usage
Use the following command to run the main script with configuration options. For example:

* `python main.py --use_cfg --seeds 0 --dataset $dataset`

* `python -u train.py --data_name=ml-100k --use_one_hot_fea --device -1 --ARR 0.0001 --layers 3 --data_valid_ratio 0.05 --model_activation tanh --train_decay_patience 60 --hyperedge_num 256`
