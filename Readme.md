# MHCL

This is the code for paper:
> Multi-Channel Hypergraph Contrastive Learning for Matrix Completion

## Dependencies
Recent versions of the following packages for Python 3 are required:
* networkx==2.8.4
* numpy==1.22.3
* PyYAML==6.0.1
* Requests==2.31.0
* scikit_learn==1.2.2
* scipy==1.10.1
* setuptools==60.2.0
* sphinx_gallery==0.15.0
* tensorboardX==2.6.2
* torch==1.10.1
* torch_cluster==1.6.0
* torch_geometric==2.2.0
* torch_sparse==0.6.13
* tqdm==4.65.0
* dgl==0.4.1

## Datasets
All of the datasets we use are publicly available datasets.
### Link
The used datasets are available at:
* Retail_Rocket https://tianchi.aliyun.com/competition/entrance/231719/information/
* Alibaba https://github.com/xuehansheng/DualHGCN
* Amazon https://github.com/YingtongDou/CARE-GNN/tree/master/data
* YelpChi https://docs.dgl.ai/api/python/dgl.data.html#fraud-dataset

Kindly note that there may be two versions of node features for YelpChi. The old version has a dimension of 100 and the new version is 32. In our paper, the results are reported based on the new features.

## Usage
Use the following command to run the main script with configuration options:

* `python main.py --use_cfg --seeds 0 --dataset $dataset`


* `python -u train.py --data_name=yelp  --use_one_hot_fea --device -1  --ARR 0.0000003 --train_early_stopping_patience 150 --layers 4 --gcn_agg_units 90 --train_lr 0.01 --data_valid_ratio 0.111 --model_activation tanh --gcn_out_units 90 --train_decay_patience 30 --hyperedge_num 32`

* `python -u train.py --data_name=ml-100k --use_one_hot_fea --device -1 --ARR 0.0001 --layers 3 --data_valid_ratio 0.05 --model_activation tanh --train_decay_patience 60 --hyperedge_num 256`

* `python -u train.py --data_name=amazon  --use_one_hot_fea --device -1  --ARR 0.000000 --train_early_stopping_patience 200 --layers 3 --gcn_agg_units 90 --train_lr 0.01 --data_valid_ratio 0.111 --model_activation tanh --gcn_out_units 90 --train_decay_patience 60 --alpha 0.05 --hyperedge_num 16`

* `python -u train.py --data_name=yahoo_music --use_one_hot_fea --gcn_agg_accum=stack --device -1 --ARR 0.000000000002 --train_early_stopping_patience 100 --layers 3 --gcn_agg_units 45 --train_lr 0.01 --data_valid_ratio 0.1 --model_activation tanh --gcn_out_units 45`

* `python -u train.py --data_name=ml-1m --use_one_hot_fea --device -1 --ARR 0.000001 --gcn_agg_units 1200 --train_early_stopping_patience 1200 --train_decay_patience 200 --layers 2 --data_valid_ratio 0.09 --model_activation tanh --hyperedge_num 512`

