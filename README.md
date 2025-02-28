# Structural Information Guided Representation Reconstruction for Dynamic Graph Anomaly Detection

PyTorch implementation of the paper "[Structural Information Guided Representation Reconstruction for Dynamic Graph Anomaly Detection]()".

#  Requirments
+ torch==1.13.1+cu116
+ torchvision==0.14.1+cu116
+ torch-geometric==2.5.3
+ torch-scatter==2.1.1
+ torch-sparse==0.6.17
+ scikit-learn==1.3.2

# Preprocessing

## Dataset
Download data.csv into file './dataset/'  
  
[Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)  

[Mooc](http://snap.stanford.edu/jodie/mooc.csv)

[LastFM](https://snap.stanford.edu/jodie/lastfm.csv)

[Myket](https://github.com/erfanloghmani/myket-android-application-market-dataset)

## Preprocessing
We use the data processing method of the reference [TGAT](https://openreview.net/pdf?id=rJeW1yHYwH), [repo](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs#inductive-representation-learning-on-temporal-graphs-iclr-2020).  

We use then dense npy format to save the features in binary format. If edge features features are absent, it will replaced by a vector of zeros. 

We use the Wikipedia and Mooc datasets below as an example for running instructions:
```python
# Preprocess Wikipedia
python build_dataset_graph.py --data wikipedia --bipartite --clusters 10
# Preprocess Mooc
python build_dataset_graph.py --data mooc --bipartite --clusters 10
```
## Model Training

Training the SIGAD Graph network based on half of black samples. The dimension of input_dim is aligned with the dimension of the interaction feature and defaults to 172.
```python
# Run Wikipedia
python train.py --data_set wikipedia --anomaly_alpha 1e-1 --mask_label --mask_ratio 0.5
# Run Mooc
python train.py --data_set mooc --anomaly_alpha 1e-1 --mask_label --mask_ratio 0.5 --input_dim 4
```


