# Traffic prediction model
We consider and test here traffic prediction models.
The main goal for us is to consider models, that incorporate graph properties and graph features.


We start from the set of the following models

1 T-GCN is the source codes for the paper named “T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction” published in IEEE Transactions on Intelligent Transportation Systems (T-ITS) which forged the T-GCN model the spatial and temporal dependence simultaneously.

2 AST-GCN is the source codes for the paper named “AST-GCN: Attribute-Augmented Spatiotemporal Graph Convolutional Network for Traffic Forecasting” published in IEEE Access which strengthen the T-GCN model model with attribute information.

3 DFD-GCN https://github.com/GestaltCogTeam/DFDGCN/tree/main/DFDGCN 

## Main contribution and changes of the model 

We are adding the local information from graphs encoding their features into the vector information.
We distinguish two main sets of features:
- structural features of graph (degree, betweenness, clustering, community structure) 
- temporal features of graph such as listed here https://github.com/romanroff/ITMO_subjects/blob/main/NIR/utils/features.py 

Encoding of graph features is done using the framework of encoding/decoding proposed in [Hamilton, R. Ying, and J. Leskovec. 2017. Representation Learning on Graphs: Methods and Applications. IEEE Data Engineering Bulletin (2017]

### Related libraries
Basic set of models for predictions 
https://github.com/GestaltCogTeam/BasicTS 
