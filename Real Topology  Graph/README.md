# Scalable Graph Neural Network
Graph Neural Net for six robots to perform consensus.

<p float="center">
  <img src="consensus_graph1.PNG" width="370" hspace="20"/>
  <img src="consensus_graph2.PNG" width="370" /> 
</p>

### Table of Content

- [Consensus Algorithms](#Consensus%20Algorithms)
- [Data Collection](#Data%20Collection)
- [GNN Model 1](#GNN%20Model%201)
- [GNN Model 2](#GNN%20Model%202)
- [GNN Model 3](#GNN%20Model%203)
- [GNN Model 4](#GNN%20Model%204)


For running our same robotic scene load `Data Collection/Scene_of_Six_Robots.ttt`.
Note: read comments within each module for more details about the code.

## Consensus Algorithms
This repo contains code for implementing the consensus algorithm for Fully Connnected and Line Graph.

* `main_consensus_algorithm_fully_connected.py`: code for consensus algorithm for six mobile robots Full Graph adjacency matrix.
* `main_consensus_algorithm_line_graph.py`: code for consensus algorithm for six mobile robots Line Graph adjacency matrix.

If you want to use a different graph like cyclic, just change the adjancency matrix at the beginning of the code.

## Data Collection
This repo contains all our code for data collection for the different versions of the GNN models we built.

The major content of this repo is:

* `data_collection_v1.py`: consensus algorithm of six robots using fully connected graph, and saving data for fully connected graph.
* `data_collection_v2.py`: consensus algorithm of six robots using fully connected graph, and saving data for cyclic graph. 
* `data_collection_v3.py`: consensus algorithm of six robots using fully connected graph, and saving data for fully connected graph.
* `data_collection_v4.py`: consensus algorithm of six robots using fully connected graph, and saving data for cyclic graph. 

All details of variables stored for each version are within each module.

## GNN Model 1
A two input two output GNN model for the following cases:

1) Fully Connected Graph:

Trained GNN model on fully connected graph data.

* `MLP_Model.py`: code for building and training GNN model.
* `Main_MLP_Cyclic.py `: code for running consensus on a cyclic graph. 
* `Main_MLP_Fully.py`: code for running consensus on a fully connected graph.
* `Main_MLP_line.py`: code for running consensus on a line graph. 



2) Cyclic Graph:

Trained GNN model on cyclic graph data.

* `MLP_Model.py`: code for building and training GNN model.
* `Main_MLP_Cyclic.py `: code for running consensus on a cyclic graph. 
* `Main_MLP_Fully.py`: code for running consensus on a fully connected graph.
* `Main_MLP_line.py`: code for running consensus on a line graph. 

## GNN Model 2
A four input two output GNN model for the following cases:

1) Fully Connected Graph:

Trained GNN model on fully connected graph data.

* `MLP_Model.py`: code for building and training GNN model.
* `Main_MLP_Cyclic.py `: code for running consensus on a cyclic graph. 
* `Main_MLP_Fully.py`: code for running consensus on a fully connected graph.
* `Main_MLP_line.py`: code for running consensus on a line graph. 



2) Cyclic Graph:

Trained GNN model on cyclic graph data.

* `MLP_Model.py`: code for building and training GNN model.
* `Main_MLP_Cyclic.py `: code for running consensus on a cyclic graph. 
* `Main_MLP_Fully.py`: code for running consensus on a fully connected graph.
* `Main_MLP_line.py`: code for running consensus on a line graph. 

