# Introduction

**[A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)**

**[Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/)**

**[Build your first Graph Neural Network model to predict traffic speed in 20 minutes](https://towardsdatascience.com/build-your-first-graph-neural-network-model-to-predict-traffic-speed-in-20-minutes-b593f8f838e5)**

**[CNN-explainer](https://poloclub.github.io/cnn-explainer/)**

[**Fast Fourier Transform**](https://zhuanlan.zhihu.com/p/31584464)

[**如何理解 Graph Convolutional Networks**](https://www.zhihu.com/question/54504471/answer/332657604)

[**如何理解 Graph Attention Networks**](https://zhuanlan.zhihu.com/p/81350196)

Graphs are all around us; real world objects are often defined in terms of their connections to other things. A set of objects, and the connections between them, are naturally expressed as a graph.

Researchers have developed neural networks that operate on graph data (called **graph neural networks, or GNNs**) for over a decade.




# Graph

A graph represents the relations (*edges*) between a collection of entities (*nodes*).

- Three types of attributes we might find in a graph
  - Vertex (or node) attributes
  - Edge (or link) attributes and directions
  - Global (or master node) attributes
- To further describe each node, edge or the entire graph, we can store information in each of these pieces of the graph.
  - Vertex (or node) embedding
  - Edge (or link) attributes and embedding
  - Global (or master node) embedding
- We can additionally specialize graphs by associating directionality to edges (*directed, undirected*).
  - The edges can be directed, where an edge $𝑒$ has a source node, $𝑣_{𝑠𝑟𝑐}$, and a destination node $𝑣_{𝑑𝑠𝑡}$. In this case, information flows from $𝑣_{𝑠𝑟𝑐}$ to $𝑣_{𝑑𝑠𝑡}$.
  - They can also be undirected, where there is no notion of source or destination nodes, and information flows both directions.

## Images & Text as graphs

We typically think of images as rectangular grids with image channels, representing them as arrays (e.g., 244x244x3 floats). Another way to think of images is as graphs with regular structure, where each pixel represents a node and is connected via an edge to adjacent pixels. Each non-border pixel has exactly 8 neighbors, and the information stored at each node is a 3-dimensional vector representing the RGB value of the pixel.

A way of visualizing the connectivity of a graph is through its *adjacency matrix*.

------

We can digitize text by associating indices to each character, word, or token, and representing text as a sequence of these indices. This creates a simple directed graph, where each character or index is a node and is connected via an edge to the node that follows it.

This representation (a sequence of character tokens) refers to the way text is often represented in RNNs; other models, such as Transformers, can be considered to view text as a fully connected graph where we learn the relationship between tokens. 

------

Of course, in practice, this is not usually how text and images are encoded: these graph representations are redundant since all images and all text will have very regular structures. For instance, images have a banded structure in their adjacency matrix because all nodes (pixels) are connected in a grid. The adjacency matrix for text is just a diagonal line, because each word only connects to the prior word, and to the next one.

## Graph-valued data in the wild

**Molecules as graphs.** Molecules are the building blocks of matter, and are built of atoms and electrons in 3D space. All particles are interacting, but when a pair of atoms are stuck in a stable distance from each other, we say they share a covalent bond. Different pairs of atoms and bonds have different distances (e.g. single-bonds, double-bonds). It’s a very convenient and common abstraction to describe this 3D object as a graph, where nodes are atoms and edges are covalent bonds.

**Social networks as graphs.** Social networks are tools to study patterns in collective behaviour of people, institutions and organizations. We can build a graph representing groups of people by modelling individuals as nodes, and their relationships as edges.

**Citation networks as graphs.** Scientists routinely cite other scientists’ work when publishing papers. We can visualize these networks of citations as a graph, where each paper is a node, and each *directed* edge is a citation between one paper and another. Additionally, we can add information about each paper into each node, such as a word embedding of the abstract. 

**Other examples.**In computer vision, we sometimes want to tag objects in visual scenes. We can then build graphs by treating these objects as nodes, and their relationships as edges. Machine learning models, programming code and math equations can also be phrased as graphs, where the variables are nodes, and edges are operations that have these variables as input and output. You might see the term “dataflow graph” used in some of these contexts.

The structure of real-world graphs can vary greatly between different types of data — some graphs have many nodes with few connections between them, or vice versa. Graph datasets can vary widely (both within a given dataset, and between datasets) in terms of the number of nodes, edges, and the connectivity of nodes.

## Types of prediction tasks on graphs

There are three general types of prediction tasks on graphs: graph-level, node-level, and edge-level.

In a graph-level task, we predict a single property for a whole graph. For a node-level task, we predict some property for each node in a graph. For an edge-level task, we want to predict the property or presence of edges in a graph.

### Graph-level task

In a graph-level task, our goal is to predict the property of an entire graph. For example, for a molecule represented as a graph, we might want to predict what the molecule smells like, or whether it will bind to a receptor implicated in a disease.

This is analogous to image classification problems with MNIST and CIFAR, where we want to associate a label to an entire image. With text, a similar problem is sentiment analysis where we want to identify the mood or emotion of an entire sentence at once.

### Node-level task

Node-level tasks are concerned with predicting the identity or role of each node within a graph.

A classic example of a node-level prediction problem is Zach’s karate club. The dataset is a single social network graph made up of individuals that have sworn allegiance to one of two karate clubs after a political rift. As the story goes, a feud between Mr. Hi (Instructor) and John H (Administrator) creates a schism in the karate club. The nodes represent individual karate practitioners, and the edges represent interactions between these members outside of karate. The prediction problem is to classify whether a given member becomes loyal to either Mr. Hi or John H, after the feud. In this case, distance between a node to either the Instructor or Administrator is highly correlated to this label.

Following the image analogy, node-level prediction problems are analogous to *image segmentation*, where we are trying to label the role of each pixel in an image. With text, a similar task would be predicting the parts-of-speech of each word in a sentence (e.g. noun, verb, adverb, etc).

### Edge-level task

One example of edge-level inference is in image scene understanding. Beyond identifying objects in an image, deep learning models can be used to predict the relationship between them. We can phrase this as an edge-level classification: given nodes that represent the objects in the image, we wish to predict which of these nodes share an edge or what the value of that edge is. If we wish to discover connections between entities, we could consider the graph fully connected and based on their predicted value prune edges to arrive at a sparse graph.

<img src="https://distill.pub/2021/gnn-intro/merged.0084f617.png" style="zoom: 33%;" />

<img src="https://distill.pub/2021/gnn-intro/edges_level_diagram.c40677db.png" style="zoom:50%;" />



## The challenges of using graphs in machine learning

So, how do we go about solving these different graph tasks with neural networks? 

The first step is to think about how we will represent graphs to be compatible with neural networks.

Machine learning models typically take rectangular or grid-like arrays as input. So, it’s not immediately intuitive how to represent them in a format that is compatible with deep learning. Graphs have up to four types of information that we will potentially want to use to **make predictions: nodes, edges, global-context and connectivity.** The first three are relatively straightforward: for example, with nodes we can form a node feature matrix $N$ by assigning each node an index $i$ and storing the feature for $node_i$ in $N$. While these matrices have a variable number of examples, they can be processed without any special techniques.

However, representing a graph’s connectivity is more complicated. Perhaps the most obvious choice would be to use an adjacency matrix, since this is easily tensorisable. However, this representation has a few drawbacks. From the example dataset table, we see the number of nodes in a graph can be on the order of millions, and the number of edges per node can be highly variable. Often, this leads to very sparse adjacency matrices, which are space-inefficient. 

Another problem is that there are many adjacency matrices that can encode the same connectivity, and there is no guarantee that these different matrices would produce the same result in a deep neural network (that is to say, they are not permutation invariant).

One elegant and memory-efficient way of representing sparse matrices is as **adjacency lists**. These describe the connectivity of edge $e_k$ between nodes $n_i$ and $n_j$ as a tuple $(i,j)$ in the k-th entry of an adjacency list. Since we expect the number of edges to be much lower than the number of entries for an adjacency matrix $n_{nodes}^2$ , we avoid computation and storage on the disconnected parts of the graph.

Most practical tensor representations have vectors per graph attribute(per node/edge/global). Instead of a node tensor of size $[n_{nodes}]$ we will be dealing with node tensors of size $[n_{nodes},node_{dim}]$. Same for the other graph attributes.



## The Challenges of Computation on Graphs

### Lack of Consistent Structure

Graphs are extremely flexible mathematical models; but this means they lack consistent structure across instances. 

Consider the task of predicting whether a given chemical molecule is toxic. Looking at a few examples, the following issues quickly become apparent:

- Molecules may have different numbers of atoms.
- The atoms in a molecule may be of different types.
- Each of these atoms may have different number of connections.
- These connections can have different strengths.

Representing graphs in a format that can be computed over is non-trivial, and the final representation chosen often depends significantly on the actual problem.

### Node-Order Equivariance

Extending the point above: graphs often have no inherent ordering present amongst the nodes. Compare this to images, where every pixel is uniquely determined by its absolute position within the image!

The same graph labelled in two different ways. The alphabets indicate the ordering of the nodes.

![](https://distill.pub/2021/understanding-gnns/images/node-order-alternatives.svg)

As a result, we would like our algorithms to be node-order equivariant: they should not depend on the ordering of the nodes of the graph. If we permute the nodes in some way, the resulting representations of the nodes as computed by our algorithms should also be permuted in the same way.

### Scalability

Graphs can be really large! Think about social networks like Facebook and Twitter, which have over a billion users. Operating on data this large is not easy.

Luckily, most naturally occuring graphs are ‘sparse’: they tend to have their number of edges linear in their number of vertices. We will see that this allows the use of clever methods to efficiently compute representations of nodes within the graph. Further, the methods that we look at here will have significantly fewer parameters in comparison to the size of the graphs they operate on.



# Graph Neural Networks

Now that the graph’s description is in a matrix format that is permutation invariant, we will describe using graph neural networks (GNNs) to solve graph prediction tasks.

**A GNN is an optimizable transformation on all attributes of the graph (nodes, edges, global-context) that preserves graph symmetries (permutation invariances).** 

GNNs adopt a “graph-in, graph-out” architecture meaning that these model types accept a graph as input, with information loaded into its nodes, edges and global-context, and progressively transform these embeddings, without changing the connectivity of the input graph.

## The simplest GNN

We will start with the simplest GNN architecture, one where we learn new embeddings for all graph attributes (nodes, edges, global), but where we do not yet use the connectivity of the graph.

This GNN uses a separate multilayer perceptron (MLP) on each component of a graph; we call this a GNN layer. For each node vector, we apply the MLP and get back a learned node-vector. We do the same for each edge, learning a per-edge embedding, and also for the global-context vector, learning a single embedding for the entire graph.

![](https://distill.pub/2021/gnn-intro/arch_independent.0efb8ae7.png)

A single layer of a simple GNN. A graph is the input, and each component (V,E,U) gets updated by a MLP to produce a new graph. Each function subscript indicates a separate function for a different graph attribute at the n-th layer of a GNN model. 

As is common with neural networks modules or layers, we can stack these GNN layers together. Because a GNN does not update the connectivity of the input graph, we can describe the output graph of a GNN with the same adjacency list and the same number of feature vectors as the input graph. But, the output graph has updated embeddings, since the GNN has updated each of the node, edge and global-context representations.

## GNN Predictions by Pooling Information

We will consider the case of binary classification, but this framework can easily be extended to the multi-class or regression case. If the task is to make binary predictions on nodes, and the graph already contains node information, the approach is straightforward — for each node embedding, apply a linear classifier.

![](https://distill.pub/2021/gnn-intro/prediction_nodes_nodes.c2c8b4d0.png)

However, it is not always so simple. For instance, you might have information in the graph stored in edges, but no information in nodes, but still need to make predictions on nodes. We need a way to collect information from edges and give them to nodes for prediction. We can do this by *pooling*. 

- Pooling proceeds in two steps:
  - For each item to be pooled, *gather* each of their embeddings and concatenate them into a matrix.
  - The gathered embeddings are then *aggregated*, usually via a sum operation.

We represent the *pooling* operation by the letter $\rho$, and denote that we are gathering information from edges to nodes as  $\rho_{E_n\to V_n}$.

So If we only have edge-level features, and are trying to predict binary node information, we can use pooling to route (or pass) information to where it needs to go. The model looks like this.

<img src="https://distill.pub/2021/gnn-intro/prediction_edges_nodes.e6796b8e.png" style="zoom:10%;" />

If we only have node-level features, and are trying to predict binary edge-level information, the model looks like this.

<img src="https://distill.pub/2021/gnn-intro/prediction_nodes_edges.26fadbcc.png" style="zoom:10%;" />

If we only have node-level features, and need to predict a binary global property, we need to gather all available node information together and aggregate them. This is similar to *Global Average Pooling* layers in CNNs. The same can be done for edges.

<img src="https://distill.pub/2021/gnn-intro/prediction_nodes_edges_global.7a535eb8.png" style="zoom: 10%;" />

In our examples, the classification model $c$ can easily be replaced with any differentiable model, or adapted to multi-class classification using a generalized linear model.

This is an end-to-end prediction task with a GNN model.

![](https://distill.pub/2021/gnn-intro/Overall.e3af58ab.png)

Now we’ve demonstrated that we can build a simple GNN model, and make binary predictions by routing information between different parts of the graph. This pooling technique will serve as a building block for constructing more sophisticated GNN models. If we have new graph attributes, we just have to define how to pass information from one attribute to another.

Note that in this simplest GNN formulation, we’re not using the connectivity of the graph at all inside the GNN layer. Each node is processed independently, as is each edge, as well as the global context. We only use connectivity when pooling information for prediction.

## Passing messages between parts of the graph

We could make more sophisticated predictions by using pooling within the GNN layer, in order to make our learned embeddings aware of graph connectivity. We can do this using *message passing*, where neighboring nodes or edges exchange information and influence each other’s updated embeddings.

- Message passing works in three steps:
  - For each node in the graph, *gather* all the neighboring node embeddings (or messages), which is the $g$ function described above.
  - Aggregate all messages via an aggregate function (like sum).
  - All pooled messages are passed through an *update function*, usually a learned neural network.

Just as pooling can be applied to either nodes or edges, message passing can occur between either nodes or edges.

These steps are key for leveraging the connectivity of graphs. We will build more elaborate variants of message passing in GNN layers that yield GNN models of increasing expressiveness and power.

This sequence of operations, when applied once, is the simplest type of message-passing GNN layer.

This is reminiscent of standard convolution: in essence, message passing and convolution are operations to aggregate and process the information of an element’s neighbors in order to update the element’s value. In graphs, the element is a node, and in images, the element is a pixel. However, the number of neighboring nodes in a graph can be variable, unlike in an image where each pixel has a set number of neighboring elements.

By stacking message passing GNN layers together, a node can eventually incorporate information from across the entire graph: after three layers, a node has information about the nodes three steps away from it.

We can update our architecture diagram to include this new source of information for nodes. Schematic for a GCN architecture, which updates node representations of a graph by pooling neighboring nodes at a distance of one degree.

![](https://distill.pub/2021/gnn-intro/arch_gcn.40871750.png)

## Learning edge representations

Our dataset does not always contain all types of information (node, edge, and global context). When we want to make a prediction on nodes, but our dataset only has edge information, we showed above how to use pooling to route information from edges to nodes, but only at the final prediction step of the model. We can share information between nodes and edges within the GNN layer using message passing.

We can incorporate the information from neighboring edges in the same way we used neighboring node information earlier, by first pooling the edge information, transforming it with an update function, and storing it.

However, the node and edge information stored in a graph are not necessarily the same size or shape, so it is not immediately clear how to combine them. One way is to learn a linear mapping from the space of edges to the space of nodes, and vice versa. Alternatively, one may concatenate them together before the update function.

Architecture schematic for Message Passing layer. The first step “prepares” a message composed of information from an edge and it’s connected nodes and then “passes” the message to the node.

![](https://distill.pub/2021/gnn-intro/arch_mpnn.a13c2294.png)

Which graph attributes we update and in which order we update them is one design decision when constructing GNNs. We could choose whether to update node embeddings before edge embeddings, or the other way around. This is an open area of research with a variety of solutions– for example we could update in a ‘weave’ fashion where we have four updated representations that get combined into new node and edge representations: node to node (linear), edge to edge (linear), node to edge (edge layer), edge to node (node layer).

Some of the different ways we might combine edge and node representation in a GNN layer:

<img src="https://distill.pub/2021/gnn-intro/arch_weave.352befc0.png"  />

## Adding global representations

There is one flaw with the networks we have described so far: nodes that are far away from each other in the graph may never be able to efficiently transfer information to one another, even if we apply message passing several times. For one node, If we have k-layers, information will propagate at most k-steps away. This can be a problem for situations where the prediction task depends on nodes, or groups of nodes, that are far apart. One solution would be to have all nodes be able to pass information to each other. Unfortunately for large graphs, this quickly becomes computationally expensive (although this approach, called ‘virtual edges’, has been used for small graphs such as molecules).

One solution to this problem is by using the global representation of a graph (U) which is sometimes called a **master node** or context vector. This global context vector is connected to all other nodes and edges in the network, and can act as a bridge between them to pass information, building up a representation for the graph as a whole. This creates a richer and more complex representation of the graph than could have otherwise been learned.

Schematic of a Graph Nets architecture leveraging global representations:

![](https://distill.pub/2021/gnn-intro/arch_graphnet.b229be6d.png)

In this view all graph attributes have learned representations, so we can leverage them during pooling by conditioning the information of our attribute of interest with respect to the rest. For example, for one node we can consider information from neighboring nodes, connected edges and the global information. To condition the new node embedding on all these possible sources of information, we can simply concatenate them. Additionally we may also map them to the same space via a linear map and add them or apply a feature-wise modulation layer, which can be considered a type of featurize-wise attention mechanism.

Schematic for conditioning the information of one node based on three other embeddings (adjacent nodes, adjacent edges, global). This step corresponds to the node operations in the Graph Nets Layer.

![](https://distill.pub/2021/gnn-intro/graph_conditioning.3017e214.png)



## Problem Setting and Notation

There are many useful problems that can be formulated over graphs:

- **Node Classification:** Classifying individual nodes.
- **Graph Classification:** Classifying entire graphs.
- **Node Clustering:** Grouping together similar nodes based on connectivity.
- **Link Prediction:** Predicting missing links.
- **Influence Maximization:** Identifying influential nodes.
- $\cdots \cdots$

<img src="https://distill.pub/2021/understanding-gnns/images/graph-tasks.svg"  />

A common precursor in solving many of these problems is **node representation learning**: learning to map individual nodes to fixed-size real-valued vectors (called ‘representations’ or ‘embeddings’).

Different GNN variants are distinguished by the way these representations are computed. Generally, however, GNNs compute node representations in an iterative process. We will use the notation $h_v^{(k)}$ to indicate the representation of node $v$ after the $k^{th}$  iteration. Each iteration can be thought of as the equivalent of a ‘layer’ in standard neural networks.

We will define a graph $G$ as a set of nodes, $V$ with a set of edges $E$ connecting them. Nodes can have individual features as part of the input: we will denote by $x_v$ the individual feature for node $v\in V$. For example, the ‘node features’ for a pixel in a color image would be the red, green and blue channel (RGB) values at that pixel.

Sometimes we will need to denote a graph property by a matrix $M$, where each row $M_v$ represents a property corresponding to a particular vertex $v$.



# Modern Graph Neural Networks

ChebNet was a breakthrough in learning localized filters over graphs, and it motivated many to think of graph convolutions from a different perspective.

We return back to the result of convolving $x$ by by the polynomial kernel $p_w(L)=L$ , focussing on a particular vertex $v$:
$$
\begin{align}
(Lx)_v
& = L_vx \\
& = \sum_{u\in G} L_{vu}x_u \\
& = \sum_{u\in G}(D_{vu}-A_{vu})x_u \\
& = D_vx_v-\sum_{u\in N(v)x_u}
\end{align}
$$
As we noted before, this is a 1-hop localized convolution. But more importantly, we can think of this convolution as arising of two steps:

- Aggregating over immediate neighbour features $x_u$
- Combining with the node’s own feature $x_v$

**Key Idea:** What if we consider different kinds of ‘aggregation’ and ‘combination’ steps, beyond what are possible using polynomial filters?

By ensuring that the aggregation is node-order equivariant, the overall convolution becomes node-order equivariant.

These convolutions can be thought of as ‘message-passing’ between adjacent nodes: after each step, every node receives some ‘information’ from its neighbours.

By iteratively repeating the 1-hop localized convolutions $K$ times (i.e., repeatedly ‘passing messages’), the receptive field of the convolution effectively includes all nodes upto $K$ hops away.

Message-passing forms the backbone of many GNN architectures today. We describe the most popular ones in depth below:

## Graph Convolutional Networks (GCN)

![](https://github.com/CorneliusDeng/Markdown-Photos/blob/main/Graph%20Neural%20Networks/Graph%20Convolutional%20Networks.png?raw=true)

## Graph Attention Networks (GAT)

![](https://github.com/CorneliusDeng/Markdown-Photos/blob/main/Graph%20Neural%20Networks/Graph%20Attention%20Networks.png?raw=true)

## Graph Sample and Aggregate (GraphSAGE)

![](https://github.com/CorneliusDeng/Markdown-Photos/blob/main/Graph%20Neural%20Networks/Graph%20Sample%20and%20Aggregate.png?raw=true)

## Graph Isomorphism Network (GIN)

![](https://github.com/CorneliusDeng/Markdown-Photos/blob/main/Graph%20Neural%20Networks/Graph%20Isomorphism%20Network.png?raw=true)

## Learning GNN Parameters

All of the embedding computations we’ve described here, whether spectral or spatial, are completely differentiable. This allows GNNs to be trained in an end-to-end fashion, just like a standard neural network, once a suitable loss function $L$ is defined:

- **Node Classification**: By minimizing any of the standard losses for classification tasks, such as categorical cross-entropy when multiple classes are present:
  $$
  L(y_v,\widehat{y}_v)=\sum_cy_{vc}log\;\widehat{y}_{vc}
  $$
  where $\widehat{y}_{vc}$ is the predicted probability that node $v$ is in class $c$. GNNs adapt well to the semi-supervised setting, which is when only some nodes in the graph are labelled. In this setting, one way to define a loss $L_G$ over an input graph $G$ is:
  $$
  L_G=\frac{\sum_{v\in Lab(G)}L(y_v,\widehat{y}_v)}{|Lab(G)|}
  $$
  where, we only compute losses over labelled nodes $Lab(G)$.

- **Graph Classification**: By aggregating node representations, one can construct a vector representation of the entire graph. This graph representation can be used for any graph-level task, even beyond classification.

- **Link Prediction**: By sampling pairs of adjacent and non-adjacent nodes, and use these vector pairs as inputs to predict the presence/absence of an edge. For a concrete example, by minimizing the following ‘logistic regression’-like loss:
  $$
  \begin{align}
  L(y_v,y_u,e_{vu}) & =-e_{vu}log(p_{vu})-(1-e_{vu})log(1-p_{vu}) \\
  p_{vu} & = \sigma(y_v^Ty_u)
  \end{align}
  $$
  where $\sigma$ is the sigmoid function, and $e_{vu} = 1$ iff there is an edge between nodes $v$ and $u$, being 0 otherwise.

- **Node Clustering**: By simply clustering the learned node representations.

The broad success of pre-training for natural language processing models such as ELMo and BERT has sparked interest in similar techniques for GNNs . The key idea in each of these papers is to train GNNs to predict local (eg. node degrees, clustering coefficient, masked node attributes) and/or global graph properties (eg. pairwise distances, masked global attributes).

Another self-supervised technique is to enforce that neighbouring nodes get similar embeddings, mimicking random-walk approaches such as node2vec and DeepWalk :
$$
L_G=\sum_v\sum_{u\in N_R(v)}log\frac{exp\;z_v^Tz_u}{exp\;z^T_{u'}z_u}
$$
where $N_R(v)$ is a multi-set of nodes visited when random walks are started from $v$. For large graphs, where computing the sum over all nodes may be computationally expensive, techniques such as Noise Contrastive Estimation are especially useful.



# Graph Convolutional Networks

## Extending Convolutions to Graphs

Convolutional Neural Networks have been seen to be quite powerful in extracting features from images. However, images themselves can be seen as graphs with a very regular grid-like structure, where the individual pixels are nodes, and the RGB channel values at each pixel as the node features.

A natural idea, then, is to consider generalizing convolutions to arbitrary graphs. However, ordinary convolutions are not node-order invariant, because they depend on the absolute positions of pixels. It is initially unclear as how to generalize convolutions over grids to convolutions over general graphs, where the neighbourhood structure differs from node to node.

Convolutions in CNNs are inherently localized. GNNs can perform localized convolutions mimicking CNNs.

CNN中的卷积本质上就是利用一个共享参数的过滤器(Kernel)，**通过计算中心像素点以及相邻像素点的加权和来构成 feature map 实现空间特征的提取**，当然加权系数就是卷积核的权重系数。**离散卷积本质就是一种加权求和**

**卷积核的参数通过优化求出才能实现特征提取的作用，GCN的理论很大一部分工作就是为了引入可以优化的卷积参数**

GCN的本质目的就是用来提取拓扑图的空间特征，除了 graph convolution这一种途径外，在 vertex domain(spatial domain) 和 spectral domain 实现目标是两种最主流的方式。

- **空间维度**

  Vertex domain(spatial domain) 是非常直观的一种方式。顾名思义：提取拓扑图上的空间特征，那么就把每个顶点相邻的 neighbors 找出来，其中蕴含了两个子问题：

  a. 按照什么条件去找中心vertex的neighbors，也就是如何确定 receptive field

  b. 确定receptive field，按照什么方式处理包含不同数目neighbors的特征

  [Learning Convolutional Neural Networks for Graphs](http://proceedings.mlr.press/v48/niepert16.pdf) 给出的方法主要缺点如下：

  每个顶点提取出来的neighbors不同，使得计算处理必须针对每个顶点；提取特征的效果可能没有卷积好

- **图谱维度**

  **Spectral domain** 就是GCN的理论基础了。这种思路就是希望借助图谱的理论来实现拓扑图上的卷积操作。从整个研究的时间进程来看：首先研究GSP（graph signal processing）的学者定义了graph上的Fourier Transformation，进而定义了graph上的Convolution，最后与深度学习结合提出了Graph Convolutional Network

  从vertex domain分析问题，需要逐节点（node-wise）的处理，而图结构是非欧式的连接关系，这在很多场景下会有明显的局限，而spectral domain是将问题转换在“频域”里分析，不再需要node-wise的处理，对于很多场景能带来意想不到的便利，也成为了GSP的基础

## Graph Convolution

对于图 $G=(V,E)$ ，其Laplacian矩阵的定义为 $L=D-A$，其中 $L$ 是Laplacian 矩阵，$D$ 是顶点的度矩阵（对角矩阵），对角线上元素依次为各个顶点的度，$A$ 是图的邻接矩阵。

<img src="https://picx.zhimg.com/80/v2-5f9cf5fdeed19b63e1079ed2b87617b4_1440w.webp?source=1940ef5c" style="zoom:100%;" />

- 为什么GCN要用拉普拉斯矩阵？
  - 拉普拉斯矩阵是对称矩阵，可以进行特征分解（谱分解），这就和GCN的spectral domain对应上了
  - 拉普拉斯矩阵只在中心顶点和一阶相连的顶点上（1-hop neighbor）有非0元素，其余之处均为0
  - 通过拉普拉斯算子与拉普拉斯矩阵进行类比

**GCN的核心基于拉普拉斯矩阵的谱分解**

矩阵的谱分解，特征分解，对角化都是同一个概念。不是所有的矩阵都可以特征分解，其充要条件为n阶方阵存在n个线性无关的特征向量。拉普拉斯矩阵是半正定对称矩阵，有如下三个性质：

- 实对称矩阵一定n个线性无关的特征向量
- 半正定矩阵的特征值一定非负
- 实对阵矩阵的特征向量总是可以化成两两相互正交的正交矩阵

由上可以知道拉普拉斯矩阵一定可以谱分解，且分解后有特殊的形式，对于拉普拉斯矩阵其谱分解为：
$$
L= U
\left(\begin{matrix}\lambda_1 & \\&\ddots \\ &&\lambda_n \end{matrix}\right)
U^{-1}
$$
其中 $U=(\vec{u_1},\vec{u_2},\cdots,\vec{u_n})$，是列向量为单位特征向量的矩阵，也就说 $\vec{u_l}$ 是列向量。由于 $U$ 是正交矩阵，即 $UU^{T}=E$ ，$E$ 是单位矩阵

所以特征分解又可以写成：
$$
L= U\left(\begin{matrix}\lambda_1 & \\&\ddots \\ &&\lambda_n \end{matrix}\right) U^{T}
$$
把传统的傅里叶变换以及卷积迁移到Graph上来，核心工作其实就是把拉普拉斯算子的特征函数 $e^{-i\omega t}$ 变为Graph对应的拉普拉斯矩阵的特征向量。

### Graph上的傅里叶变换

传统的傅里叶变换定义为：$F(\omega)=\mathcal{F}[f(t)]=\int_{}^{}f(t)e^{-i\omega t} dt$，就是 信号 $f(t)$ 与基函数 $e^{-i\omega t}$ 的积分，从数学上看，$e^{-i\omega t}$ 是拉普拉斯算子的特征函数（满足特征方程）, $\omega$ 就和特征值有关。

广义的特征方程定义为：$A V=\lambda V$，其中 $A$ 是一种变换，$V$ 是特征向量或者特征函数（无穷维的向量），$\lambda$ 是特征值。

$e^{-i\omega t}$ 满足：$\Delta e^{-i\omega t}=\frac{\partial^{2}}{\partial t^{2}} e^{-i\omega t}=-\omega^{2} e^{-i\omega t}$，当然 $e^{-i\omega t}$ 就是变换 $\Delta$ 的特征函数，$\omega$ 和特征值密切相关。

那么，可以联想了，处理Graph问题的时候，用到拉普拉斯矩阵（拉普拉斯矩阵就是离散拉普拉斯算子），自然就去找拉普拉斯矩阵的特征向量了。

$L$ 是拉普拉斯矩阵，$V$ 是其特征向量，自然满足下式：$LV=\lambda V$

离散积分就是一种内积形式，仿上定义Graph上的傅里叶变换：$F(\lambda_l)=\hat{f}(\lambda_l)=\sum_{i=1}^{N}{f(i) u_l^*(i)}$，$f$ 是Graph上的 $N$ 维向量，$f(i)$ 与Graph的顶点一一对应，$u_l(i)$ 表示第 $l$ 个特征向量的第 $i$ 个分量。那么特征值（频率）$\lambda_l$ 下的，$f$ 的Graph 傅里叶变换就是与 $\lambda_l$ 对应的特征向量 $u_l$ 进行内积运算。

利用矩阵乘法将Graph上的傅里叶变换推广到矩阵形式：
$$
\left(\begin{matrix} \hat{f}(\lambda_1)\\ \hat{f}(\lambda_2) \\ \vdots \\\hat{f}(\lambda_N)\end{matrix}\right)=\left(\begin{matrix}\ u_1(1) &u_1(2)& \dots &u_1(N) \\u_2(1) &u_2(2)&\dots &u_2(N)\\ \vdots &\vdots &\ddots & \vdots\\ u_N(1) &u_N(2)& \dots &u_N(N)\end{matrix}\right)\left(\begin{matrix}f(1)\\ f(2) \\ \vdots \\f(N) \end{matrix}\right)
$$
即 $f$ 在Graph上傅里叶变换的矩阵形式为：$\hat{f}=U^Tf $

### Graph上的傅里叶逆变换

类似地，传统的傅里叶逆变换是对频率 $\omega$ 求积分：$\mathcal{F}^{-1}[F(\omega)]=\frac{1}{2\Pi}\int_{}^{}F(\omega)e^{i\omega t} d\omega$

迁移到Graph上变为对特征值 $\lambda_l$ 求和：$f(i)=\sum_{l=1}^{N}{\hat{f}(\lambda_l) u_l(i)}$

利用矩阵乘法将Graph上的傅里叶逆变换推广到矩阵形式：
$$
\left(\begin{matrix}f(1)\\ f(2) \\ \vdots \\f(N) \end{matrix}\right)= \left(\begin{matrix}u_1(1) &u_2(1)& \dots &u_N(1) \\u_1(2) &u_2(2)& \dots &u_N(2)\\ \vdots &\vdots&\ddots & \vdots\\ u_1(N) &u_2(N)& \dots &u_N(N) \end{matrix}\right)\left(\begin{matrix} \hat{f}(\lambda_1)\\ \hat{f}(\lambda_2) \\ \vdots \\\hat{f}(\lambda_N)\end{matrix}\right)
$$
即 $f$ 在Graph上傅里叶逆变换的矩阵形式为：$f=U\hat{f}$

### 推广卷积

在上面的基础上，利用卷积定理类比来将卷积运算，推广到Graph上。

卷积定理：函数卷积的傅里叶变换是函数傅立叶变换的乘积，即对于函数 $f(t)$ 与 $h(t)$ 两者的卷积是其函数傅立叶变换乘积的逆变换：
$$
f*h=\mathcal{F}^{-1}\left[ \hat{f}(\omega)\hat{h}(\omega) \right]=\frac{1}{2\Pi}\int_{}^{}\hat{f}(\omega)\hat{h}(\omega)e^{i\omega t} d\omega
$$
类比到Graph上并把傅里叶变换的定义带入，$f$ 与卷积核 $h$ 在Graph上的卷积可按下列步骤求出：

$f$ 的傅里叶变换为 $\hat{f}=U^Tf$

卷积核 $h$ 的傅里叶变换写成对角矩阵的形式即为： $\left(\begin{matrix}\hat h(\lambda_1) &\\&\ddots \\ &&\hat h(\lambda_n) \end{matrix}\right)$

$\hat{h}(\lambda_l)=\sum_{i=1}^{N}{h(i) u_l^*(i)}$ 是根据需要设计的卷积核 $h$ 在Graph上的傅里叶变换

两者的傅立叶变换乘积即为：$\left(\begin{matrix}\hat h(\lambda_1) & \\&\ddots \\ &&\hat{h}(\lambda_n) \end{matrix}\right)U^Tf$

再乘以 $U$ 求两者傅立叶变换乘积的逆变换，则求出卷积：$(f*h)_G= U\left(\begin{matrix}\hat h(\lambda_1) & \\&\ddots \\ &&\hat h(\lambda_n)\end{matrix}\right) U^Tf$

注：很多论文中的Graph卷积公式为：$(f*h)_G=U((U^Th)\odot(U^Tf)) $

$\odot$ 表示Hadamard product（哈达马积），对于两个维度相同的向量、矩阵、张量进行对应位置的逐元素乘积运算，其实两式是完全等价的

### **为什么拉普拉斯矩阵的**特征向量可以作为傅里叶变换的基？

傅里叶变换一个本质理解就是：把任意一个函数表示成了若干个正交函数（由 $\sin\omega t,\cos\omega t$ 构成）的线性组合。

<img src="https://picx.zhimg.com/80/v2-e9e00533154bfdad940e966e7eca5075_1440w.webp?source=1940ef5c" style="zoom:100%;" />

graph傅里叶变换也把graph上定义的任意向量 $f$，表示成了拉普拉斯矩阵特征向量的线性组合，即：$f=\hat{f}(\lambda_1)u_1+\hat{f}(\lambda_2)u_2+\cdots +\hat{f}(\lambda_n)u_n$

那么：为什么graph上任意的向量 $f$ 都可以表示成这样的线性组合？

原因在于 $(\vec{u_1},\vec{u_2},\cdots,\vec{u_n})$ 是graph上 $n$ 维空间中的 $n$ 个线性无关的正交向量，由线性代数的知识可以知道：$n$ 维空间中 $n$ 个线性无关的向量可以构成空间的一组基，而且拉普拉斯矩阵的特征向量还是一组正交基。

此外，对于传统的傅里叶变换，拉普拉斯算子的特征值 $\omega$ 表示谐波 $\sin\omega t,\cos\omega t$ 的频率。与之类似，拉普拉斯矩阵的特征值 $\lambda_i$ 也表示图拉普拉斯变换的频率。

## Graph Convolution Neural Network

Deep Learning 中的 Convolution 就是要设计含有 trainable 共享参数的 kernel，而 Graph Convolution 中的卷积参数就是 $diag(\hat{h}(\lambda_l))$

###  The first generation GCN

[Spectral Networks and Locally Connected Networks on Graphs](https://arxiv.org/abs/1312.6203) 中简单粗暴地把 $diag(\hat h(\lambda_l) )$ 变成了卷积核 $diag(\theta_l )$ ，也就是：
$$
y_{output}=\sigma \left(U g_\theta(\Lambda) U^T x \right) \\
g_\theta(\Lambda)=\left(\begin{matrix}\theta_1 &\\&\ddots \\ &&\theta_n\end{matrix}\right)
$$
它就是标准的第一代GCN中的 layer了，其中 $\sigma(\cdot)$ 是激活函数，$\Theta=({\theta_1},{\theta_2},\cdots,{\theta_n})$ 就跟三层神经网络中的weight一样是任意的参数，通过初始化赋值然后利用误差反向传播进行调整，$x$ 就是graph上对应于每个顶点的feature vector（由特数据集提取特征构成的向量）。

（为避免混淆，记 $g_\theta(\Lambda)$ 是卷积核，$U g_\theta(\Lambda) U^T$ 的运算结果为卷积运算矩阵）

- 第一代的参数方法存在着一些弊端：主要在于：
  - 每一次前向传播，都要计算 $U$, $diag(\theta_l )$ 及 $U^T$ 三者的矩阵乘积，特别是对于大规模的graph，计算的代价较高，也就是论文中
    $\mathcal{O}(n^3)$ 的计算复杂度
  - 卷积核不具有spatial localization
  - 卷积核需要 n 个参数

###  The second generation GCN

[Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://proceedings.neurips.cc/paper_files/paper/2016/hash/04df4d434d481c5bb723be1b6df1ee65-Abstract.html) 把 $\hat h(\lambda_l)$ 巧妙地设计成了 $\sum_{j=0}^K \alpha_j \lambda^j_l$，也就是：
$$
y_{output}=\sigma \left(U g_\theta(\Lambda) U^T x \right) \\
g_\theta(\Lambda)=\left(\begin{matrix}\sum_{j=0}^K \alpha_j \lambda^j_1 &\\&\ddots \\&& \sum_{j=0}^K \alpha_j \lambda^j_n \end{matrix}\right)
$$
上面的公式仿佛还什么都看不出来，下面利用矩阵乘法进行变换：$\left(\begin{matrix}\sum_{j=0}^K \alpha_j \lambda^j_1 &\\&\ddots \\ && \sum_{j=0}^K\alpha_j \lambda^j_n \end{matrix}\right)=\sum_{j=0}^K \alpha_j \Lambda^j$

进而可以导出：$U \sum_{j=0}^K \alpha_j \Lambda^j U^T =\sum_{j=0}^K \alpha_j U\Lambda^j U^T =\sum_{j=0}^K \alpha_j L^j$ 

上式成立是因为 $L^2=U \Lambda U^TU \Lambda U^T=U \Lambda^2 U^T$ 且 $U^T U=E$

那么，等式变换为
$$
y_{output}=\sigma \left( \sum_{j=0}^{K-1} \alpha_j L^j x \right)
$$
其中 $({\alpha_0},{\alpha_1},\cdots,{\alpha_{K-1}})$ 是任意的参数，通过初始化赋值然后利用误差反向传播进行调整。

- 这样设计的卷积核其优点在于
  - 卷积核只有 $K$ 个参数，一般 $K$ 远小于 $n$，参数的复杂度被大大降低了
  - 矩阵变换后，神奇地发现不需要做特征分解了，直接用拉普拉斯矩阵 $L$ 进行变换。然而由于要计算 $L^j$，计算复杂度还是 $\mathcal{O}(n^3)$
  - 卷积核具有很好的spatial localization，特别地，$K$ 就是卷积核的receptive field，也就是说每次卷积会将中心顶点K-hop neighbor上的feature进行加权求和，权系数就是 $\alpha_k$

更直观地看，$K=1$ 就是对每个顶点上一阶neighbor的feature进行加权求和，如下图所示：

<img src="https://pic1.zhimg.com/80/v2-5f756da1ce39f38d408bd771a15c8ad3_1440w.webp?source=1940ef5c" style="zoom:100%;" />

同理，$K=2$ 的情形如下图所示：

<img src="https://picx.zhimg.com/80/v2-a13b82907a364c3707a18bb8572b3a63_1440w.webp?source=1940ef5c" style="zoom:100%;" />

### Chebyshev polynomials are used as convolution kernels

在GCN领域中，利用Chebyshev多项式作为卷积核是非常通用的形式，此处暂不展开



# Graph Attention Networks

[Graph Attention Networks](https://openreview.net/forum?id=rJXMpikCZ) 

GCN是处理transductive任务的一把利器（transductive任务是指：训练阶段与测试阶段都基于同样的图结构），然而GCN有**两大局限性**是经常被诟病的：

1. **无法完成inductive任务，即处理动态图问题。**

   inductive任务是指：训练阶段与测试阶段需要处理的graph不同。通常是训练阶段只是在子图（subgraph）上进行，测试阶段需要处理未知的顶点。（unseen node）

2. **处理有向图的瓶颈，不容易实现分配不同的学习权重给不同的neighbor**

在[Graph Attention Networks](https://arxiv.org/abs/1710.10903)中提到，**GAT本质上可以有两种运算方式**

1. **Global graph attention**

   顾名思义，就是每一个顶点 $i$ 都对于图上任意顶点都进行attention运算

   优点：完全不依赖于图的结构，对于inductive任务无压力

   缺点：（1）丢掉了图结构的这个特征，无异于自废武功，效果可能会很差（2）运算面临着高昂的成本

2. **Mask graph attention**

   注意力机制的运算只在邻居顶点上进行

   而在这篇文章中，作者也是采取这样的方式

GAT的计算步骤主要有两步：

- **计算注意力系数（attention coefficient）**

  对于顶点 $i$，逐个计算它的邻居们（ $j\in N_i$ ）和它自己之间的相似系数
  $$
  e_{ij}=a([Wh_i||Wh_j]),j\in N_i
  $$
  解释：首先一个共享参数 $W$ 的线性映射对于顶点的特征进行了增维，当然这是一种常见的特征增强（feature augment）方法；$[\cdot || \cdot]$ 对于顶点 $i,j$ 的变换后的特征进行了拼接（concatenate）；最后 $a(\cdot)$ 把拼接后的高维特征映射到一个实数上，作者是通过 single-layer feedforward neural network实现的

  显然学习顶点 $i,j$ 之间的相关性，就是通过可学习的参数 $W$ 和映射 $a(\cdot)$ 完成的

  有了相关系数，再通过 softmax 归一化计算注意力系数
  $$
  \alpha_{ij}=\frac{exp(\text{Leaky ReLU}(e_{ij}))}{\sum_{k\in N_i}exp(\text{Leaky ReLU}(e_{ik}))}
  $$
  
- **加权求和（aggregate）**

  根据计算好的注意力系数，把特征加权求和（aggregate）
  $$
  h'_i=\sigma(\sum_{j\in N_i}\alpha_{ij}Wh_j)
  $$
  
  其中，$h'_i$ 就是 GAT 输出的对于每个顶点 $i$ 的新特征（融合了邻域信息）， $\sigma(\cdot)$ 是激活函数。
  
  再使用 **multi-head attention** 进一步增强
  $$
  h'_i(K)= ||_{k=1}^K\; \sigma(\sum_{j\in N_i}\alpha_{ij}^kW^kh_j)
  $$
  其中，$||$ 表示拼接操作，multi-head attention也可以理解成用了ensemble的方法

**几点深入理解**

- 在本质上，GCN与GAT都是将邻居顶点的特征聚合到中心顶点上（一种aggregate运算），利用graph上的local stationary学习新的顶点特征表达。不同的是GCN利用了拉普拉斯矩阵，GAT利用attention系数。一定程度上而言，GAT会更强，因为顶点特征之间的相关性被更好地融入到模型中。

- GAT适用于有向图最根本的原因是GAT的运算方式是逐顶点的运算（node-wise）。每一次运算都需要循环遍历图上的所有顶点来完成。逐顶点运算意味着，摆脱了拉普利矩阵的束缚，使得有向图问题迎刃而解。

- GAT适用于inductive任务原因是，GAT中重要的学习参数是 $W$ 与 $a(\cdot)$，因为逐顶点运算方式，这两个参数仅与顶点特征相关，与图的结构毫无关系。所以测试任务中改变图的结构，对于GAT影响并不大，只需要改变 $N_i$，重新计算即可。

  与此相反的是，GCN是一种全图的计算方式，一次计算就更新全图的节点特征。学习的参数很大程度与图结构相关，这使得GCN在inductive任务上遇到困境。



