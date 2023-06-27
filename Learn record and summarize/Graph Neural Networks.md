# Introduction

**[A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)**

Graphs are all around us; real world objects are often defined in terms of their connections to other things. A set of objects, and the connections between them, are naturally expressed as a graph.

Researchers have developed neural networks that operate on graph data (called **graph neural networks, or GNNs**) for over a decade.

1. First, we look at what kind of data is most naturally phrased as a graph, and some common examples.

2. Second, we explore what makes graphs different from other types of data, and some of the specialized choices we have to make when using graphs.

3. Third, we build a modern GNN, walking through each of the parts of the model, starting with historic modeling innovations in the field. We move gradually from a bare-bones implementation to a state-of-the-art GNN model. 

4. Fourth, we provide a GNN playground where you can play around with a real-word task and dataset to build a stronger intuition of how each component of a GNN model contributes to the predictions it makes.


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



# Types of prediction tasks on graphs

There are three general types of prediction tasks on graphs: graph-level, node-level, and edge-level.

In a graph-level task, we predict a single property for a whole graph. For a node-level task, we predict some property for each node in a graph. For an edge-level task, we want to predict the property or presence of edges in a graph.

## Graph-level task

In a graph-level task, our goal is to predict the property of an entire graph. For example, for a molecule represented as a graph, we might want to predict what the molecule smells like, or whether it will bind to a receptor implicated in a disease.

This is analogous to image classification problems with MNIST and CIFAR, where we want to associate a label to an entire image. With text, a similar problem is sentiment analysis where we want to identify the mood or emotion of an entire sentence at once.

## Node-level task

Node-level tasks are concerned with predicting the identity or role of each node within a graph.

A classic example of a node-level prediction problem is Zach’s karate club. The dataset is a single social network graph made up of individuals that have sworn allegiance to one of two karate clubs after a political rift. As the story goes, a feud between Mr. Hi (Instructor) and John H (Administrator) creates a schism in the karate club. The nodes represent individual karate practitioners, and the edges represent interactions between these members outside of karate. The prediction problem is to classify whether a given member becomes loyal to either Mr. Hi or John H, after the feud. In this case, distance between a node to either the Instructor or Administrator is highly correlated to this label.

Following the image analogy, node-level prediction problems are analogous to *image segmentation*, where we are trying to label the role of each pixel in an image. With text, a similar task would be predicting the parts-of-speech of each word in a sentence (e.g. noun, verb, adverb, etc).

## Edge-level task

One example of edge-level inference is in image scene understanding. Beyond identifying objects in an image, deep learning models can be used to predict the relationship between them. We can phrase this as an edge-level classification: given nodes that represent the objects in the image, we wish to predict which of these nodes share an edge or what the value of that edge is. If we wish to discover connections between entities, we could consider the graph fully connected and based on their predicted value prune edges to arrive at a sparse graph.

<img src="https://distill.pub/2021/gnn-intro/merged.0084f617.png" style="zoom: 33%;" />

<img src="https://distill.pub/2021/gnn-intro/edges_level_diagram.c40677db.png" style="zoom:50%;" />



# The challenges of using graphs in machine learning

So, how do we go about solving these different graph tasks with neural networks? 

The first step is to think about how we will represent graphs to be compatible with neural networks.

Machine learning models typically take rectangular or grid-like arrays as input. So, it’s not immediately intuitive how to represent them in a format that is compatible with deep learning. Graphs have up to four types of information that we will potentially want to use to make predictions: nodes, edges, global-context and connectivity. The first three are relatively straightforward: for example, with nodes we can form a node feature matrix $N$ by assigning each node an index $i$ and storing the feature for $node_i$ in $N$. While these matrices have a variable number of examples, they can be processed without any special techniques.

However, representing a graph’s connectivity is more complicated. Perhaps the most obvious choice would be to use an adjacency matrix, since this is easily tensorisable. However, this representation has a few drawbacks. From the example dataset table, we see the number of nodes in a graph can be on the order of millions, and the number of edges per node can be highly variable. Often, this leads to very sparse adjacency matrices, which are space-inefficient. 

Another problem is that there are many adjacency matrices that can encode the same connectivity, and there is no guarantee that these different matrices would produce the same result in a deep neural network (that is to say, they are not permutation invariant).

One elegant and memory-efficient way of representing sparse matrices is as adjacency lists. These describe the connectivity of edge $e_k$ between nodes $n_i$ and $n_j$ as a tuple $(i,j)$ in the k-th entry of an adjacency list. Since we expect the number of edges to be much lower than the number of entries for an adjacency matrix $n_{nodes}^2$ , we avoid computation and storage on the disconnected parts of the graph.

Most practical tensor representations have vectors per graph attribute(per node/edge/global). Instead of a node tensor of size $[n_{nodes}]$ we will be dealing with node tensors of size $[n_{nodes},node_{dim}]$. Same for the other graph attributes.



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

![](https://distill.pub/2021/gnn-intro/prediction_edges_nodes.e6796b8e.png)

If we only have node-level features, and are trying to predict binary edge-level information, the model looks like this.

![](https://distill.pub/2021/gnn-intro/prediction_nodes_edges.26fadbcc.png)

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
