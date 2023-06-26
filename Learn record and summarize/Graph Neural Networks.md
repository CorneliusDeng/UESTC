# Introduction

**[A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)**

Graphs are all around us; real world objects are often defined in terms of their connections to other things. A set of objects, and the connections between them, are naturally expressed as a graph.

Researchers have developed neural networks that operate on graph data (called **graph neural networks, or GNNs**) for over a decade.

First, we look at what kind of data is most naturally phrased as a graph, and some common examples.

Second, we explore what makes graphs different from other types of data, and some of the specialized choices we have to make when using graphs.

Third, we build a modern GNN, walking through each of the parts of the model, starting with historic modeling innovations in the field. We move gradually from a bare-bones implementation to a state-of-the-art GNN model. 

Fourth, we provide a GNN playground where you can play around with a real-word task and dataset to build a stronger intuition of how each component of a GNN model contributes to the predictions it makes.



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

## Images and Text as graphs

We typically think of images as rectangular grids with image channels, representing them as arrays (e.g., 244x244x3 floats). Another way to think of images is as graphs with regular structure, where each pixel represents a node and is connected via an edge to adjacent pixels. Each non-border pixel has exactly 8 neighbors, and the information stored at each node is a 3-dimensional vector representing the RGB value of the pixel.

A way of visualizing the connectivity of a graph is through its *adjacency matrix*.

------

We can digitize text by associating indices to each character, word, or token, and representing text as a sequence of these indices. This creates a simple directed graph, where each character or index is a node and is connected via an edge to the node that follows it.

This representation (a sequence of character tokens) refers to the way text is often represented in RNNs; other models, such as Transformers, can be considered to view text as a fully connected graph where we learn the relationship between tokens. 

------

Of course, in practice, this is not usually how text and images are encoded: these graph representations are redundant since all images and all text will have very regular structures. For instance, images have a banded structure in their adjacency matrix because all nodes (pixels) are connected in a grid. The adjacency matrix for text is just a diagonal line, because each word only connects to the prior word, and to the next one.



