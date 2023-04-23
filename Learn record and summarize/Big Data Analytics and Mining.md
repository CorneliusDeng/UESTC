# Introduction to Big Data Mining

- Big data is a buzzword, or catch-phrase, used to describe a massive volume of both structured and unstructured data that is so large that it's difficult to process using traditional database and software techniques. 

- The four V's of big data

  - Volume: scale of data
  - Variety: different forms of data
  - Velocity: analysis of streaming data
  - Veracity: uncertainty of data

- Data mining consists of applying data analysis and discovery algorithms that, under acceptable computational efficiency limitations, produce a particular  enumeration of patterns over the data

- Main Data Mining Tasks

  - Association Rule Mining
  - Cluster Analysis
  - Classification/Prediction
  - Outlier Detection

- Main Directions of Big Data Mining

  - Scalable Data Mining Algorithms 

    Basic idea: propose new data mining algorithms to handle big data using differernt strategies, mainly focus on: 

    - Novel scalable algorithms (Sampling, Hashing, Divide-and-Conquer, etc.)
    - Map-reduce oriented Parallell platforms (Hadoop, Spark, GraphLab)
    - Speed-up Hardwares (GPUs/Clouds/Clusters)

  - Data Stream Mining

    Basic idea: propose new data mining algorithms to handle evloving data streams using differernt strategies, mainly focus on:

    - Handle Evolving Data Streams 
    - Clustering on Massive Data Streams
    - Classification/Prediction
    - Semi-supervised Learning
    - Irregular nodes/patterns mining on data streams

  - Multi-source or multi-type data mining

    Basic idea: propose new algorithms for structure or unstructure data, or different types of data, different sources, etc,  mainly focus on:

    - Different types of data (E.g. categorical data, nominal data, mixed- type data mining )
    - Different forms of data (vector data, graph data, image/text data, hetergeneous data) 
    - Learning on different sources of data (Multi-view learning, Transfer learning, Multi-task learning, etc)
    - Data fusion or data integration (Multiple-kernel learning）

  - Uncertainty Analysis, Link/Missing value prediction

    - Link Prediction (Clustering-based, Structure-based, Multi-view based)

    - Uncertain data Clustering (Probability-function based)

    - Recommender System (Missing items prediction,...)

    - Robust Machine learning (Adversarial examples)

      

# Foundation of Data Mining

## Main tasks in machine learning

- Supervised learning: targets to learn the mapping function or relationship between the features and the labels based on the labeled data.
- Unsupervised learning: aims at learning the intrinsic structure from unlabeled data.
- Semi-supervised learning: can be regarded as the unsupervised learning with some constraints on labels, or the supervised learning with additional information on the distribution of data. 

Supervised Learning
- Given training data $X=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$ where $y_i$ is the corresponding label of data $x_i$, supervised learning learns the mapping function $Y=F(X|\theta)$, or the posterior distribution $P(Y|X)$
- Supervised problems: Classification、Regression、 Learn to Rank、Tagging

## Loss Function

To measure the predicted results, we introduce the loss function $L(Y,F(X|θ))$, which a non-negative function

- 0-1 loss 
  $$
  L(y,F(x))= 
  \begin{cases}
  0, &  y = F(x|\theta) \\
  1, &  y ≠ F(x|\theta)
  \end{cases}
  $$

- Squared loss: $L(y,F(x|\theta))=(y-F(x|\theta))^2$

- Absolute loss: $L(y,F(x|\theta))=|y-F(x|\theta)|$

- Log loss (Cross Entropy): $L(y,P(y|x,\theta))=-logP(y|x,\theta)=-\displaystyle\sum_{j=1}^Cy_jlogp_j=-[y_jlogp_j+(1-y_j)log(1-p_j)]$

Training loss: loss on training data   

Test loss: loss on test data

Generalization 

- Empirical  risk: $R(F)=\frac{1}{N}\displaystyle\sum_{i=1}^NL(y_i,F(x_i))$
- Note: A good model cannot only take training loss into account and minimize the empirical risk. Instead, improve the model generalization.

Model Selection: To avoid Underfitting and Overfitting 

Strategy of avoiding overfit

- Increase Sample
- Remove Outliers
- Decrease model complexity
- Train-Validation-Test
- Regularization 

Model Selection and Regularization

- Structural risk: Empirical risk + regularization 

  $\frac{1}{N}\displaystyle\sum_{i=1}^NL(y_i,F(x_i))+\lambda\phi(F)$

  $\phi(F)$ measures model complexity, aiming at selecting a model that can fit the current data as simple as possible

  $\lambda$ is the trade-off between model fitness and model complexity

- Choice of $\phi(F)$

  - $l_2$ norm: $L(\beta)=\frac{1}{N}\displaystyle\sum_{i=1}^NL(Y_i,F(X_i|\beta))+\frac{\lambda}{2}||\beta||_2$
  - $l_1$ norm: $L(\beta)=\frac{1}{N}\displaystyle\sum_{i=1}^NL(Y_i,F(X_i|\beta))+\frac{\lambda}{2}||\beta||_1$

## Classification

### Nearest Neighbor Classifiers

Basic idea: If it walks like a duck, quacks like a duck, then it’s probably a duck

K-nearest neighbors of a record x are data points that have the k smallest distance to x

Predict class label of test instance with major vote strategy

Remarks:

✅ Highly effective method for noisy training data

✅ Target function for a whole space may be described as a combination of less complex local approximations

✅ Learning is very simple (lazy learning)

❎ Classification is time consuming

❎ Difficult to determine the optimal k

❎ Curse of Dimensionality

### Naïve Bayes

Given training data  $X$, posteriori probability of a hypothesis $H$, $P(H|X)$ follows the Bayes theorem $P(H|X)=\frac{P(X|H)P(H)}{P(X)}$

Predicts $X$ belongs to $C_2$ iff the probability $P(C_2|X)$ is the highest among all the $P(C_k|X)$ for all the $k$ classes

Practical difficulty: require initial knowledge of many probabilities, significant computational cost

Class Conditional independent :
$$
\begin{align}
& P(X|C_i)=\displaystyle\prod_{k=1}^nP(x_k|C_i) \\
& P(C_i|X)=\frac{P(X|C_i)P(C_i)}{P(X)}=\frac{P(C_i) \displaystyle \prod_{k=1}^n P(x_k|C_i) }{P(X)} \\
& \underset{i}{arg\; max}\,P(C_i|X)=P(C_i)\displaystyle\prod_{k=1}^nP(X_i|C_i)
\end{align}
$$

### Decision Tree

$Information\;Gain(A) =Entropy(S)-\displaystyle\sum_{v\in Values(A)}\frac{|S_v|}{|S|}·Entropy(S_v) $

$Entropy=-\displaystyle\sum_{d\in Decisions}p(d)log((p(d)))$

### Support Vector Machine

There are infinite lines (hyperplanes) separating the two classes but we want to find the best one (the one that minimizes classification error on unseen data)

SVM searches for the hyperplane with the largest margin, i.e., maximum marginal hyperplane (MMH)

The data point closest to the separation hyperplane in the sample points of the training data set is called the support vector. Only the support vector plays a role in determining the optimal hyperplane, while the other data points do not.Moving or even deleting the non-support vectors does not have any effect on the optimal hyperplane. In other words, the support vector plays a decisive role in the model

- SVM—Linearly Separable


A separating hyperplane can be written as: $W\cdot X+b=0$, The hyperplane defining the sides of the margin: $H_1:w_0+w_1x_1+w_1x_1\geq1,for \; y_i=+1; H_2:w_0+w_1x_1+w_1x_1\leq-1,for \; y_i=-1$

Any training tuples that fall on hyperplanes $H_1$ or $H_2$ (i.e., the sides defining the margin) are support vectors.

This becomes a constrained (convex) quadratic optimization problem: 
$$
margin=\underset{w,b}{max}\frac{2}{||w||}\Longleftrightarrow\underset{w,b}{min}\frac{1}{2}||w||^2 \\
s.t. \; y_i(w\cdot x_i+b)\geq1,\; i=1,2,\cdots,N \\
\\
\text{1、Constrained optimization problem to non-constrained problem with augmented Lagrangian multipliers
} \\
L(w,b,\alpha)=\frac{1}{2}||w||^2-\displaystyle\sum_{i=1}^N\alpha_i(y_i(w\cdot x_i+b)-1) \\
2、Let \; \theta(w)=\underset{\alpha_i\geq0}{max}L(w,b,\alpha)  \\
\theta(w)=
\begin{cases}
\frac{1}{2}||w||^2, & y_i(w\cdot x_i+b)\geq1 \\
+\infty, & y_i(w\cdot x_i+b)\leq1
\end{cases}
\Longrightarrow
\underset{w,b}{min}\;\theta(w)=\underset{w,b}{min}\;\underset{\alpha_i\geq0}{max}L(w,b,\alpha) =p^*\\ \\
\text{Because Lagrange duality},\quad \underset{w,b}{min}\;\theta(w)=\underset{w,b}{min}\;\underset{\alpha_i\geq0}{max}L(w,b,\alpha) =p^* \rightarrow \underset{\alpha_i\geq0}{max}\; \underset{w,b}{min}L(w,b,\alpha) =d^* \\

\text{KTT Conditions}
\begin{cases}
\alpha_i \geq 0 \\
y_i(w_i\cdot x_i+b) - 1 \geq 0 \\
\alpha_i(y_i(w_i\cdot x_i+b) - 1) = 0, \quad \text{$\alpha_i$ is support vector }
\end{cases}
$$

$$
\begin{align}
L(w,b,\alpha)
& =\frac{1}{2}\displaystyle\sum_{i=1}^N\sum_{j=1}^N \alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^N\alpha_iy_i((\sum_{j=1}^N\alpha_jy_jx_j)\cdot x_i+b)+\sum_{i=1}^N\alpha_i  \\
& = -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i\alpha_jy_iy_j(x_i\cdot x_j)+\sum_{i=1}^N\alpha_i \\ \\

& \underset{w,b}{min}L(w,b,\alpha)=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i\alpha_jy_iy_j(x_i\cdot x_j)+\sum_{i=1}^N\alpha_i \\

& \underset{\alpha}{max} -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i\alpha_jy_iy_j(x_i\cdot x_j)+\sum_{i=1}^N\alpha_i \\
& s.t. \sum_{i=1}^N\alpha_iy_i=0, \quad\alpha_i\geq0,i=1,2,\cdots,N
\end{align}
$$

- SVM—Linearly Inseparable


Transform the original input data into a higher dimensional space

Search for a linear separating hyperplane in the new space

Kernel Trick: Instead of computing the dot product on the transformed data tuples, it is mathematically equivalent to instead applying a kernel function $K(X_i,X_j)$ to the original data, i.e., $K(x,z)=\phi(x)\phi(z)$
$$
\begin{aligned}
\text{Typical Kernel Functions: } \\

\text{Polynomial kernel: }
&\quad k(x,z)=(x\cdot z+1)^p,p\geq1
\\
\text{Gaussian kernel(RBF):}
&\quad k(x,z)=exp(-\frac{||x-z||^2}{2\theta^2})
\\
\text{Laplace kernel:}
&\quad k(x,z)=exp(-\frac{||x-z||^2}{\theta}),\theta>0
\\
\text{Sigmoid kernel:}
&\quad k(x,z)=tanh(\beta x\cdot z+\theta),\beta>0,\theta<0
\end{aligned}
$$

$$
SVM+Kernel \quad 
\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i\alpha_jy_iy_j(\phi(x_i)\cdot \phi(x_j))-\sum_{i=1}^N\alpha_i \\
s.t. \; \sum_{i=1}^N\alpha_iy_i=0;\quad 0\leq\alpha_i\leq C \\
\text{Objective function}\quad \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i\alpha_jy_iy_j\cdot k(x,z)-\sum_{i=1}^N\alpha_i \\
Classifier \quad sign(\sum_{i=1}^N\alpha_iy_i(x_i\cdot x+1)^p+b) \\
Kmeans+Kernel \quad \underset{H^TH=1,H\geq0}{max}Tr(H^TX^TXH) \\
PCA+Kernel \quad C_F=\frac{1}{N}\phi(X)[\phi(X)]^T=\frac{1}{N}\sum_{i=1}^N\phi(x_i)\phi(x_i)^T
$$

## Ensemble Learning

- Criteria: Good performance + Diversity

- Strategies: Bagging、Boosting、Stacking

- Rationale for Ensemble Learning
  - No Free Lunch thm: There is no algorithm that is always the most accurate
  - Generate a group of base-learners which when combined have higher accuracy
  - Different learners use different：Algorithms、Parameters、Representations (Modalities)、Training sets、Subproblems

### Bagging - Aggregate Bootstrapping

Given a standard training set $D$ of size $n$
For i = 1 ... M
	Draw  a sample of size $n*<n$ from $D$ uniformly and with replacement
	Learn classifier $C_i$
Final classifier is a vote of $C_1\cdots C_M$ 
Increases classifier stability/reduces variance

- Random Forest
  - Ensemble consisting of a bagging of un-pruned decision tree learners with a randomized selection of features at each split.
  - Grow many trees on datasets sampled from the original dataset with replacement (a bootstrap sample). 
    - Draw K bootstrap samples of a fixed size
    - Grow a DT, randomly sampling a few attributes/dimensions to split on at each internal node
  - Average the predictions of the trees for a new query (or take majority vote)

### Boosting

Train classifiers (e.g. decision trees) in a sequence

A new classifier should focus on those cases which were incorrectly classified in the last round.

Combine the classifiers by letting them vote on the final prediction (like bagging).

Each classifier is “weak” but the ensemble is “strong.”

AdaBoost is a specific boosting method.

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Adaboost.png)

### Stacking

<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Stacking%20Framework.png" style="zoom: 50%;" />

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Stacking.png)

## Clustering

### K-means Clustering

- Partitional clustering approach 

  Each cluster is associated with a centroid (center point) 

  Each point is assigned to the cluster with the closest centroid

  Number of clusters, K, must be specified

  The basic algorithm is very simple

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/K-means%20Clustering.png)

- Solutions to Initial Centroids Problem
  - Multiple runs (Helps, but probability is not on your side)
  - Sample and use hierarchical clustering to determine initial centroids
  - Select more than k initial centroids and then select among these initial centroids (Select most widely separated)
- Evaluating K-means Clusters. Most common measure is Sum of Squared Error (SSE)
  - For each point, the error is the distance to the nearest cluster
  - To get SSE, we square these errors and sum them. $SSE=\displaystyle\sum_{i=1}^K\sum_{x\in C_i}dist^2(m_i,x)$
  - $x$ is a data point in cluster $C_i$ and  $m_i$ is the representative point for cluster $C_i$ (can show that $m_i$ corresponds to the center (mean) of the cluster)
  - Given two clusters, we can choose the one with the smallest error
  - One easy way to reduce SSE is to increase K (the number of cluster)
  - A good clustering with smaller K can have a lower SSE than a poor clustering with higher K

### Hierarchical Clustering 

- Produces a set of nested clusters organized as a hierarchical tree

  Can be visualized as a dendrogram (A tree like diagram that records the sequences of merges or splits)

- Strengths of Hierarchical Clustering
  - Do not have to assume any particular number of clusters. Any desired number of clusters can be obtained by ‘cutting’ the dendogram at the proper level
  - They may correspond to meaningful taxonomies. 
- Two main types of hierarchical clustering (Similarity or distance matrix is crucia)
  - Agglomerative:  Start with the points as individual clusters. At each step, merge the closest pair of clusters until only one cluster (or k clusters) left
  - Divisive: Start with one, all-inclusive cluster. At each step, split a cluster until each cluster contains a point (or there are k clusters).

Basic algorithm (Key operation is the computation of the similarity of two clusters)

1. Compute the proximity matrix
2. Let each data point be a cluster
3. Repeat
4. ​		Merge the two closest clusters
5. ​		Update the proximity matrix
6. Until only a single cluster remains

Note: Different approaches to defining the distance between clusters distinguish the different algorithms

### Density-based Clustering: DBSCAN

- Clustering based on density (local cluster criterion), such as density-connected points
- Major features: 
  - Discover clusters of arbitrary shape
  - Handle noise
  - One scan
  - Need density parameters as termination condition
- Several interesting studies:
  - DBSCAN: Ester, et al. (KDD’96)
  - OPTICS: Ankerst, et al (SIGMOD’99).
  - DENCLUE: Hinneburg & D. Keim  (KDD’98)
  - CLIQUE: Agrawal, et al. (SIGMOD’98)
- DBSCAN Key concepts
  - Density = number of points within a specified radius (Eps)
  - A point is a core point if it has more than a specified number of points (MinPts) within Eps [These are points that are at the interior of a cluster]
  - A border point has fewer than MinPts within Eps, but is in the neighborhood of a core point
  - A noise point is any point that is not a core point or a border point
- Two parameters:
  - Eps: Maximum radius of the neighbourhood
  - MinPts: Minimum number of points in an Eps-neighbourhood of that point
- $N_{Eps}(p):\{q\; belongs \; to\; D |dist(p,q)\leq Eps\}$
- Directly density-reachable: A point $p$ is directly density-reachable from a point $q$ wrt. $Eps, MinPts$ if
  - $p$ belongs to $N_{Eps}(q)$
  - core point condition: $|N_{Eps}(q)\geq MinPts|$
- Density-reachable:  A point $p$ is density-reachable from a point $q$ wrt. Eps, MinPts if there is a chain of points $p_1,\cdots,p_n,\;p_1=q,p_n=p$ such that $p_{i+1}$ is directly density-reachable from $p_i$
- Density-connected: A point $p$ is density-connected to a point $q$ wrt. Eps, MinPts if there is a point $o$ such that both, $p$ and $q$ are density-reachable from $o$ wrt. Eps and MinPts.
- A cluster is defined as a maximal set of density-connected points

## Subspace Learning

### Motivation

- Curse of dimensionality (Neighbors Search). Similarity Calculation is a difficult thing for high-dimensional data.
- Curse of dimensionality (Classification)
  - The required number of samples (to achieve the same accuracy) grows exponentially with the number of variables
  - In practice: number of training examples is fixed. => the classifier’s performance usually will degrade for a large number of features

### Dimension Reduction

- Linear methods

  - Principal component analysis (PCA)
  - Multidimensional scaling (MDS)

- Nonlinear methods Goal: to unfold, rather than to project (linearly)

  - Locally linear embedding (LLE)
  - Laplacian eigenmaps (LEM)
  - Isomap
  - Stochastic Neighbor Embedding (SNE)

- Principal Component Analysis (PCA)

  - Find projections that capture the largest amounts of variation in data

  - Find the eigenvectors of the covariance matrix, and these eigenvectors define the new space

  - Definition: Given a set of data $X\in R^{d\times N}$, find the principal axes are those orthonormal axes onto which the variance retained under projection is maximal

  - Formulation

    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/PCA.png)

- Multidimensional Scaling (MDS)

  - Attempts to preserve pairwise distances

  - Different formulation of PCA, but yields similar result form
    $$
    \underset{Y}{min} \sum_{i=1}^N\sum_{j=1}^N(d_{ij}^{(X)}-d_{ij}^{(Y)})^2 \\
    where \; d_{ij}^{(X)}=||x_i-x_j||^2 \; and \; d_{ij}^{(Y)}=||y_i-y_j||^2
    $$


- Locally Linear Embedding (LLE)

  Procedure: 

  Identify the neighbors of each data point

  Compute weights that best linearly reconstruct the point from its neighbors $\underset{w}{min}\displaystyle\sum_{i=1}^N||x_i-\sum_{j=1}^kw_{ij}x_{N_{i(j)}}||^2$

  Find the low-dimensional embedding vector which is best reconstructed by the weights determined in Step 2 $\underset{w}{min}\displaystyle\sum_{i=1}^N||y_i-\sum_{j=1}^kw_{ij}y_{N_{i(j)}}||^2 \; \Leftrightarrow \; \underset{Y}{min}\;tr(Y^TYL)$, Centering Y with unit variance

- Laplacian Eigenmaps (LEM)

  - Similar to locally linear embedding

  - Different in weights setting and objective function
    $$
    Weights: \quad W_{ij}=
    \begin{cases}
    1, & \text {i, j are connected} \\
    exp(\frac{-||x_i-x_j||^2}{s}), & \text {otherwise}
    \end{cases}
    $$
    Objective:  $\underset{Y}{min}\displaystyle\sum_{i=1}^N\sum_{j=1}^N(y_i-y_j)^2W_{ij}\;\Leftrightarrow\; \underset{Y}{min}\;tr(YLY^T)$, where $L=R-W, R$ is diagonal and $R_{ii}=\sum_{j=1}^NW_{ij}$

- Isometric Feature Mapping (ISOMAP)

  - Construct the neighborhood graph
  - Compute the shortest path length (geodesic distance) between pairwise  data points
  - Recover the low-dimensional  embeddings of the data by Multi-Dimensional Scaling (MDS) with  preserving those geodesic distances

- Stochastic Neighbor Embedding (SNE)

  - The probability that $i$ picks $j$ as its neighbor, $p_{ij}=\frac{exp(-d_{ij}^2)}{\sum_{k≠i}exp(-d_{ik}^2)}$
  - Neighborhood distribution in the embedded space $q_{ij}\frac{exp(-||y_i-y_j||^2)}{\sum_{k≠i}exp(-||y_i-y_k||^2)}$
  - Neighborhood distribution preservation $\zeta=\displaystyle\sum_{ij}p_{ij}log\frac{p_{ij}}{q_{ij}}=\sum_iKL(P_i||Q_i)$

### Feature Selection

- Motivation

  - Especially when dealing with a large number of variables there is a need for dimensionality reduction
  - Feature Selection can significantly improve a learning algorithm’s performance

- Feature Subset Selection Goal: Find the optimal feature subset. (or at least a “good one.”)

- Classification of methods:

  - Filters
  - Wrappers
  - Embedded Methods

- Filter Methods Select subsets of variables as a pre-processing step,independently of the used classifier

- Wrapper Methods

  <img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/Wrapper%20Methods.png" style="zoom:50%;" />

- Embedded Methods

  - Specific to a given learning machine

  - Performs variable selection (implicitly) in the process of training

  - LASSO

    <img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/LASSO.png" style="zoom:50%;" />

### Subspace Clustering

- Challenge: Traditional clustering algorithms are inappropriate to handle high-dimensional data, due to the “curse of dimensionality”.

  What is curse of dimensionality?

  1⃣️ The more features we have, the more difficult we process it.

  2⃣️ Distances among points are disappear. $\underset{d\to\infty}{lim}\frac{dist_{max}-dist_{min}}{dist_{min}}\to 0$

  3⃣️ Existing relevant and irrelevant attributes. However, The relevance of certain attributes differ for different groups of objects within the same dataset. Thus global feature reduction methods are failed!

  4⃣️ There are correlations among subsets of attributes. There are redundant attributes in the data set. Somehow it can be regarded as a breakthrough point for subspace clustering. 

- PCA-based algorithms: 4C
  <img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Machine%20Learning/PCA-based%20algorithms%204C.png" style="zoom:50%;" />

- Sparse subspace clustering (SSC)

  $\underset{Z}{min}\;||Z||_0, \;s.t.X=XZ,diag(Z)=0$

  $\underset{Z}{min}\;||Z||_1, \;s.t.X=XZ,diag(Z)=0$ (convex relaxation)

  Affinity matrix: $W=\frac{1}{2}(|Z|+|Z|^T)$

- Low-rank representation (LRR)

  $\underset{Z}{min}\;rank(Z), \;s.t.X=XZ$

  $\underset{Z}{min}\;||Z||_*, \;s.t.X=XZ$ (convex relaxation)

  The solution Z to the LRR is block diagonal when the  subspaces are independent.



# Hashing 

Challenge in big data applications: Curse of dimensionality、Storage cost、Query speed

## Case Study, Finding Similar Documents

- Given a body of documents, e.g., the Web, find pairs of documents with a lot of text in common, e.g.:
  - Mirror sites, or approximate mirrors. Application: Don’t want to show both in a search.
  - Plagiarism, including large quotations.
  - Similar news articles at many news sites. Application: Cluster articles by “same story.”
- Three Essential Techniques for Similar Documents
  - Shingling : convert documents, emails, etc., to sets.
  - Minhashing : convert large sets to short . signatures, while preserving similarity.
  - Locality-sensitive hashing : focus on pairs of signatures likely to be similar.

<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Case%20Study%20Finding%20Similar%20Documents.png" style="zoom:50%;" />

## Shingles

- A k -shingle (or k -gram) for a document is a sequence of k characters that appears in the document.

  Example: k=2; doc = abcab.  Set of 2-shingles = {ab, bc, ca}. Option: regard shingles as a bag, and count ab twice.

- Represent a doc by its set of k-shingles.

- Assumption
  - Documents that have lots of shingles in common have similar text, even if the text appears in different order.
  - Careful: you must pick k  large enough, or most documents will have most shingles. k = 5 is OK for short documents; k = 10 is better for long documents.

## Min-Hashing

- Basic Data Model: Sets

  Many similarity problems can be couched as finding subsets of some universal set that have significant intersection.

  Examples include: Documents represented by their sets of shingles (or hashes of those shingles). Similar customers or products.

- Jaccard Similarity of Sets

  The Jaccard similarity  of two sets is the size of their intersection divided by the size of their union.

  $Sim(C_1,C_2)=\frac{C_1\cap C_2}{C_1\cup C_2}$

- From Sets to Boolean Matrices

  - Rows = elements of the universal set.
  - Columns = sets.
  - 1 in row $e$ and column $S$  if and only if $e$ is a member of $S$.
  - Column similarity is the Jaccard similarity of the sets of their rows with 1.
  - Typical matrix is sparse.

  <img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Example%20Jaccard%20Similarity%20of%20Columns.png" style="zoom:25%;" />

- When Is Similarity Interesting?

  - When the sets are so large or so many that they cannot fit in main memory.
  - Or, when there are so many sets that comparing all pairs of sets takes too much time.
  - Or both above.

- Outline of Min-Hashing 

  - Compute signatures of columns = small summaries of columns.

  - Examine pairs of signatures to find similar signatures.

    Essential: similarities of signatures and columns are related.

  - Optional: check that columns with similar signatures are really similar.

- Signatures

  - Key idea: “hash” each column C  to a small signature Sig (C), such that:

    Sig (C) is small enough that we can fit a signature in main memory for each column.

    $Sim(C_1,C_2)$ is the same as the “similarity” of $Sig(C_1)$ and $Sig(C_2)$

  - Imagine the rows permuted randomly.

    Define “hash” function $h(C)$ = the number of the first (in the permuted order) row in which column $C$ has 1

    Use several (e.g., 100) independent hash functions to create a signature.

  - The probability (over all permutations of the rows) that $h(C_1)$ = $h(C_2)$ is the same as $Sim(C_1, C_2)$.

- Minhash Signatures

  - Pick (say) 100 random permutations of the rows.
  - Think of $Sig(C)$ as a column vector.
  - Let $Sig(C)[i]$ = according to the  $i_{th}$ permutation, the number of the first row that has a 1 in column C. 

  <img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Example%20Min%20Hashing.png" style="zoom:50%;" />

  Input Matrix: d $\times$ N, $\Longrightarrow$ Signature Matrix: d' $\times$ N

- Implementation 

  - Suppose 1 billion rows. 

    Hard to pick a random permutation from 1…billion. 

    Representing a random permutation requires 1 billion entries. 

    Accessing rows in permuted order leads to thrashing.

  - A good approximation to permuting rows: pick 100 (?) hash functions.

    For each column $c$ and each hash function $h_i$,  keep a “slot” $M(i,c)$

    Intent: $M(i,c)$ will become the smallest value of $h_i(r)$ for which column $c$  has 1 in row $r$.

    I.e., $h_i(r)$ gives order of rows for $i_{th}$ permuation.
    
    $i$ could denote permutation , $c$ could denote documents 
  
  ```c
  Initialize M(i,c) to ∞ for all i and c
  for each row r
    for each column c 
      if c has 1 in row r
        for each hash function h_i do
          if h_i(r) is a smaller value than M(i,c)
            M(i,c):= h_i(r);
  ```

  <img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Example%20Minhashing%20Implementation.png" style="zoom:50%;" />

  

  - Often, data is given by column, not row.

    E.g., columns = documents, rows = shingles.
  
    If so, sort matrix once so it is by row.
  
    And always  compute $h_i(r)$ only once for each row.

## Locality-Sensitive Hashing, Find Similar Items

Many Web-mining problems can be expressed as finding “similar” sets:
- Pages with similar words, e.g., for classification by topic.
- NetFlix users with similar tastes in movies, for recommendation systems.
- Movies with similar sets of fans.
- Images of related things.

Suppose we have, in main memory, data representing a large number of objects.

May be the objects themselves. May be signatures as in minhashing.

We want to compare each to each, finding those pairs that are sufficiently similar.

Checking All Pairs is Hard: While the signatures of all columns may fit in main memory, comparing the signatures of all pairs of columns is quadratic in the number of columns.

**General idea:** Use a function $f(x,y)$ that tells whether or not x and y  is a **candidate pair**(a pair of elements whose similarity must be evaluated)

For minhash matrices: Hash columns to many buckets, and make elements of the same bucket candidate pairs.

Candidate Generation From Minhash Signatures

- Pick a similarity threshold $s$, a fraction < 1

- A pair of columns $c$ and $d$ is a candidate pair if their signatures agree in at least fraction $s$ of the rows

  I.e. M(i,c) = M(i,d) for at least fraction $s$ values of $i$

LSH for Minhash Signatures

- Big idea: hash columns of signature matrix $M$ several times.
- Arrange that (only) similar columns are likely to hash to the same bucket.
- Candidate pairs are those that hash at least once to the same bucket.
- Trick: divide signature rows into bands. Each hash function based on one band.

Partition into Bands

<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/partition%20into%20bands.png" style="zoom:33%;" />

- Divide matrix $M$ into $b$ bands of $r$ rows.
- For each band, hash its portion of each column to a hash table with $k$ buckets. Make $k$ as large as possible.
- Candidate column pairs are those that hash to the same bucket for ≥ 1 band.
- Tune $b$ and $r$ to catch most similar pairs, but few dissimilar pairs.

<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Matrix%20and%20Buckets.png" style="zoom:33%;" />

- LSH involves a Tradeoff

  Pick the number of minhashes, the number of bands, and the number of rows per band to balance false positives/negatives.

  Example: if we had only 15 bands of 5 rows, the number of false positives would go down, but the number of false negatives would go up.

  if $b$ Brands of $r$ Rows are given 
  <img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/LSH%20tradeoff.png" style="zoom: 50%;" /> 

## Learn to Hash

- Radom Projection

  The hashing function of LSH to produce Hash Code
  $$
  h_r(x)=
  \begin{cases}
  1, & r^Tx \geq 0 \\
  0, & otherwise
  \end{cases}
  $$
  $r^Tx \geq 0$ is a hyperplane separating the space

  Assume we have already learned a distance metric A from domain knowledge. $X^TAX$ has better quantity than simple metrics such as Euclidean distance.

  Take random projections of data. Quantize each projection with few bits

  <img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Hash-Radom%20Projection.png" style="zoom:50%;" />

- PCA Hashing

  1. Projection Stage: 

     Get Transform matrix $W$, we can project $X$ on new feature space.

     $Y=W^TX$

  2. Quantization Stage

     $h(x)=sign(W^Tx)$

     The strategy is too simple, The biggest problem is that assigning same bits to directions along which the data has a greater range. 

  Minimize the quantization loss: $Q(B,Y)=||B-Y^TR||_F^2$, $R$ is orthogonal matrix , $B=sign(Y^TR)$

  The basic idea is rotating the data to minimize quantization loss.

  Solution: Beginning with the random initialization of R, and adopt a k-means-like procedure. In each iteration, each data point is first assigned to the nearest vertex of the binary hypercube, and then R is updated to minimize the quantization loss given this assignment.

- Spectral Hashing
  $$
  \underset{\{y_i\}}{min}\sum_{ij}W_{ij}||y_i-y_j||^2 \\
  subject\;to. 
  \begin{align}
  & y_i \in \{-1,1\}^k \\
  & \sum_iy_i=0 \\
  & \frac{1}{n}\sum_iy_iy_i^T= 1
  \end{align}
  $$
  
  where $W_{ij}$ is the similarity between $x_i$ and $x_j$, the constraint $\sum_iy_i=0$ requires each bit to be fire 50% of the time, and the constraint $\frac{1}{n}\sum_iy_iy_i^T= 1$ requires the bits to be uncorrelated

- General  Approach to Learning-Based Hashing 

  Decomposing the hashing learning problem into two steps:

  1.  hash bit learning
  2. and hash function learning based on the learned bits. 

  <img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/General%20%20Approach%20to%20Learning-Based%20Hashing.png" style="zoom:50%;" />

# Sampling

## Inverse Transform Sampling

Sampling based on the inverse of Cumulative Distribution Function (CDF，累积分布函数) 

- CDF Sampling
  - $Y_i\sim Uniform(0,1)$
  - $X_i=CDF^{-1}(Y_i)$

Drawbacks：Usually, it’s hard to get the inverse function

## Rejection Sampling

Idea: Accept the samples in the region under the graph of its density function and reject others

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Rejection%20Sampling.png)

Proposal distribution $q(x)$ should always covers the target distribution $p(x)$

Acceptance ratio = $\frac{p(x)}{Mq(x)}$, $M$ is a big positive number

## Importance Sampling

Basic Idea: Not reject but assign weight to each instance so that the correct distribution is targeted.
$$
\begin{align}
E(f(x)) 
& = \sum_{All \; x}f(x)\cdot p(x) = \int^{+\infty}_{-\infty}{f(x)p(x)dx} \\
& = \int^{+\infty}_{-\infty}{f(x)\frac{p(x)}{q(x)}\cdot q(x)dx}, \; let \; R(x)=f(x)\frac{p(x)}{q(x)} \\
& = \int^{+\infty}_{-\infty}{R(x)\cdot q(x)dx} = E(R(x)) \\
& = \sum_{All\;x}R(x)\cdot q(x)  \\
& \approx \frac{1}{n}\sum_{All\;x}R(x) \\
& \text{n is very large, 大数定理} \; E(x)\approx \overline{X}=\frac{1}{n}\sum x_i \\
& = \frac{1}{n}\sum_{All\;x}f(x)\cdot w(x), \text{where $w(x)$ is}\; \frac{p(x)}{q(x)}
\end{align}
$$

- Importance Sampling (IS) V.S. Rejection Sampling(RS)
  - Instances from RS share the same “weight”, only some of instances are reserved
  - Instances from IS have different weight, all instances are reserved
  - IS is less sensitive to proposal distribution 

## Markov chain Monte Carlo (MCMC)

MCMC methods are a class of algorithms for sampling from a probability distribution based on constructing a Markov chain that has the desired distribution as its equilibrium distribution. The state of the chain after a number of steps is then used as a sample of the desired distribution. 

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Markov%20chain%20Monte%20Carlo.png)

A Markov chain is a sequence of random variab $x_1,x_2,x_3,\cdots$ with Markov property, namely that, given the present state, the future and past states are independent. 

$Pr(X_{n+1}=x|X_1=x_1,X_2=x_2,\cdots,X_n=x_n)=Pr(X_{n+1}=x|X_n=x_n)$

Utilize MCMC to generate a Markov chain, such that we have a Markov chain $\{X_1,X_2,\cdots,X_i,X_{i+1},\cdots,X_n\}$

$X_1,X_2,\cdots,X_i$ is the burn in period, $X_{i+1},\cdots,X_n$ is sampling. if $n$ is large enough, $X_n\sim p(X)$

Example: $X=\{e,t,w\},\; \widehat{p}(X)=\widehat{p}(e,t,w)=\displaystyle\sum_{i=n-1000}^n\frac{X_i}{1000}$

**Detailed Balance Condition(细致平衡条件)**

**Theorem:** If the transition matrix P of non-periodic Markov chain and the distribution of π(x) satisfy $\pi(i)P_{ij}=\pi_jP_{ji},\; for\; all\;i,j$
The $\pi(x)$ is the equilibrium distribution, where $P_{ij}$ means the probability of transiting from state $i$ to state $j$, and $\pi(i)$ is the probability of been state $i$

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Detailed%20Balance%20Condition.png)

Given p(X), we target to find a transition matrix Q(X), such that: $p(X_i)Q(X_j|X_i)=P(X_j)Q(X_i|X_j),\; for\; all\;i,j$

But, find distribution $Q(x|y)$ is very difficult. There is a solution, for example, we don't know whether $a=b$, but we can confirm $ab=ba$

Then loosen the condition by introducing the acceptance ratio $\alpha$, so that
$$
P(X_i)Q(X_j|X_i)\alpha(X_j|X_i)=P(X_j)Q(X_i|X_j)\alpha(X_i|X_j) \\
where\;
\begin{cases}
\alpha(X_j|X_i) = P(X_j)Q(X_i|X_j) \\
\alpha(X_i|X_j) = P(X_i)Q(X_j|X_i)
\end{cases} 
\\
P(X_i)\underset{Q'(X_j|X_i)}{\underbrace{Q(X_j|X_i)\alpha(X_j|X_i)}}=P(X_j)\underset{Q'(X_i|X_j)}{\underbrace{Q(X_i|X_j)\alpha(X_i|X_j)}}
$$
Therefore, $Q'(X_j|X_i)=Q(X_j|X_i)P(X_j)Q(X_i|X_j)$, because $P(X_j)Q(X_i|X_j) \in[0,1]$, so $Q'(X_j|X_i)<Q(X_j|X_i)$, then using rejection sampling for $Q'$

Above all, the mcmc samlping algorithm is:

<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/MCMC%20Sampling%20Algorithm.png" style="zoom:50%;" />

## Metropolis–Hastings (MH) Sampling

Equivalence：

$p(X_i)q(X_j|X_i)*0.1=p(X_j)q(X_i|X_j)*0.2$

$p(X_i)q(X_j|X_i)*0.5=p(X_j)q(X_i|X_j)*1.0$

Idea: Magnify acceptance ratio by $\alpha(X_j|X_i)=min(1,\frac{p(X_j)q(X_i|X_j)}{p(X_i)q(X_j|X_i)})$

Note that it wouldn’t violate the detailed balance condition
$$
\begin{align}
p(X_i)q(X_j|X_i)\alpha(X_j|X_i)
& = p(X_i)q(X_j|X_i)\cdot min(1,\frac{p(X_j)q(X_i|X_j)}{p(X_i)q(X_j|X_i)}) \\ 
& = min(p(X_i)q(X_j|X_i),p(X_j)q(X_i|X_j)) \\
& = p(X_j)q(X_i|X_j)\cdot min(\frac{p(X_i)q(X_j|X_i)}{p(X_j)q(X_i|X_j)}) \\
& = p(X_j)q(X_i|X_j)\alpha(X_i|X_j)
\end{align}
$$
<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/MH%20Sampling.png" style="zoom:50%;" />

## Gibbs Sampling

Idea: MH has large acceptance ratio, Gibbs sampling further make acceptance ratio being 100%. Using conditional distribution and marginal distribution to satisfy detailed balance condition which is necessary for markov chain.

Two-dimension example: Given a joint distribution $p(x,y)$, corresponding to the point in the figure below, $A(x_1,y_1)$ and $B(x_1,y_2)$
<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Gibbs%20Sampling%20Example.png" style="zoom: 50%;" />

<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Gibbs%20Sampling.png" style="zoom:50%;" />

Like every other MCMC style algorithm, Gibbs sampling still have the burn-in period

- Example: $X=\{e,t,w\}$
  - Step 1: Initialize $e,t,w$, such that we have $X_0=(e_0,t_0,w_0)$
  - Step 2: generate $e_1$ based on $p(e|t=t_0,w=w_0)$
  - Step 3: generate $t_1$ based on $p(t|e=e_1,w=w_0)$
  - Step 4: generate $w_1$ based on $p(w|e=e_1,t=t_1)$
  - Step 5: repeat steps 2-4 n times, and we obtain a Markov chain

## Latent Dirichlet Allocation

Please refer to: https://zhuanlan.zhihu.com/p/172021192

Documents are represented as random mixtures over latent topics, where each topic is characterized by a distribution over all the words.

<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Latent%20Dirichlet%20allocation%201.png" style="zoom: 67%;" />

<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Latent%20Dirichlet%20allocation%202.png" style="zoom: 50%;" />

<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Latent%20Dirichlet%20allocation%203.png" style="zoom:67%;" />

## Sampling on Data Stream: Reservoir Sampling

- Fixed $k$ uniform sample from arbitrary size $N$ stream in one pass

  - No need to know stream size in advance
  - Include first $k$ items with probability 1
  - Include item $n > k$  with probability $p(n) = k/n, n > k $
    - Pick $j$ uniformly from $\{1,2,…,n\}$
    - If $j\leq k$, swap item $n$ into location $j$ in reservoir, discard replaced item

- Proof 

  - n-item selection probability $k/n$

  - n-item reservation

    - n+1: $1 -\frac{k}{n+1}\cdot\frac{1}{k}=\frac{n}{n+1}$ 
    - n+1: $1 -\frac{k}{n+2}\cdot\frac{1}{k}=\frac{n+1}{n+2}$ 
    - N: $1 -\frac{k}{N}\cdot\frac{1}{k}=\frac{N-1}{N}$ 

    $\frac{n}{n+1}\cdot \frac{n+1}{n+2} \cdots \frac{N-1}{N}=\frac{n}{N}$

    $\frac{k}{n}\cdot\frac{n}{N}=\frac{k}{N}$

  

# Data Stream Mining

## Concept & Features 

- A data stream is a massive sequence of data objects which have some unique features:

  - One by One
  - Potentially Unbounded 
  - Concept Drift 

- Data Stream: Infinite Length, Evolving Nature.

  Challenges :

  - Single Pass Handling

  - Memory Limitation

  - Low Time Complexity

  - Concept Drift: In predictive analytics and machine learning, the concept drift means that the statistical properties of the target variable, which the model is trying to predict, change over time in unforeseen ways. **In a word, the probability distribution changes.**

    - Change in $P(C)$, $C$ donates class
    - Change in $P(X)$, $X$ donates features 
    - Change in $P(C|X)$, Concept Drift, remain features, but class has changed 

    $P(C_i|X)=\frac{P(C_i)P(X|C_i)}{p(X)}$

    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Rea-Virtuall%20Concept%20Drift.png)
    
    <img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Example%20Concept-Drift.png" style="zoom: 50%;" />

## Concept Drift Detection

###  Distribution-based detector

Monitoring the change of data distributions between two  fixed or adaptive windows of data. (e.g. ADWIN)
<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Distribution-based%20detector.png" style="zoom: 50%;" />
Drawbacks: Hard to determine window size、Learn concept drift slower、Virtual concept drift

- Adaptive Windowing（ADWIN）

  The idea is simple: whenever two “large enough” subwindows of W exhibit “distinct enough” averages, one can conclude that the corresponding expected values are different,and the older portion of the window is dropped.

  <img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/ADWIN.png" style="zoom:67%;" />

### Error-rate based detector

Capture concept drift based on the change of the classification performance. (i.e. comparing the current classification performance to the average historical error rate with statistical analysis.) (e.g. PHT)
<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Error-rate%20based%20detector.png" style="zoom:67%;" />
Drawbacks: Sensitive to noise、Hard to deal with gradual concept drift、Depend on learning model itself heavily

- Drift  detection method: DDM

  A significant increase in the error of the algorithm, suggest a change in the class distribution, and whether is a significant increase  is based on following formula.

  $p_i+s_i \geq p_{min}+3*s_{min}$

  the error-rate is the probability of observed False $p_i$, with standard deviation given by $s_i=sqrt(p_i(1-p_i)/i)$

## Data Stream Classification

- Classification: Model construction based on training sets

- Typical classification methods
  - K-Nearest neighbor approach
  - Decision tree
  - Bayesian classification
  - Neural network approach
  - Support Vector Machines  
  - Other methods

- Data stream classification Circle

  <img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/Data%20stream%20classification%20Circle.png" style="zoom: 50%;" />
  Process an example at a time, and inspect it only once 
  Be ready to predict at any point
  Use a limited amount of memory 
  Work in a limited amount of time

  Procedure:

  The algorithm is passed the next available example from the stream (requirement 1) 
  The algorithm processes the example, updating its data structures. It does so without exceeding the memory bounds set on it (requirement 2), and as quickly as possible (requirement 3).
  The algorithm is ready to accept the next example. (requirement 4) 

- Typical algorithms

  - VFDT (Very Fast Decision Tree)
  - CVFDT
  - SyncStream

- Decision Tree Learning: One of the most effective and widely-used classification methods

  - Induce models in the form of decision trees: Each node contains a test on the attribute; Each branch from a node corresponds to a possible outcome of the test; Each leaf contains a class prediction
  - Challenges
    - Classic decision tree learners assume all training data can be simultaneously stored in main memory
    - Disk-based decision tree learners repeatedly read training data from disk sequentially. Prohibitively expensive when learning complex trees
    - Goal: design decision tree learners that read each example at most once, and use a small constant time to process it

  ### VFDT (Very Fast Decision Tree)

- A decision-tree learning system based on the Hoeffding tree algorithm.

- In order to find the best attribute at a node, it may be sufficient to consider only a small subset of the training examples that pass through that node.

  - Given a stream of examples, use the first ones to choose the root attribute.
  - Once the root attribute is chosen, the successive examples are passed down to the corresponding leaves, and used to choose the attribute there, and so on recursively.

- Use Hoeffding bound to decide how many examples are enough at each node 

  

- How many examples are enough?
  - $G(X_i):$ the heuristic measure used to choose test attributes (e.g. Information Gain, Gini Index)
  - $X_a:$ the attribute with the highest attribute evaluation value after seeing n examples
  - $X_b:$ the attribute with the second highest split evaluation function value after seeing n examples
  - Given a desired, if $\Delta \overline{G}=\overline{G}(X_a)-\overline{G}(X_b)>\epsilon$,  after seeing n examples at a node 
    - Hoeffding bound guarantees the true $\Delta \overline{G}-\epsilon>0$
    - This node can be split using $X_a$, the succeeding examples will be passed to the new leaves

- Hoeffding Bound

  - Hoeffding's inequality:  A result in probability theory that gives an upper bound on the probability for the sum of random variables to deviate from its expected value

  - Based on Hoeffding Bound principle, classifying different samples leads to the same model with high probability —can use a small set of samples

  - Hoeffding Bound (Additive Chernoff Bound)

    - Given: $r$ denotes random variable, $R$ denotes range of r, $N$ denotes independent observations

    - True mean of $r$ is at least $r_{avg}-\epsilon$, with the probaility $1-\epsilon$(where $\epsilon$ is user-specified)

      $\epsilon=\sqrt{\frac{R^2ln(1/\epsilon)}{2N}}$



- Algorithm
  - Calculate the information gain for the attributes and determines the best two attributes
  - At each node, check for the condition: $\Delta \overline{G}=\overline{G}(X_a)-\overline{G}(X_b)>\epsilon$
  - If condition satisfied, create child nodes based on the test at the node
  - If not, stream in more examples and perform calculations till condition satisfied

<img src="https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Big%20Data%20Analytics%20and%20Mining/VFDT.png" style="zoom: 33%;" />

- VFDT Strengths 
  - Scales better than traditional methods (Sublinear with sampling, Very small memory utilization)
  - Incremental (Make class predictions in parallel, New examples are added as they come)
- VFDT Weaknesses
  - Could spend a lot of time with ties
  - Memory used with tree expansion
  - Number of candidate attributes

### CVFDT (Concept-adapting Very Fast Decision Tree learner)





## Data Stream Clustering





# Graph Mining





# Hadoop-Spark





# Review

- What is big data?

- The 4V features in big data.

- What is  data mining?

- The main tasks of data mining?

- Main tasks in machine learning

- Loss function 

- Generalization: $L(M)=\sum_{i=1}^n(Y_i-Y_i^*)^2+\lambda \Phi(M)$

- How to avoid overfitting?

  - Increasing samples
  - Remove outlines
  - Train-Validation-Test
  - Decreasing the model complexity 
  - Regulization ($l_1$ norm, $l_2$ norm)

- Classical Algorithm (KNN、Naive Bases)

- Decision Tree

  - How to find the best split?

    Information Gain: $IG(x) = H(Y) -H(Y|X)$

  - Advantages 

- Support Vector Machine

  - Basic idea: class margin maximum 

  - Define **any** three parallel hyperplane   
    $$
    \begin{cases}
    wx+b=1 \\
    wx+b=0 \\
    wx+b=-1
    \end{cases}
    \Longrightarrow
    \begin{cases}
    y_i(wx_i+b) \geq 1, & \text {empirical risk} \\
    max\frac{1}{||w||}, & \text {margin maximum} \\
    min\frac{1}{2}||w||^2
    \end{cases}
    $$

- Kernel trick: $k(x,z)=\Phi(x)\cdot \Phi(z)$

- Ensemble Learning

  - Bagging -> Random Forest

  - Booting -> Adaboost / XGBoost

  - Stacking

- Clustering

  - K-Means

  - DBSCAN

- Subspace Learning

  - Dimension Reduction

    - Linear: PCA, MDS

    - Non-Linear: LLE, LEM, Isomap, NSE

  - Feature Selection 

    - Filter: Information Gain
    - Wrappers: 
    - Embedded: LASSO

  - Subspace Clustering

    - Local PCA-based method: 4C

    - Self-expressive Representation

      SSC: min$||Z||_1$, s.t. X=XZ, diag(Z)=0

      LRR: min$||Z||_*$, s.t. X=XZ

- Why we need Hashing?

- Finding similar items

  - K-shingles (k-gram) convert documents into sets

  - Min-Hashing convert input matrix into signature matrix 

    - Definition: the first row of the column contains "1"

    - Surprising properties: $Sim(C_1,C_2)=P(H(C_1)=H(C_2))=\frac{a}{a+b+c}$

    - How to compute signature matrix 

      trick:  using hashing function to implement permutation 

- Locality-Sensitive Hashing (LSH)

  Basic idea: divide the input matrix (signature matrix) into same bands, and hash them into different buckets.

- Learn to Hash

  - Radom Projection
    $$
    h(x)=
    \begin{cases}
    1, & r^Tx \geq 0 \\
    0, & else
    \end{cases}
    $$

  - Unsupervised Hash

    - PCA Hashing

      h(x) = sign(wx), w is determined by PCA

    - Spectral Hash

  - Supervised Hash

- Inverse Transform Sampling 

  $a=u(0,1),\;x=CDF^{-1}(a)$

- Rejection Sampling 

  Basic idea: draw samples from a simple proposed distribution, and the reject some samples to fit target distribution 

- Importance Sampling 

- MCMC Sampling 

  Basic idea: To construct a markov chain, where its equilibrium distribution converges to target distribution p(x)

  Markov properties: $Pr(X_{n+1}=x|X_1=x_1,X_2=x_2,\cdots,X_n=x_n)=Pr(X_{n+1}=x|X_n=x_n)$

  Detailed Balance Condition(细致平衡条件):  $P(X_i)\cdot Q(X_j|X_i)=P(X_j)\cdot Q(X_i|X_j)$

  Trick: It's difficult to find $Q(x|y)$. Thereby, change it as $P(X_i)Q(X_j|X_i)\alpha(X_j|X_i)=P(X_j)Q(X_i|X_j)\alpha(X_i|X_j)$

  But how to draw samples from $Q'(X_j|X_i)$ and $Q'(X_i|X_j)$ ?

  Solution: Rejection Sampling for $Q'$

- MH Sampling

  Basic idea: MAMC Sampling is not efficient, increasing acceptance ratio 

  $P(X_i)\cdot Q'(X_j|X_i)=P(X_j)\cdot Q'(X_i|X_j)$, both sides times a constant M 

  Magnify acceptance ratio by $\alpha(X_j|X_i)=min(1,\frac{p(X_j)q(X_i|X_j)}{p(X_i)q(X_j|X_i)})$

- Gibbs Sampling

  Basic idea: draw samples from conditional distribution to construct markov chain (100% acceptance ratio)

  $p(x_1,y_1)p(y_2|x_1)=p(x_1,y_2)p(y_1|x_1)$

  Trick: Sampling along with one direction

- Latent Dirichlet Allocation

  LDA is a tpoic modeling with probabilistic graph model

  However, $P(w|\theta,\phi,z,\alpha,\beta)$ is intractable 

  $\underset{z,\theta,\phi,\alpha,\beta}{max}\sum_{i=1}^nlog\,P(w_i|\theta,\phi,\alpha,\beta)$ 

  Solution: $P(z|w)$

  Draw samples from $P(z|w)$

  Using Gibbs Sampling, $P(z_i|z_{\lnot i},w)$

  Get $\theta, \phi$, get word-topic and document-topic

- Sampling on Data Stream

  Reservoir Sampling: draw samples with uniform distribution

- Data Stream Mining

  - Challenge of Data Stream MIning

  - Concept Drift

  - Concept Drifting Detection 

- next time



# Report

- Research Report / Research Survey 
- Format
  - Abstract 
  - Key words
  - Instruction
  - Methods
  - Experiment 
  - Conclusion 
- Deadline: Next Sunday of the end of this course 
- Topic: AI-field 
- Requirement: English & 3000 words