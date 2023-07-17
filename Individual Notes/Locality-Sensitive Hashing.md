## Locality-Sensitive Hashing for Documents

However, often we want only the most similar pairs or all pairs that are above some lower bound in similarity. If so, then we need to focus our attention only on pairs that are likely to be similar, without investigating every pair.

locality-sensitive hashing(LSH) is same as near-neighbor search.

### LSH for Minhash Signatures

One general approach to LSH is to “hash” items several times, in such a way that similar items are more likely to be hashed to the same bucket than dissimilar items are.  We then consider any pair that hashed to the same bucket for any of the hashings to be a candidate pair. We check only the candidate pairs for similarity.  The hope is that most of the dissimilar pairs will never hash to the same bucket, and therefore will never be checked. 

If we have minhash signatures for the items, an effffective way to choose the hashings is to divide the signature matrix into $b$ bands consisting of $r$ rows each. For each band, there is a hash function that takes vectors of $r$ integers (the portion of one column within that band) and hashes them to some large number of buckets. We can use the same hash function for all the bands, but we use a separate bucket array for each band, so columns with the same vector in difffferent bands will not hash to the same bucket.

###  Analysis of the Banding Technique

Suppose we use $b$ bands of $r$ rows each, and suppose that a particular pair of documents have Jaccard similarity $s$.

The probability the minhash signatures for these documents agree in any one particular row of the signature matrix is $s$.

We can calculate the probability that these documents (or rather their signatures) become a candidate pair as follows:

1. The probability that the signatures agree in all rows of one particular band is $s^r$.
2. The probability that the signatures disagree in at least one row of a particular band is $1 − s^r$.
3.  The probability that the signatures disagree in at least one row of each of the bands is $(1 − s^r)^b$.
4. The probability that the signatures agree in all the rows of at least one band, and therefore become a candidate pair, is $1 − (1 − s^r)^b$.

Notice that the threshold, the value of s at which the curve has risen halfway, is just slightly more than 0.5.

It must be emphasized that this approach can produce false negatives – pairs of similar documents that are not identified as such because they never become a candidate pair.



## Distance Measures

Suppose we have a set of points, called a $space$. A distance measure on this space is a function $d(x, y)$ that takes two points in the space as arguments and produces a real number, and satisfifies the following axioms:

1. $d(x,y)\geq0$  (no negative distances)
2. $d(x,y)=0$ iff $x=y$  (distances are positive, except for the distance from a point to itself)
3. $d(x,y)=d(y,x)$  (distance is symmetric)
4. $d(x,y)\leq d(x,z)+d(z,y)$  (the triangle inequality)

### Euclidean Distances

An n-dimensional Euclidean space is one where points are vectors of n real numbers.  The conventional distance measure in this space, which we shall refer to as the $L_2-norm$, is defined:
$$
d([x_1,x_2,\cdots,x_n],[y_1,y_2,\cdots,y_n])=\sqrt{\sum_{i=1}^n(x_i-y_i)^2}
$$
That is, we square the distance in each dimension, sum the squares, and take the positive square root.

### Jaccard Distance

Jaccard distance is:
$$
d(x,y)=1-Sim(x,y)=1-\frac{x\cap y}{x\cup y}
$$
That is, the Jaccard distance is 1 minus the ratio of the sizes of the intersection and union of sets x and y.

### Cosine Distance

The cosine distance makes sense in spaces that have dimensions.  In such a space, points may be thought of as directions.

The cosine distance between two points is the angle that the vectors to those points make. This angle will be in the range 0 to 180 degrees, regardless of how many dimensions the space has. 

We can calculate the cosine distance by first computing the cosine of the angle, and then applying the arc-cosine function to translate to an angle in the 0-180 degree range.

 Given two vectors $x$ and $y$,  the cosine of the angle between them is the dot product $x\cdot y$ divided by the $L_2-norms$ of $x$ and $y$.

The  dot product of vectors $[x_1,x_2,\cdots,x_n]\cdot[y_1,y_2,\cdots,y_n]$ is $\sum_{i=1}^nx_iy_i$

Cosine distance is:
$$
d(x,y)=\frac{\sum_{i=1}^nx_iy_i}{\sqrt{\sum_{i=1}^n(x_i-y_i)^2}}
$$

### Edit Distance

Edit distance makes sense when points are strings. The distance between two strings $x=x_1x_2\cdots x_n$ and $y=y_1y_2\cdots y_m$ is the smallest number of insertions and deletions of single characters that will convert $x$ to $y$.

Given two strings $x=abcde$ and $y=acfdeg$, $d(x,y)=3$

### Hamming Distance

Given a space of vectors, we define the Hamming distance between two vectors to be the number of components in which they differ.

Hamming distance cannot be negative, and if it is zero, then the vectors are identical.

The Hamming distance between the vectors 10101 and 11110 is 3. That is, these vectors diffffer in the second, fourth, and fififth components, while they agree in the fifirst and third components.



## The Theory of Locality-Sensitive Functions

The LSH technique of the minhash functions is one example of a family of functions that can be combined (by the banding technique) to distinguish strongly between pairs at a low distance from pairs at a high distance.

We shall explore other families of functions, besides the minhash functions, that can serve to produce candidate pairs efficiently. These functions can apply to the space of sets and the Jaccard distance, or to another space and/or another distance measure.

 There are three conditions that we need for a family of functions:

1. They must be more likely to make close pairs be candidate pairs than distant pairs.
2. They must be statistically independent, in the sense that it is possible to estimate the probability that two or more functions will all give a certain response by the product rule for independent events.
3. They must be effiffifficient, in two ways:
   1. They must be able to identify candidate pairs in time much less than the time it takes to look at all pairs. For example, minhash functions have this capability, since we can hash sets to minhash values in time proportional to the size of the data, rather than the square of the number of sets in the data. Since sets with common values are colocated in a bucket, we have implicitly produced the candidate pairs for a single minhash function in time much less than the number of pairs of sets.
   2. They must be combinable to build functions that are better at avoiding false positives and negatives, and the combined functions must also take time that is much less than the number of pairs.

