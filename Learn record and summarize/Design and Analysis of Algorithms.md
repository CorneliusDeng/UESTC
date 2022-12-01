# Question kinds

- Decision Problem (with yes-no answers)
- Optimal Value/Optimal Solution
- Numerical Calculation



# Programs and algorithms

A computer program is an instance, or concrete representation, for an algorithm in some programming language.

- A program is to be read by computer
- An algorithm is to be read by human being
- Algorithms can be expressed by pseudocode just some short steps that are easy to read and understand



# Different Classes of Problems

- P: a solution can be solved in polynomial time.（多项式时间）

- NP: a solution can be checked in polynomial time.

- NPC: problems that may not have polynomial-time algorithms.

  

# The Stable Matching Problem

- **Goal:**  Given n men and n women, find a "suitable" matching. Participants rate members of opposite sex. Each man lists women in order of preference from best to worst. Each woman lists men in order of preference from best to worst

- **Perfect matching:** everyone is matched monogamously. Each man gets exactly one woman. Each woman gets exactly one man.

- **Stability:** no incentive for some pair of participants to undermine assignment by joint action. In matching M, an unmatched pair m-w is unstable if man m and woman w prefer each other to current partners. Unstable pair m-w could each improve by eloping.

- **Stable matching:** perfect matching with no unstable pairs.

- **Stable matching problem:**  Given the preference lists of n men and n women, find a stable matching if one exists.

- **Propose-and-reject algorithm.** [Gale-Shapley 1962] Intuitive method that guarantees to find a stable matching.

  ```c
  Initialize each person to be free.
  while (some man is free and hasn't proposed to every woman) {
  				Choose such a man m
   				w = 1st woman on m's list to whom m has not yet proposed
  				  if (w is free)
  							assign m and w to be engaged
  				  else if (w prefers m to her fiancé m')
  							assign m and w to be engaged, and m' to be free
  				 else
  						   w rejects m
  }
  ```
  
  

# Asymptotic Order of Growth渐进分析

用渐进表达式表达数量级的差别，渐进表式的核心内容就是忽略常数，是一个被大家认同的计算机里表式运行时间和空间的方法。

- **渐近上界记号O，理解相当于≤**
  - O(g(n)) = { f(n) | 存在正常数c和n0使得对所有n≥n0有：0≤f(n)≤cg(n) }
  - 读“O”
- **渐近下界记号Ω，理解相当于≥**
  - Ω(g(n)) = { f(n) | 存在正常数c和n0使得对所有n≥n0有：0≤cg(n)≤f(n) }
  - 读”omega”
- **渐近近界记号Θ，理解相当于=**
  - Θ(g(n)) = { f(n) | 存在正常数c1,c2和n0使得对所有n≥n0有：c1g(n)≤f(n)≤c2g(n) }
  - 读”theta”
- **非紧上界记号o，理解相当于<**
  - o(g(n)) = { f(n) | 对于任何正常数c>0，存在正数和n0 >0使得对所有n≥n0有：0≤f(n)<cg(n) }，等价于 f(n) / g(n) →0 ，as n→∞。
- **非紧下界记号ω，理解相当于>**
  - ω(g(n)) = { f(n) | 对于任何正常数c>0，存在正数和n0 >0使得对所有n≥n0有：0≤cg(n)<f(n) }，等价于 f(n) / g(n) →∞，as n→∞。 f(n) ∈ ω(g(n))↔️g(n) ∈ o (f(n))

定理 如果t1(n) ∈O(g1(n))并且t2(n) ∈O(g2(n))，则t1(n)+ t2(n) ∈O(max{(g1(n), g2(n)})
对于符号Ω和Θ，该定理也成立
该定理表明：当算法由两个连续执行部分组成时，该算法的整体效率由具有较大增长次数的那部分所决定。

规则O(f(n))+O(g(n)) = O(max{f(n),g(n)}) 的证明：

​	对于任意f1(n) ∈ O(f(n)) ，存在正常数c1和自然数n1，使得对所有n ≥ n1，有f1(n) ≤ c1f(n) 。

​	类似地，对于任意g1(n) ∈ O(g(n)) ，存在正常数c2和自然数n2，使得对所有n ≥ n2，有g1(n) ≤ c2g(n) 。

​	令c3=max{c1,c2}， n3 =max{n1,n2}，h(n)= max{f(n),g(n)} ，则对所有的 n ≥ n3，有：

​	f1(n) +g1(n) ≤ c1f(n) + c2g(n) ≤ c3f(n) + c3g(n)= c3(f(n) + g(n)) ≤ c3 · 2max{f(n),g(n)} = c3 · 2h(n) = O(max{f(n),g(n)}) .

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/equation.png)

- **渐近增长率比较的三种方法**

  - 定义法

    - 找到正常数c和n0使得对所有n ≥ n0有f(n) ≤ cg(n)，则  f(n) = O(g(n))

  - 极限法

    - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/limitation.png)

  - 取对数法

    - 对于比较难的比较的两个函数，我们可以对它们同时取对数后再进行比较

      

# Three Techniques for Designing Algorithms

## **Greedy Algorithms贪心算法**

- **Basic Idea:** Build up a solution incrementally, myopically optimizing some local criterion.

  In each iteration, we choose the “best” solution at that moment. This “best” solution may not yield the BEST final solution

- **Local optimal v.s. Global optimal:** Greedy algorithm is to make the locally optimal choice at each moment. In some problems, such strategy can lead to a global optimal solution

- Advantage: Simple, efficient

- Disadvantage: May be incorrect / may not be optimal

- Greedy Analysis Strategies

  - Greedy algorithm stays ahead. Show that after each step of the greedy algorithm, its solution is at least as good as any other algorithm's. 
  - Structural. Discover a simple "structural" bound asserting that every possible solution must have a certain value. Then show that your algorithm always achieves this bound.
  - Exchange argument. Gradually transform any solution to the one found by the greedy algorithm without hurting its quality.

- **Interval scheduling区间调度**

  - Job j starts at sj and finishes at fj.

    Two jobs compatible if they don't overlap.

    Goal: find maximum subset of mutually compatible jobs

  - Greedy template. Consider jobs in some order. Take each job provided it's compatible with the ones already taken
    - [Earliest start time] Consider jobs in ascending order of start time sj.
    - [Earliest finish time] Consider jobs in ascending order of finish time fj
    - [Shortest interval] Consider jobs in ascending order of interval length fj - s
    - [Fewest conflicts] For each job, count the number of conflicting jobs cj . Schedule in ascending order of conflicts cj

## **Divide and Conquer分治法**

- **Basic Idea:** Break up a problem into some sub-problems, solve each sub-problem independently, and combine solution to sub-problems to form solution to original problem. 
- **Most common usage：**Break up problem of size n into two equal parts of size ½n. Solve two parts recursively. Combine two solutions into overall solution in linear time.
- **Multiply乘法**
  - Given two n-digit integers a and b, compute a ×b.

  - Brute force solution: O(n^2) bit operations.
  - To multiply two n-digit integers: Multiply four ½n-digit integers. Add two ½n-digit integers, and shift to obtain result.
  - Karatsuba Multiplication: Add two ½ n digit integers. Multiply three ½ n-digit integers. Add, subtract, and shift ½ n-digit integers to obtain result.
  - Fast matrix multiplication
    - Divide: partition A and B into ½ n-by-½ n blocks.
    - Compute: 14 ½ n-by-½ n matrices via 10 matrix additions.
    - Conquer: multiply 7 ½ n-by-½ n matrices recursively.
    - Combine: 7 products into 4 terms using 8 matrix additions.
- **Master theorem 主定理**
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/master%20theorem%201.png)
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/master%20theorem%202.png)
  - 理解主定理
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/master%20theorem%203.png)
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/master%20theorem%204.png)


## **Dynamical Programming动态规划**

- **Basic Idea:** Break up a problem into a series of overlapping sub-problems, and build up solutions to larger and larger sub-problems.

- The most important step of designing dynamic programming algorithms:To find a good way to separate the problem into many overlapping subproblems.

- Overlapping Subproblems:

  - When a recursive algorithm re-visits the same problem over and over again, we say that the problem has overlapping subproblems.
  - An idea to save the running time is to avoid computing the same subproblem twice.
  - This idea is the essential of dynamic programming.

- **The steps of solving problem**

  - separate the problem into many overlapping subproblems.
  - construct a recurrence relation
  - use memoization or bottom-up methods to avoid double counting of the same subproblem

- **Weighted Interval Scheduling带权值的区间调度**

  - Job j starts at sj , finishes at fj , and has weight or value vj . 

    Two jobs compatible if they don't overlap.

    Goal: find maximum weight subset of mutually compatible jobs

  - Notation. Label jobs by finishing time: f1 ≤ f2 ≤ . . . ≤ fn .
    Def. p(j) = largest index i < j such that job i is compatible with j.

    OPT(j) = value of optimal solution to the problem consisting of job requests 1, 2, ..., j

    - Case 1: OPT selects job j.

      – can't use incompatible jobs { p(j) + 1, p(j) + 2, ..., j - 1 }

      – must include optimal solution to problem consisting of remaining compatible jobs 1, 2, ..., p(j)

    - Case 2: OPT does not select job j.

      – must include optimal solution to problem consisting of remaining compatible jobs 1, 2, ..., j-1

    - *OPT*( *j*) = 

      - 0，if j = 0
      - max(vj+OPT(p(j), OPT(j-1))，otherwise

  - Weighted Interval Scheduling: Memoization

    Memoization. Store results of each sub-problem in a cache; lookup as needed.

    - code

      ```c
      Input: n,s1,…,sn,f1,…,fn ,v1,…,vn
      Sort jobs by finish times so that f1 ≤ f2 ≤...≤ fn
      Compute p(1),p(2),……,p(n)
        

    for j = 1 to n
        M[j] = empty
    M[0]=0
      M-Compute-Opt(n) {
      if (M[n] is empty)
          M[n] = max(vn + M-Compute-Opt(p(n)), M-Compute-Opt(n-1))
    	return M[n]
      }
    ```

  - Weighted Interval Scheduling: Bottom-Up

    Bottom-up dynamic programming. Unwind recursion.

    - code

      ```c
      Input: n, s1,…,sn , f1,…,fn , v1,…,vn
      Sort jobs by finish times so that f1 ≤ f2 ≤ ... ≤ fn.
      Compute p(1), p(2), …, p(n)
      Iterative-Compute-Opt {
      	M[0] = 0
      	for j = 1 to n
      		M[j] = max(vj + M[p(j)], M[j-1])
      }
      ```

- **Knapsack Problem背包问题**

  - Given n objects and a "knapsack."

    Item i weighs wi > 0 kilograms and has value vi > 0.

    Knapsack has capacity of W kilograms.

    Goal: fill knapsack so as to maximize total value

  - Greedy: repeatedly add item with maximum ratio vi / wi.

    But, greedy is not optimal.

  - Def. OPT(i, w) = max profit subset of items 1, …, i with weight limit w.

    - Case 1: OPT does not select item i.

      – OPT selects best of { 1, 2, …, i-1 } using weight limit w 

    - Case 2: OPT selects item i.

      – new weight limit = w – wi 

      – OPT selects best of { 1, 2, …, i–1 } using this new weight limit

  - *OPT*(*i*, *w*) =

    - 0，if i = 0
    - OPT(i −1, w)，if wi > w
    - max(OPT(i −1, w), vi + OPT(i −1, w− wi))，otherwise

  - Code, Running time:O(n·W)，Not polynomial in input size

    ```c
    Input: n, w1, ……, wn, v1, ……, vn
    for w = 0 to w
    M[0,w] = 0
    for i = 1 to n
    for w = 1 to w
        if (wi > w)
        M[i,w] = M[i-1,w]
        else
        M[i,w] = max(M[i-1,w], vi + M[i-1,w-wi])
    return M[n,w]
  ```

- **Sequence Alignment序列对齐**

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Sequence%20Alignment%201.png)

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Sequence%20Alignment%202.png)

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Sequence%20Alignment%20%203.png)

  ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Sequence%20Alignment%204.png)

# 网络流Flow Network

## **Definition 定义**

Abstraction for material flowing through the edges.
G = (V, E) = directed graph, no parallel edges.
Two distinguished nodes: s = source, t = sink.
c(e) = capacity of edge e.

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Flow%20Network.png)

Def. An s-t cut is a partition (A, B) of V with s ∈ A and t ∈ B.
Def. The capacity of a cut (A, B) is:
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Cuts.png)

**Minimum Cut Problem 最小割问题：Find an s-t cut of minimum capacity.**

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Flows.png)

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Flow%20value%20lemma.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Flow%20value%20lemma%20Pf.png)

**Max flow problem 最大流问题： Find s-t flow of maximum value.**

**Weak duality 弱对偶性：** Let f be any flow, and let (A, B) be any s-t cut. Then the value of the flow is at most the capacity of the cut.
eg.Cut capacity = 30 ⇒ Flow value ≤ 30
Let f be any flow. Then, for any s-t cut (A, B) we have：v(f) ≤ cap(A, B).
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Weak%20Duality%20Pf.png)
**Corollary 推论：If v(f) = cap(A, B), then f is a max flow and (A, B) is a min cut**

## Residual Graph 残存图

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Residual%20Graph.png)
解释：
流网络的一条边可以允许的额外流量等于该边的容量减去该边的流量
一条边所能允许的反向流量最多将其正向流量抵消，残存网络中的这些反向边允许算法将已经发送出来的流量发送回去，其意义是缩减该边的流量

## Augmenting Path 增广路径

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Augmenting%20Path.png)
增广路径是残存网络中一条从源点s到汇点t的简单路径。

**Choosing Good Augmenting Paths：**
Use care when selecting augmenting paths：Some choices lead to exponential algorithms. Clever choices lead to polynomial algorithms. If capacities are irrational, algorithm not guaranteed to terminate!
Goal: choose augmenting paths so that Can find augmenting paths efficiently, Few iterations.
Choose augmenting paths with: Max bottleneck capacity. Sufficiently large bottleneck capacity. Fewest number of edges.

## Ford-Fulkerson

**Ford-Fulkerson方法用来解决最大流问题，其核心是沿着增广路径重复增加路径上的流量，直到找到一个最大流为止。**

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Ford-Fulkerson.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Ford-Fulkerson%20Example.jpg)
(a)~(e)图的左边部分描述的是残存图，覆盖阴影的路径是增广路径，右图描述的是当前流网络

**Running Time：**
**Assumption.** All capacities are integers between 1 and C.
**Invariant.** Every flow value f(e) and every residual capacity cf(e) remains an integer throughout the algorithm.
**Theorem.** The algorithm terminates in at most v(f*) ≤ mC iterations. **Pf.** Each augmentation increase value by at least 1.
**Corollary.** If C = 1, Ford-Fulkerson runs in O(m^2) time.
**Integrality theorem.** If all capacities are integers, then there exists a max flow f for which every flow value f(e) is an integer.
**Pf.** Since algorithm terminates, theorem follows from invariant.

## Max-Flow Min-Cut Theorem 最大流最小割定理

**Augmenting path theorem：** Flow f is a max flow iff there are no augmenting paths.

**Max-flow min-cut theorem：** [Elias-Feinstein-Shannon 1956, Ford-Fulkerson 1956] **The value of the max flow is equal to the value of the min cut.**
Pf. We prove both simultaneously by showing:
(i) There exists a cut (A, B) such that v(f) = cap(A, B).
(ii) Flow f is a max flow.
(iii) There is no augmenting path relative to f

(i) ⇒ (ii) This was the corollary to weak duality lemma.
(ii) ⇒ (iii) We show contrapositive.Let f be a flow. If there exists an augmenting path, then we can improve f by sending flow along path.
(iii) ⇒ (i)Let f be a flow with no augmenting paths. Let A be set of vertices reachable from s in residual graph. By definition of A, s ∈ A. By definition of t, t ∉ A



# NP and Computational Intractability

## Polynomial-Time Reductions 多项式时间归约

- Classify Problems: Classify problems according to those that can be solved in polynomial-time and those that cannot.

- Reduction: Problem **X** polynomial **reduces to** problem **Y** if arbitrary instances of problem X can be solved using

  - Polynomial number of standard computational steps, plus

  - Polynomial number of calls to oracle that solves problem Y. (Oracle means computational model supplemented by special piece of hardware that solves instances of Y in a single step)

  - Notation
    $$
    X ≤_P Y
    $$
    ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Polynomial-Time%20Reductions.png)

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Polynomial-Time%20Reduction.png)

## Independent Set and Vertex Cover 独立集和点覆盖

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Independent%20Set.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Vertex%20Cover.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Vertex%20Cover%20and%20Independent%20Set.png)

## Vertex Cover Reduces to Set Cover 点覆盖归约于集合覆盖

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Set%20Cover.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Vertex%20Cover%20Reduces%20to%20Set%20Cover.png)

Lemma. G = (V, E) contains a vertex cover of size k iff (U, S, k) contains a set cover of size k.

## SAT & 3-SAT 

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/SAT.png)

**3 Satisfiability Reduces to Independent Set**
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/3%20Satisfiability%20Reduces%20to%20Independent%20Set.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/3%20sat%20proof.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Review.png)

## Self-Reducibility

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Self-Reducibility.png)

## Definition of NP

- Decision problem.
  - X is a set of strings.
  - Instance: string s.
  - Algorithm A solves problem X: A(s) = yes iff s ∈ X.
- Polynomial time. 
  - Algorithm A runs in poly-time if for every string s, A(s) terminates in at most p(|s|) "steps", where p() is some polynomial. The |s| means length of s.
- Certification algorithm intuition.
  - Certifier views things from "managerial" viewpoint.
  - Certifier doesn't determine whether s ∈ X on its own; Rather, it checks a proposed proof t that s ∈ X
- ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/NP.png)
  NP stands for nondeterministic polynomial-time.
- Example
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/NP%20composite.png)
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/NP%20SAT.png)
  - ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/NP%20HAMILTON-CYCLE.png)

- ![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/P-NP-EXP.png)

## NP-Completeness

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Def%201.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Def%202.png)

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Complexity%201.png)

More Hard Computational Problems
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Complexity%202.png)

![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Complexity%203.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Complexity%204.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Complexity%205.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Complexity%206.png)
![](https://raw.githubusercontent.com/CorneliusDeng/Markdown-Photos/main/Design%20and%20Analysis%20of%20Algorithms/Complexity%207.png)

# 近似算法 Approximation Algorithms

Q. Suppose I need to solve an NP-hard problem. What should I do?

A. Theory says you're unlikely to find a poly-time algorithm.

- Must sacrifice one of three desired features.
  - Solve problem to optimality.
  - Solve problem in poly-time.
  - Solve arbitrary instances of the problem.
- ρ-approximation algorithm.
  - Guaranteed to run in poly-time.
  - Guaranteed to solve arbitrary instance of the problem
  - Guaranteed to find solution within ratio ρ of true optimum.
- Challenge. Need to prove a solution's value is close to optimum, without even knowing what optimum value is!