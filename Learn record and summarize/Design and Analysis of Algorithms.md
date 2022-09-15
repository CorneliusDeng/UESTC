## Question kinds

- Decision Problem (with yes-no answers)
- Optimal Value/Optimal Solution
- Numerical Calculation



## Programs and algorithms

A computer program is an instance, or concrete representation, for an algorithm in some programming language.

- A program is to be read by computer
- An algorithm is to be read by human being
- Algorithms can be expressed by pseudocode just some short steps that are easy to read and understand



## Different Classes of Problems

- P: a solution can be solved in polynomial time.（多项式时间）

- NP: a solution can be checked in polynomial time.

- NPC: problems that may not have polynomial-time algorithms.

  

## The Stable Matching Problem

- **Goal:**  Given n men and n women, find a "suitable" matching. Participants rate members of opposite sex. Each man lists women in order of preference from best to worst. Each woman lists men in order of preference from best to worst

- **Perfect matching:** everyone is matched monogamously. Each man gets exactly one woman. Each woman gets exactly one man.

- **Stability:** no incentive for some pair of participants to undermine assignment by joint action. In matching M, an unmatched pair m-w is unstable if man m and woman w prefer each other to current partners. Unstable pair m-w could each improve by eloping.

- **Stable matching:** perfect matching with no unstable pairs.

- **Stable matching problem:**  Given the preference lists of n men and n women, find a stable matching if one exists.

- **Propose-and-reject algorithm.** [Gale-Shapley 1962] Intuitive method that guarantees to find a stable matching.

  `Initialize each person to be free.`
  `while (some man is free and hasn't proposed to every woman) {`
  				` Choose such a man m`
   				`w = 1st woman on m's list to whom m has not yet proposed`
  				  `if (w is free)`
  							`assign m and w to be engaged`
  				  `else if (w prefers m to her fiancé m')`
  							`assign m and w to be engaged, and m' to be free`
  				 `else`
  						   `w rejects m`
  `}`



## Asymptotic Order of Growth渐进分析

用渐进表达式表达数量级的差别，渐进表式的核心内容就是忽略常数，是一个被大家认同的计算机里表式运行时间和空间的方法。

- **渐进上界记号O，理解相当于≤**
  - O(g(n)) = { f(n) | 存在正常数c和n0使得对所有n≥n0有：0≤f(n)≤cg(n) }
- **渐近下界记号Ω，理解相当于≥**
  - Ω(g(n)) = { f(n) | 存在正常数c和n0使得对所有n≥n0有：0≤cg(n)≤f(n) }
- **紧渐近界记号Θ，理解相当于=**
  - Θ(g(n)) = { f(n) | 存在正常数c1,c2和n0使得对所有n≥n0有：c1g(n)≤f(n)≤c2g(n) }
- **非紧上界记号o，理解相当于<**
  - o(g(n)) = { f(n) | 对于任何正常数c>0，存在正数和n0 >0使得对所有n≥n0有：0≤f(n)<cg(n) }，等价于 f(n) / g(n) →0 ，as n→∞。
- **非紧下界记号ω，理解相当于>**
  - ω(g(n)) = { f(n) | 对于任何正常数c>0，存在正数和n0 >0使得对所有n≥n0有：0≤cg(n)<f(n) }，等价于 f(n) / g(n) →∞，as n→∞。 f(n) ∈ ω(g(n))↔️g(n) ∈ o (f(n))

规则O(f(n))+O(g(n)) = O(max{f(n),g(n)}) 的证明：

​	对于任意f1(n) ∈ O(f(n)) ，存在正常数c1和自然数n1，使得对所有n ≥ n1，有f1(n) ≤ c1f(n) 。

​	类似地，对于任意g1(n) ∈ O(g(n)) ，存在正常数c2和自然数n2，使得对所有n ≥ n2，有g1(n) ≤ c2g(n) 。

​	令c3=max{c1,c2}， n3 =max{n1,n2}，h(n)= max{f(n),g(n)} ，则对所有的 n ≥ n3，有：

​	f1(n) +g1(n) ≤ c1f(n) + c2g(n) ≤ c3f(n) + c3g(n)= c3(f(n) + g(n)) ≤ c3 · 2max{f(n),g(n)} = c3 · 2h(n) = O(max{f(n),g(n)}) .



## Three Techniques for Designing Algorithms

- **Greedy Algorithms贪心算法**
- **Divide and Conquer**
- **Dynamical Programming动态规划**