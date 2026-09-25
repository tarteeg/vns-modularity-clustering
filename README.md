# VNS for Modularity-Based Graph Clustering

An implementation in Julia of label-propagation heuristics and the Variable Neighborhood Search (VNS) metaheuristic for graph clustering (community detection) by **modularity maximization**.

The project studies how an undirected graph can be partitioned into communities whose internal connections are dense and whose external connections are sparse. It rebuilds, step by step, the pipeline used by Aloise et al. **[1]** in the 10th DIMACS Implementation Challenge:

```
LPA  ──►  LPAm  ──►  LPAm+ (greedy or MSG merging)  ──►  VNS / VNDS
fast       modularity-    escapes LPAm local maxima       escapes LPAm+ local maxima
baseline   aware moves    by merging communities          by random perturbations
```

## Academic context

This project is an educational replication of a scientific article proposed by Professor Daniel Aloise as supplementary reading for the course **INF8111 — Data Mining** at **Polytechnique Montréal**, during the intensive summer session of 2026.

The purpose of this repository is to reproduce and better understand the algorithmic ideas of the reference article: modularity-based community detection, local-search heuristics, community merging strategies, neighborhood perturbations, stochastic optimization, and experimental comparison on benchmark graphs.

This repository is an independent educational implementation. It is not the official implementation of the original article and is not affiliated with Polytechnique Montréal or Professor Daniel Aloise.

The original authors implemented their algorithm in C++. Julia was chosen here because it offers execution speed in the same league as compiled languages while keeping the readability and development speed of Python. Moreover, my objective was to understand the mathematical concepts and algorithms behind the results of the main paper, which is why I chose the language I was most comfortable with.

## Project status

| Component | Status |
|---|---|
| Data acquisition (9 benchmark graphs) | ✅ Done |
| Newman–Girvan modularity (weighted and unweighted) | ✅ Done |
| LPA (Raghavan et al. / Barber & Clark) | ✅ Done, benchmarked |
| LPAm (Barber & Clark, label-queue variant of Aloise et al.) | ✅ Done, benchmarked |
| LPAm+ with classical greedy merging | ✅ Done, benchmarked on the 7 smaller graphs |
| LPAm+ with multistep greedy merging (MSG) | ✅ Done, benchmarked on all 9 graphs |
| VNS perturbations: SINGLETON, DIVISION | ✅ Implemented |
| VNS perturbations: NEIGHBOR, EDGE, FUSION, REDISTRIBUTION | 🚧 In progress |
| VNDS main loop (decomposition + shaking + LPAm+) | 🚧 In progress |
| Tuning of the MSG level parameter `l` per dataset | 🚧 In progress |

---

## Mathematical background

### Notation

Let $G = (V, E, \omega)$ be an undirected graph with $n = |V|$ vertices and $m = |E|$ edges. $A$ is the adjacency matrix ($A_{uv} = \omega(\{u,v\})$, or $1$ for unweighted graphs), $k_u = \sum_v A_{uv}$ is the degree (strength) of vertex $u$, and $\sum_u k_u = 2m$.

A clustering is a label vector $\ell = (\ell_1, \dots, \ell_n)$: vertex $u$ belongs to community $\ell_u$. For a community (label) $c$ we write

- $I_c$ — the weight of edges with both ends in $c$,
- $K_c = \sum_{u} k_u \, \delta(\ell_u, c)$ — the total degree (the "volume") of $c$,
- $I_{ab}$ — the weight of edges between communities $a$ and $b$,
- $N_{v\ell} = \sum_{u \neq v} A_{uv}\,\delta(\ell_u, \ell)$ — the number (weight) of neighbors of $v$ carrying label $\ell$.

### 1. Modularity (Newman & Girvan)

Modularity compares the fraction of edges falling inside communities with the fraction expected in a random graph that has the same degree sequence (the *configuration null model*, in which an edge between $u$ and $v$ appears with probability $P_{uv} = k_u k_v / 2m$):

$$
Q = \frac{1}{2m} \sum_{u,v} \left( A_{uv} - \frac{k_u k_v}{2m} \right) \delta(\ell_u, \ell_v).
$$

Grouping the terms by community gives the form used in the code (`newman_modularity`) and in equation (2) of **[1]**:

$$
Q = \sum_{c} \left[ \frac{I_c}{m} - \left( \frac{K_c}{2m} \right)^2 \right].
$$

The first term rewards internal edges; the second term is the internal fraction a random graph would produce anyway. $Q$ close to $0$ means no structure beyond chance, while values approaching $1$ indicate a strong community structure. Maximizing $Q$ gives both the partition and the number of communities, but the problem is NP-hard, which is why heuristics are needed. The trivial one-community partition always has $Q = 0$ (this is the sanity check in the notebook).

With $B_{uv} = A_{uv} - k_u k_v / 2m$ (the *modularity matrix*), $Q = \frac{1}{2m}\sum_{u,v} B_{uv}\,\delta(\ell_u, \ell_v)$. Every algorithm below is a different strategy for climbing this function.

### 2. Label Propagation Algorithm (LPA)

**Procedure (Raghavan et al., reformulated by Barber & Clark [4]).** Every vertex starts with a unique label. Vertices are visited asynchronously in random order, and each vertex adopts the label that is most frequent among its neighbors:

$$
\ell_v \leftarrow \arg\max_{\ell} \sum_{u} A_{uv}\, \delta(\ell_u, \ell).
$$

Ties are broken by keeping the current label when it is among the best, otherwise by picking uniformly at random among the best labels. The algorithm stops when no vertex changes label.

**Objective function.** Barber & Clark show that this rule is a local ascent on

$$
H = \frac{1}{2} \sum_{u,v} A_{uv}\, \delta(\ell_u, \ell_v),
$$

i.e. the number of edges whose endpoints share a label. Isolating vertex $v$ in $H$, the only term that depends on $\ell_v$ is $\sum_u A_{uv}\,\delta(\ell_u, \ell_v)$, which is exactly what the update rule maximizes, so $H$ never decreases.

**Drawback.** The global maximum of $H$ is the useless partition where every vertex has the same label ($H = m$). LPA only produces meaningful communities because it gets stuck in *local* maxima of $H$. This explains its instability: results depend strongly on the visiting order, and the algorithm can collapse into one or two giant communities (visible in the notebook on Jazz and C. elegans averages).

**Cost.** One sweep over the vertices costs $O(m)$.

### 3. Modularity-specialized Label Propagation (LPAm)

**Constrained objective.** Barber & Clark add a penalty term that discourages communities with a large total degree:

$$
H' = H - \lambda G_2, \qquad G_2 = \frac{1}{2} \sum_{\ell} K_\ell^2 = \frac{1}{2}\sum_{u,v} k_u k_v\, \delta(\ell_u, \ell_v).
$$

Therefore

$$
H' = \frac{1}{2} \sum_{u,v} \left( A_{uv} - \lambda k_u k_v \right) \delta(\ell_u, \ell_v),
$$

and choosing $\lambda = 1/2m$ gives exactly $H' = m\,Q$. Label propagation with the symmetric matrix $B$ in place of $A$ is thus a local ascent on modularity.

**Update rule.** Removing $v$ from its community and computing the gain of inserting it into community $\ell$ gives

$$
\ell_v \leftarrow \arg\max_{\ell} \left( N_{v\ell} - \frac{k_v\, K_\ell^{\setminus v}}{2m} \right),
$$

where $K_\ell^{\setminus v}$ is the volume of $\ell$ without $v$. This is the score computed by `choose_best_label!`, which first subtracts $k_v$ from the volume of the current label.

**Candidate labels.** A label carried by none of the neighbors has $N_{v\ell} = 0$ and a non-positive score, so only the neighbors' labels plus one *unused* label (score $0$, i.e. making $v$ a singleton) need to be evaluated. One sweep therefore still costs $O(m)$, and each accepted move strictly increases $Q$.

**Label-queue speed-up (Aloise et al. [1, §2.2]).** A vertex whose neighboring communities did not change since its last evaluation cannot improve. The implementation therefore iterates over *labels* instead of vertices: a queue initially holds every label; a label is popped and all its vertices are re-evaluated; when a vertex moves from $\ell^{old}$ to $\ell^{new}$, both labels and the labels of the moved vertex's neighbors are pushed back. The algorithm stops when the queue is empty.

**Drawback.** The $G_2$ penalty pushes LPAm toward partitions where communities have similar total degrees. LPAm therefore stalls in poor local maxima: two communities that should be one are kept apart because no *single* vertex can move profitably on its own (the "stubborn nodes" of Liu & Murata [2]).

### 4. LPAm+ : LPAm with community merging

Liu & Murata **[2]** escape these local maxima by moving whole communities at once. Moving an entire community $b$ into community $a$ is the same as merging them, and the modularity change of that merge is

$$
\Delta Q_{ab} = \frac{I_{ab}}{m} - \frac{K_a K_b}{2m^2}.
$$

Only adjacent communities ($I_{ab} > 0$) can have $\Delta Q_{ab} > 0$. LPAm+ alternates two phases until neither improves $Q$:

```
labels ← LPAm(initial labels)
while some adjacent pair has ΔQ > 0:
    merge several positive pairs (merging phase)
    labels ← LPAm(labels)                  (refinement phase)
```

Each phase increases $Q$ monotonically. Two merging strategies are implemented.

**4a. Classical greedy merging (Newman [5]).** At each round, merge the single pair with the largest $\Delta Q$, then recompute. This is simple but slow (one merge per round) and biased: large communities tend to remain the best partner and absorb their neighbors one after the other. The notebook recomputes $Q$ from scratch for each candidate pair, so this variant is only run on the seven smaller graphs.

**4b. Multistep greedy merging, MSG (Schuetz & Caflisch [3]).** All values $\Delta Q_{ab} > 0$ are computed once and sorted. Their distinct values define *levels*; any pair whose $\Delta Q$ lies within the top $l$ levels is eligible, and eligible pairs are merged in decreasing order of $\Delta Q$ as long as neither community has already been merged during the current round. After merging $a$ and $b$ into $a$, the gains toward every other community $c$ are updated locally, without scanning the graph:

$$
\Delta Q'_{ac} =
\begin{cases}
\Delta Q_{ac} + \Delta Q_{bc} & \text{if } c \text{ is adjacent to both } a \text{ and } b,\\[4pt]
\Delta Q_{ac} - \dfrac{K_b K_c}{2m^2} & \text{if } c \text{ is adjacent to } a \text{ only},\\[8pt]
\Delta Q_{bc} - \dfrac{K_a K_c}{2m^2} & \text{if } c \text{ is adjacent to } b \text{ only}.
\end{cases}
$$

These formulas follow from $I_{(a\cup b)c} = I_{ac} + I_{bc}$ and $K_{a\cup b} = K_a + K_b$, and they are what `msg_merge_pair!` implements. With $l = 1$ MSG reduces to the classical greedy algorithm; larger $l$ merges many pairs per round, which avoids the "snowball" bias and makes the method scale. A merging round costs $O(m \log n)$, and Liu & Murata estimate the total cost of LPAm+ at $O(m \log^2 n)$ on hierarchical networks. The notebook currently uses $l = 10$ on every dataset; Schuetz & Caflisch recommend letting $l$ grow with the size of the network.

### 5. Variable Neighborhood Search (VNS) and its decomposition variant (VNDS)

**Local maxima and neighborhoods.** For a neighborhood structure $\mathcal{N}$, a solution $x_L$ is a local maximum if $f(x_L) \ge f(x)$ for every $x \in \mathcal{N}(x_L)$. LPAm+ returns a local maximum of $Q$ with respect to single-vertex moves and pairwise merges, but not necessarily with respect to other kinds of moves. VNS relies on three observations **[1, §2.1]**: a local maximum for one neighborhood is not necessarily one for another; a global maximum is a local maximum for every neighborhood; and local maxima of different neighborhoods are often close to each other.

**Principle.** VNS keeps an incumbent $x$. It draws a random solution $x'$ from a neighborhood of $x$ (*shaking*), applies local search to reach $x''$, and accepts $x''$ only if $f(x'') > f(x)$, in which case the search restarts from the smallest neighborhood. Otherwise it tries a larger neighborhood. Nearby neighborhoods are explored more often than distant ones.

**Perturbation moves [1, §2.3].** A cluster chosen inside the current subproblem is perturbed with one of the following moves:

| Move | Effect | Probability in [1] |
|---|---|---|
| SINGLETON | every vertex of the cluster becomes a singleton cluster | 30 % |
| DIVISION | the cluster is split at random into two halves of equal size | 30 % |
| NEIGHBOR | each vertex is relabeled with a neighbor's label or an unused label | 28 % |
| EDGE | two linked vertices in different clusters are moved together into a randomly chosen neighboring cluster | 5 % |
| FUSION | two or more clusters are merged into one | 4 % |
| REDISTRIBUTION | the cluster is destroyed and each of its vertices joins a random neighboring cluster | 3 % |

(The article announces five neighborhoods in the text but lists six and gives a probability for each of the six; this project implements all six.)

**Decomposition (VNDS) [1, §2.4].** To handle graphs with millions of vertices, shaking and local search are restricted to a small subproblem $S$ made of one random cluster and $s-1$ neighboring clusters:

```
x ← LPAm+(random solution, whole graph P)
s ← 1
while stopping condition not met:
    S  ← random cluster of x + (s − 1) neighboring clusters
    α  ← random move drawn with the probabilities above
    x' ← shaking(x, α, S)
    x' ← LPAm+(x', S)                  # local search on the subproblem only
    if Q(x') > Q(x):
        x ← LPAm(x', P)                # fast global refinement, no merging
        s ← 1
    else:
        s ← s + 1
        if s > min(MAX_SIZE, #clusters(x)): s ← 1
return x
```

The article uses `MAX_SIZE = 15`. The stopping condition is either a number of non-improving iterations or a CPU time limit. With this method the authors found the proven optimum on every exactly solved instance they tested, and obtained the second prize in the modularity Quality challenge of DIMACS 10.

---

## Datasets

The nine benchmark networks come from the 10th DIMACS Implementation Challenge collection and are loaded through `Graphs.jl` and `MatrixDepot.jl` (SuiteSparse collection). Self-loops are removed and every graph is treated as undirected and unweighted.

| Dataset | Nodes | Edges | Description |
|---|---:|---:|---|
| Karate | 34 | 78 | Zachary's karate club friendships |
| Dolphins | 62 | 159 | Associations between dolphins of Doubtful Sound |
| Political Books | 105 | 441 | Co-purchased books about US politics |
| College Football | 115 | 613 | Games between college football teams |
| Jazz | 198 | 2,742 | Collaborations between jazz musicians |
| C. elegans | 453 | 2,025 | Metabolic network of *C. elegans* |
| E-mail | 1,133 | 5,451 | E-mail exchanges at Universitat Rovira i Virgili |
| PGP | 10,680 | 24,316 | Giant component of the PGP web of trust |
| Condmat2003 | 31,163 | 120,029 | Condensed matter co-authorship network |

---

## Results

Each method was run **40 times** per dataset (seeds 1 to 40). `Q_max` is the best modularity found, `Communities` is the number of communities of that best partition, and `Q_avg` is the mean over the 40 runs.

### LPA

| Dataset | Communities | Q_max | Q_avg |
|---|---:|---:|---:|
| Karate | 4 | 0.415105 | 0.363655 |
| Dolphins | 5 | 0.522191 | 0.490699 |
| Political Books | 4 | 0.526229 | 0.502297 |
| College Football | 10 | 0.604570 | 0.587389 |
| Jazz | 4 | 0.442395 | 0.333083 |
| C. elegans | 23 | 0.395484 | 0.125172 |
| E-mail | 57 | 0.525982 | 0.365305 |
| PGP | 1913 | 0.752234 | 0.731791 |
| Condmat2003 | 5027 | 0.625042 | 0.610228 |

### LPAm (initialized with the LPA partition)

| Dataset | Communities | Q_max | Q_avg |
|---|---:|---:|---:|
| Karate | 4 | 0.419790 | 0.381246 |
| Dolphins | 6 | 0.523021 | 0.500393 |
| Political Books | 4 | 0.526938 | 0.515698 |
| College Football | 10 | 0.604570 | 0.587840 |
| Jazz | 4 | 0.444174 | 0.359378 |
| C. elegans | 23 | 0.423861 | 0.396381 |
| E-mail | 47 | 0.546880 | 0.519969 |
| PGP | 1896 | 0.755179 | 0.735811 |
| Condmat2003 | 4175 | 0.633101 | 0.615599 |

### LPAm+ with classical greedy merging (7 smaller graphs)

| Dataset | Communities | Q_max | Q_avg |
|---|---:|---:|---:|
| Karate | 4 | 0.419790 | 0.389963 |
| Dolphins | 4 | 0.526799 | 0.510506 |
| Political Books | 4 | 0.526938 | 0.520236 |
| College Football | 10 | 0.604570 | 0.602603 |
| Jazz | 3 | 0.444469 | 0.359653 |
| C. elegans | 9 | 0.433320 | 0.401860 |
| E-mail | 10 | 0.575712 | 0.542947 |

### LPAm+ with multistep greedy merging (MSG, `l = 10`)

| Dataset | Communities | Q_max | Q_avg |
|---|---:|---:|---:|
| Karate | 4 | 0.419790 | 0.389963 |
| Dolphins | 4 | 0.526799 | 0.510506 |
| Political Books | 4 | 0.526938 | 0.520236 |
| College Football | 10 | 0.604570 | 0.602603 |
| Jazz | 3 | 0.444469 | 0.359653 |
| C. elegans | 9 | 0.432645 | 0.401831 |
| E-mail | 11 | 0.579936 | 0.543006 |
| PGP | 117 | 0.883972 | 0.882094 |
| Condmat2003 | 988 | 0.770679 | 0.767865 |

### Comparison with the literature (best modularity found)

| Dataset | This repo, LPAm+ (MSG) | Liu & Murata, LPAm+ [2] | Reference value |
|---|---:|---:|---|
| Karate | 0.419790 | 0.420 | 0.419790 (proven optimum [1]) |
| Dolphins | 0.526799 | 0.529 | 0.528519 (proven optimum [1]) |
| Political Books | 0.526938 | 0.527 | 0.527237 (proven optimum [1]) |
| College Football | 0.604570 | 0.605 | 0.604570 (proven optimum [1]) |
| Jazz | 0.444469 | 0.445 | 0.445144 (proven optimum [1]) |
| C. elegans | 0.432645 | 0.452 | 0.453248 (VNDS [1]) |
| E-mail | 0.579936 | 0.582 | 0.582828 (VNDS [1]) |
| PGP | 0.883972 | 0.884 | 0.886081 (VNDS [1]) |
| Condmat2003 | 0.770679 | 0.755* | 0.814* (SS-ML, reported in [2]) |

\* Liu & Murata use a preprocessed Condmat2003 graph with 27,519 nodes and 116,181 edges, while this repository uses the 31,163-node / 120,029-edge version. These values are not directly comparable.

### Discussion

- **LPA → LPAm.** The modularity-aware refinement always improves the LPA partition. The gain is largest on the averages of C. elegans (0.125 → 0.396) and E-mail (0.365 → 0.520), where LPA sometimes collapses into a few giant communities.
- **LPAm → LPAm+.** Merging is the decisive step on the larger graphs: PGP goes from 0.755 to 0.884 and Condmat2003 from 0.633 to 0.771, while the number of communities drops from 1,896 to 117 and from 4,175 to 988. This matches the diagnosis of Liu & Murata: LPAm leaves many small communities of similar volume that no single-vertex move can join.
- **Greedy vs MSG.** Both merging strategies give identical results on the five smallest graphs and nearly identical results on C. elegans and E-mail, but only MSG finishes on PGP and Condmat2003.
- **Distance to the optimum.** LPAm+ reaches the proven optimum on Karate and College Football, and is within 0.0003 to 0.002 of it on Political Books, Jazz and Dolphins. The gap is larger on C. elegans (0.433 vs 0.453), and the averages are clearly below those reported by Liu & Murata (e.g. 0.390 vs 0.418 on Karate). These remaining gaps are precisely what the VNS stage is designed to close.

---

## Repository structure

```text
.
├── README.md
├── literature/                          # reference articles (see License)
├── vns_modularity_clustering.ipynb      # implementation and experiments
└── .gitignore
```

The notebook is organized as follows:

1. data acquisition;
2. heuristic methods: LPA, modularity, LPAm, LPAm+ (greedy and MSG), VNS perturbations;
3. experiments and performance summaries;
4. references.

## Getting started

### Requirements

- Julia 1.9 or later (developed with Julia 1.11.6)
- Jupyter (with IJulia) or VS Code with the Julia extension
- Internet access for the first download of the datasets

### Install the Julia dependencies

```julia
using Pkg
Pkg.add(["Graphs", "MatrixDepot", "SimpleWeightedGraphs", "IJulia"])
```

`Random` is part of the Julia standard library and does not need to be installed.

### Run the project

```bash
git clone https://github.com/tarteeg/vns-modularity-clustering.git
cd vns-modularity-clustering
jupyter notebook vns_modularity_clustering.ipynb
```

Run the cells in order. The LPAm+ benchmarks on PGP and Condmat2003 are the slowest part of the notebook.

---

## References

**[1]** D. Aloise, G. Caporossi, P. Hansen, L. Liberti, S. Perron and M. Ruiz, *Modularity maximization in networks by variable neighborhood search*, Contemporary Mathematics, vol. 588, pp. 113–127, 2013.

**[2]** X. Liu and T. Murata, *Advanced modularity-specialized label propagation algorithm for detecting communities in networks*, Physica A, vol. 389, pp. 1493–1500, 2010.

**[3]** P. Schuetz and A. Caflisch, *Efficient modularity optimization by multistep greedy algorithm and vertex mover refinement*, Physical Review E, vol. 77, 046112, 2008.

**[4]** M. J. Barber and J. W. Clark, *Detecting network communities by propagating labels under constraints*, Physical Review E, vol. 80, 026129, 2009.

**[5]** M. E. J. Newman, *Fast algorithm for detecting community structure in networks*, Physical Review E, vol. 69, 066133, 2004.

**[6]** D. A. Bader, H. Meyerhenke, P. Sanders and D. Wagner (eds.), *Graph Partitioning and Graph Clustering*, 10th DIMACS Implementation Challenge Workshop, Contemporary Mathematics 588, American Mathematical Society, 2013.

## License

The original code in this repository is released under the MIT License.

The scientific articles and PDF files included in `literature/` remain subject to the copyrights and redistribution terms of their respective authors, publishers, or organizations. They are included for academic reference only.
