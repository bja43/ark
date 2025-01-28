import sys
import numpy as np
import pandas as pd

from itertools import combinations
from dao import corr, er_dag, simulate

from vtet import Wishart, Bollen_Ting, Ark


def fpc(test, alpha=0.05, fisher=True, resample=False, frac=0.5):

    V = test.variables()
    pure_clusters = []

    for S in combinations(V, 3):
        impure = False
        for v in V:
            if v in S: continue

            p_value = test.pure_triple(S, v, fisher, resample, frac)
            print(p_value)

            if p_value < alpha:
                impure = True
                break

        if impure: continue
        pure_clusters.append(set(S))

    return pure_clusters


def grow_clusters(pure_clusters, threshold=0.5):

    # what are the members of pure-clusters?
    # does order matter?

    W = []
    for pure_cluster in pure_clusters:
        for w in pure_cluster:
            if w in W: continue
            W.append(w)

    clusters = pure_clusters.copy()

    for cluster in clusters:
        for w in W:
            if w in cluster: continue

            acc = 0
            rej = 0

            for S in combinations(cluster, 2):
                test_cluster = set(S)
                test_cluster.add(w)

                if test_cluster in pure_clusters:
                    acc += 1
                else:
                    rej += 1

            if acc / (acc + rej) >= threshold:

                new_cluster = set(cluster)
                new_cluster.add(w)

                clusters.append(new_cluster)
                remove_subsets(pure_clusters, new_cluster)

    return clusters


def remove_subsets(S, a):

    to_remove = []

    for b in S:
        if a.issuperset(b):
            to_remove.append(b)

    for b in to_remove:
        S.remove(b)


def select_clusters(clusters):

    sizes = [len(cluster) for cluster in clusters]
    sorted_clusters = [cluster for _, cluster in sorted(zip(sizes, clusters))]
    selected_clusters = []

    while sorted_clusters:
        cluster = sorted_clusters.pop(-1)
        selected_clusters.append(cluster)
        remove_subsets(sorted_clusters, cluster)

    return selected_clusters


n = int(sys.argv[1])

g = np.zeros([6, 6], dtype=np.uint8)

# g[0, 4] = 1
# g[1, 4] = 1
# g[2, 4] = 1
# g[3, 4] = 1

g[0, 4] = 1
g[1, 4] = 1
g[2, 5] = 1
g[3, 5] = 1
g[4, 5] = 1

# g = er_dag(4, d=1.0)

obs = [0, 1, 2, 3]

_, B, O = corr(g)
X = simulate(B, O, n)
df = pd.DataFrame(X)[obs]

# test = Wishart(df)
# pure_triples = fpc(test, resample=True)
# print("Wishart", pure_triples)
# print()

# test = Bollen_Ting(df)
# pure_triples = fpc(test, fisher=False)
# print("Bollen Ting", pure_triples)
# print()

test = Ark(df)
pure_triples = fpc(test, resample=True)
print("Ark", pure_triples)
print()
clusters = grow_clusters(pure_triples, 0.1)
print(clusters)
print()
clusters = select_clusters(clusters)
print(clusters)
print()
