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

        if not impure: pure_clusters.append(S)

    return pure_clusters


# def grow_clusters(pure_clusters, threshold):
#
#     W = []
#     for pure_cluster in pure_clusters:
#         for w in pure_cluster:
#             if w in W: continue
#             W.append(w)
#
#     clusters = pure_clusters
#
#     for cluster in clusters:
#         for S in combinations(cluster, 2):
#             for w in W:
#                 if w in cluster: continue
#                 test_cluster = S
#
#
#
#     return clusters


# def select_clusters(clusters):
#
#     sizes = [len(cluster) for cluster in clusters]
#     sorted_clusters = [cluster for _, cluster in sorted(zip(sizes, clusters), reverse=True)]
#     selected_clusters = []
#
#     for cluster1 in sorted_clusters:
#
#         for cluster2 in selected_clusters:
#             if any([v in cluster1 for v in cluster2]):
#                 continue
#
#     return None


n = int(sys.argv[1])

g = np.zeros([6, 6], dtype=np.uint8)

# g[0, 4] = 1
# g[1, 4] = 1
# g[2, 4] = 1
# g[3, 4] = 1

# g[0, 4] = 1
# g[1, 4] = 1
# g[2, 5] = 1
# g[3, 5] = 1
# g[4, 5] = 1

g = er_dag(4, d=0.5)

obs = [0, 1, 2, 3]

_, B, O = corr(g)
X = simulate(B, O, n)
df = pd.DataFrame(X)[obs]

test = Wishart(df)
pure_triples = fpc(test, resample=True)
print("Wishart", pure_triples)
print()

test = Bollen_Ting(df)
pure_triples = fpc(test, fisher=False)
print("Bollen Ting", pure_triples)
print()

test = Ark(df)
pure_triples = fpc(test, resample=True)
print("Ark", pure_triples)
print()
