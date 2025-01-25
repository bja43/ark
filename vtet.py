import numpy as np

from numpy.linalg import det, inv, svd
from itertools import combinations
from scipy.stats import norm, chi2


class Test:

    def __init__(self, df):

        self.df = df
        self.n, self.p = df.shape
        self.S = df.cov().values

    def variables(self):

        return self.df.columns

    def tetrads(self, tets, resample=False, frac=0.5):

        pass

    def pure_triple(self, S, v, fisher=True, resample=False, frac=0.5):

        tets = []

        for a in combinations(S, 2):
            b = tuple([b for b in S if b not in a] + [v])
            tet = (a, b)
            tets.append(tet)

        if not fisher: return self.tetrads(tets)

        p_values = [max(self.tetrads([tet], resample, frac), 1e-16) for tet in tets]
        combined = -2 * np.sum(np.log(p_values))

        print()
        print(S, v)
        print(p_values)
        print()

        return 1 - chi2.cdf(combined, 2 * len(p_values))


class Wishart(Test):

    def tetrads(self, tets, resample=False, frac=0.5):

        if not resample: S = self.S
        else: S = self.df.sample(frac=frac).cov.values

        tet = tets[0]
        a, b = tet

        sigma2 = (self.n + 1) / (self.n - 1)
        sigma2 *= det(self.S[np.ix_(a, a)])
        sigma2 *= det(self.S[np.ix_(b, b)])
        sigma2 -= det(self.S[np.ix_(a + b, a + b)])
        sigma2 /= (self.n - 2)

        z_score = det(self.S[np.ix_(a, b)]) / np.sqrt(sigma2)

        return 2 * norm.cdf(-abs(z_score))


class Bollen_Ting(Test):

    def tetrads(self, tets, resample=False, frac=0.5):

        if not resample: S = self.S
        else: S = self.df.sample(frac=frac).cov.values

        V = {x for tet in tets for i in (0, 1) for x in tet[i]}
        s = [tuple(sorted([i, j])) for i, j in combinations(V, 2)]
        ss = np.zeros([len(s), len(s)])

        for i, x in enumerate(s):
            for j, y in enumerate(s):
                ss[i, j] += self.S[x[0], y[0]] * self.S[x[1], y[1]]
                ss[i, j] += self.S[x[0], y[1]] * self.S[x[1], y[0]]

        dt_ds = np.zeros([len(s), len(tets)])
        t = np.zeros([len(tets), 1])

        for i, tet in enumerate(tets):

            a, b = tet
            z = len(a)

            A = self.S[np.ix_(a, b)]
            t[i] = det(A)
            AdjT = t[i] * inv(A).T

            for j, x in enumerate(s):
                for k in range(z):
                    for l in range(z):
                        if a[k] in x and b[l] in x:
                            dt_ds[j, i] = AdjT[k, l]

        tt = dt_ds.T @ ss @ dt_ds
        T = self.n * (t.T @ inv(tt) @ t)[0, 0]

        return 1 - chi2.cdf(T, len(tets))


class Ark(Test):

    def __init__(self, df, tbd=None):

        super().__init__(df)

    def tetrads(self, tets, resample=False, frac=0.5):

        if not resample: S = self.S
        else: S = self.df.sample(frac=frac).cov.values

        tet = tets[0]
        a, b = tet
        z = len(a)

        XY = self.S[np.ix_(a, b)]

        # if T is None:
        U, _, VT = svd(XY)
        # else:
        # U, _, VT = svd(T[np.ix_(a, b)])

        XXi = inv(self.S[np.ix_(a, a)])
        YYi = inv(self.S[np.ix_(b, b)])

        A = U.T @ XXi @ U
        B = VT @ YYi @ VT.T
        C = U.T @ XXi @ XY @ YYi @ VT.T

        a = [i for i in range(z)]
        b = [i + z for i in range(z)]

        R = np.zeros([2 * z, 2 * z])
        R[np.ix_(a, a)] = A
        R[np.ix_(a, b)] = C
        R[np.ix_(b, a)] = C.T
        R[np.ix_(b, b)] = B

        D = np.diag(np.sqrt(np.diag(R)))
        Di = inv(D)
        R = Di @ R @ Di

        idx = [a[-1], b[-1]]
        idx += [i for i in a[:-1]]
        idx += [i for i in b[:-1]]

        P = inv(R[np.ix_(idx, idx)])

        p_corr = - P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
        z_score = np.sqrt(self.n - len(idx) - 1) * np.arctanh(p_corr)

        return 2 * norm.cdf(-abs(z_score))
