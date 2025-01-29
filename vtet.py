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

        raise Exception("No Implemention of 'tetrads'")

    def pure_triple(self, S, v, fisher=True, resample=False, frac=0.5):

        tets = []

        for a in combinations(S, 2):
            b = tuple([b for b in S if b not in a] + [v])
            tet = (a, b)
            tets.append(tet)

        if not fisher: return self.tetrads(tets[:-1])

        p_values = [self.tetrad(tet, resample, frac) for tet in tets]
        p_values = [max(p_value, 1e-16) for p_value in p_values]
        combined = -2 * np.sum(np.log(p_values))

        print(S, v)
        print(p_values)

        return 1 - chi2.cdf(combined, 2 * len(p_values))


class Wishart(Test):

    def tetrad(self, tet, resample=False, frac=0.5):

        if resample:
            df = self.df.sample(frac=frac)
            n, _ = df.shape
            S = df.cov().values
        else:
            n = self.n
            S = self.S

        a, b = tet

        sigma2 = (n + 1) / (n - 1)
        sigma2 *= det(S[np.ix_(a, a)])
        sigma2 *= det(S[np.ix_(b, b)])
        sigma2 -= det(S[np.ix_(a + b, a + b)])
        sigma2 /= (n - 2)

        z_score = det(S[np.ix_(a, b)]) / np.sqrt(sigma2)

        return 2 * norm.cdf(-abs(z_score))


class Bollen_Ting(Test):

    def tetrad(self, tet, resample=False, frac=0.5):

        return self.tetrads([tet], resample, frac)

    # assumes S is cov of multivariate Gaussian
    def tetrads(self, tets, resample=False, frac=0.5):

        if resample:
            df = self.df.sample(frac=frac)
            n, _ = df.shape
            S = df.cov().values
        else:
            n = self.n
            S = self.S

        V = {x for tet in tets for i in (0, 1) for x in tet[i]}
        s = [tuple(sorted([i, j])) for i, j in combinations(V, 2)]
        ss = np.zeros([len(s), len(s)])

        for i, x in enumerate(s):
            for j, y in enumerate(s):
                ss[i, j] += S[x[0], y[0]] * S[x[1], y[1]]
                ss[i, j] += S[x[0], y[1]] * S[x[1], y[0]]

        dt_ds = np.zeros([len(s), len(tets)])
        t = np.zeros([len(tets), 1])

        for i, tet in enumerate(tets):

            a, b = tet
            z = len(a)

            A = S[np.ix_(a, b)]
            t[i] = det(A)
            AdjT = t[i] * inv(A).T

            for j, x in enumerate(s):
                for k in range(z):
                    for l in range(z):
                        if a[k] in x and b[l] in x:
                            dt_ds[j, i] = AdjT[k, l]

        tt = dt_ds.T @ ss @ dt_ds
        T = n * (t.T @ inv(tt) @ t)[0, 0]

        return 1 - chi2.cdf(T, len(tets))


class CCA(Test):

    # assumes |a| = |b| = k and r = k - 1
    def tetrad(self, tet, resample=False, frac=0.5):

        if resample:
            df = self.df.sample(frac=frac)
            n, _ = df.shape
            S = df.cov().values
        else:
            n = self.n
            S = self.S

        a, b = tet
        k = len(a)

        XY = S[np.ix_(a, b)]
        a = svd(XY)[1][k - 1:]
        stat = np.sum(np.log(1 - np.power(a, 2)))
        stat *= k + 3 / 2 - n

        return 1 - chi2.cdf(stat, 1)


class ARK(Test):

    def __init__(self, df, sp=1):

        super().__init__(df)
        if sp > 0: self.sp = sp
        else: self.sp = 1 - sp
        self.S1 = np.cov(df.values[:int(self.sp * self.n), :].T)
        self.S2 = np.cov(df.values[int(self.sp * self.n):, :].T)

    def tetrad(self, tet, resample=False, frac=0.5):

        if resample:
            df = self.df.sample(frac=frac)
            n, _ = df.shape
            S1 = np.cov(df.values[:int(self.sp * n), :].T)
            S2 = np.cov(df.values[int(self.sp * n):, :].T)
        else:
            n = self.n
            S1 = self.S1
            S2 = self.S2

        tet = tet
        a, b = tet
        z = len(a)

        if self.sp < 1: XY = S2[np.ix_(a, b)]
        else: XY = S1[np.ix_(a, b)]
        U, _, VT = svd(XY)

        if self.sp < 1: XY = S1[np.ix_(a, b)]
        XXi = inv(S1[np.ix_(a, a)])
        YYi = inv(S1[np.ix_(b, b)])

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
        # idx += [i for i in b[:-1]]

        P = inv(R[np.ix_(idx, idx)])

        p_corr = - P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
        z_score = np.arctanh(p_corr)
        z_score *= np.sqrt(int(self.sp * n) - len(idx) - 1)

        return 2 * norm.cdf(-abs(z_score))
