"""Auditable centroid and projection trees; no test-dependent model choices."""
from __future__ import annotations
from itertools import combinations
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


class ProjectionTree:
    def __init__(self, max_depth=3, min_samples_leaf=8, mode='p2', random_state=0,
                 max_pairs=None):
        self.max_depth, self.min_samples_leaf = max_depth, min_samples_leaf
        self.mode, self.random_state, self.max_pairs = mode, random_state, max_pairs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.rng = np.random.default_rng(self.random_state)
        self.nodes = []
        self._build(np.asarray(X, float), np.asarray(y, int), 0)
        return self

    def _directions(self, X, y):
        p = X.shape[1]
        classes = np.unique(y)
        if self.mode == 'random':
            w = self.rng.normal(size=(64, p))
            w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
            return w, np.zeros(len(w))
        if self.mode in ('lda', 'logistic'):
            if self.mode == 'lda':
                model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
            else:
                model = LogisticRegression(C=1, max_iter=300, random_state=self.random_state).fit(X, y)
            return np.atleast_2d(model.coef_), np.zeros(len(np.atleast_2d(model.coef_)))
        means = {c: X[y == c].mean(axis=0) for c in classes}
        if self.mode == 'dense':
            ws, bs = [], []
            for a, b in combinations(classes, 2):
                ws.append(means[b] - means[a])
                bs.append((means[b] @ means[b] - means[a] @ means[a]) / 2)
            return np.asarray(ws), np.asarray(bs)
        pairs = np.asarray(list(combinations(range(p), 2)), int)
        if not len(pairs):
            pairs = np.array([[0, 0]])
        if self.mode == 'random_pair' or self.max_pairs is not None:
            cap = min(len(pairs), self.max_pairs or 16)
            pairs = pairs[self.rng.choice(len(pairs), cap, replace=False)]
        ws, bs = [], []
        for a, b in combinations(classes, 2):
            w = np.zeros((len(pairs), p))
            w[np.arange(len(pairs))[:, None], pairs] = (means[b] - means[a])[pairs]
            mid = (means[b] + means[a]) / 2
            ws.append(w)
            bs.append(w @ mid)
        return np.concatenate(ws), np.concatenate(bs)

    def _build(self, X, y, depth):
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / len(y)
        idx = len(self.nodes)
        node = dict(depth=depth, proba=probs.tolist(), n=len(y), leaf=True)
        self.nodes.append(node)
        if depth >= self.max_depth or len(y) < 2*self.min_samples_leaf or np.count_nonzero(counts) <= 1:
            return idx
        parent_gini = 1 - probs @ probs
        best = None
        best_gain = 1e-12
        W, B = self._directions(X, y)
        valid = np.linalg.norm(W, axis=1) > 1e-12
        W, B = W[valid], B[valid]
        # Bounded chunks avoid an n x all-candidates allocation.
        best_candidates = []
        Y = np.eye(self.n_classes)[y]
        for start in range(0, len(W), 128):
            w, b = W[start:start+128], B[start:start+128]
            left = X @ w.T <= b
            nl = left.sum(axis=0)
            nr = len(y) - nl
            cl = Y.T @ left.astype(float)
            cr = counts[:, None] - cl
            score = (np.sum(cl*cl, axis=0) / np.maximum(nl, 1) +
                     np.sum(cr*cr, axis=0) / np.maximum(nr, 1)) / len(y) - probs @ probs
            score[(nl < self.min_samples_leaf) | (nr < self.min_samples_leaf)] = -np.inf
            for j in np.argsort(score)[-min(16, len(score)):]:
                best_candidates.append((float(score[j]), start+int(j)))
            j = int(np.argmax(score))
            if score[j] > best_gain:
                best_gain = float(score[j])
                best = (w[j].copy(), float(b[j]), left[:, j])
        # Threshold tuning is a distinct diagnostic variant, not the original P2.
        if self.mode in ('hybrid', 'tuned', 'lda', 'logistic', 'random') and len(W):
            if self.mode in ('hybrid', 'tuned'):
                candidates = [j for _, j in sorted(best_candidates, reverse=True)[:16]]
            else:
                candidates = range(len(W))
            for j in candidates:
                z = X @ W[j]
                stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=self.min_samples_leaf,
                                               random_state=self.random_state).fit(z[:, None], y)
                if stump.tree_.node_count < 3:
                    continue
                t = stump.tree_
                gain = parent_gini - (t.n_node_samples[1]*t.impurity[1] +
                                      t.n_node_samples[2]*t.impurity[2]) / len(y)
                if gain > best_gain + 1e-12:
                    best_gain = float(gain)
                    b = float(t.threshold[0])
                    best = W[j].copy(), b, z <= b
        if self.mode in ('hybrid', 'lda', 'logistic', 'random'):
            stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=self.min_samples_leaf,
                                           random_state=self.random_state).fit(X, y)
            t = stump.tree_
            if t.node_count > 1:
                gain = parent_gini - (t.n_node_samples[1]*t.impurity[1] +
                                      t.n_node_samples[2]*t.impurity[2]) / len(y)
                if gain > best_gain + 1e-12:
                    w = np.zeros(X.shape[1]); w[t.feature[0]] = 1
                    b = float(t.threshold[0])
                    best = w, b, X @ w <= b
        if best is None:
            return idx
        w, b, left = best
        node.update(leaf=False, w=w.tolist(), b=b,
                    left=self._build(X[left], y[left], depth+1),
                    right=self._build(X[~left], y[~left], depth+1))
        return idx

    def predict_proba(self, X):
        out = np.zeros((len(X), self.n_classes))
        def walk(idx, rows):
            if len(rows) == 0: return
            node = self.nodes[idx]
            if node['leaf']:
                out[rows] = node['proba']; return
            left = X[rows] @ np.asarray(node['w']) <= node['b']
            walk(node['left'], rows[left]); walk(node['right'], rows[~left])
        walk(0, np.arange(len(X)))
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def complexity(self, X):
        burden = np.zeros(len(X)); lengths = np.zeros(len(X)); radius = np.full(len(X), np.inf)
        def walk(idx, rows):
            n = self.nodes[idx]
            if n['leaf'] or len(rows) == 0: return
            w = np.asarray(n['w']); z = X[rows] @ w - n['b']
            burden[rows] += np.count_nonzero(np.abs(w) > 1e-12)
            lengths[rows] += 1
            radius[rows] = np.minimum(radius[rows], np.abs(z) / np.sum(np.abs(w)))
            mask = z <= 0
            walk(n['left'], rows[mask]); walk(n['right'], rows[~mask])
        walk(0, np.arange(len(X)))
        return dict(leaves=sum(n['leaf'] for n in self.nodes),
                    depth=max(n['depth'] for n in self.nodes),
                    path_tests=float(lengths.mean()), path_burden=float(burden.mean()),
                    radius_median=float(np.median(radius[np.isfinite(radius)])) if np.isfinite(radius).any() else None)
