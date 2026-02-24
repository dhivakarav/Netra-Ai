import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque

class AnomalyModel:
    def __init__(self, max_samples=800, refit_every=200):
        self.buf = deque(maxlen=max_samples)
        self.refit_every = int(refit_every)
        self.model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
        self.fitted = False
        self.count = 0

    def push(self, feat):
        self.buf.append(np.asarray(feat, dtype=np.float32))
        self.count += 1
        if len(self.buf) >= 200 and (self.count % self.refit_every == 0):
            X = np.stack(list(self.buf), axis=0)
            self.model.fit(X)
            self.fitted = True

    def score(self, feat):
        if not self.fitted:
            return 0.0
        X = np.asarray(feat, dtype=np.float32).reshape(1, -1)
        # higher = more anomalous
        s = -float(self.model.score_samples(X)[0])
        return s
