import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class SLearner:
    """S-Learner custom wrapper."""
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=100, max_depth=8,
            random_state=self.random_state, n_jobs=-1,
        )

    def fit(self, X, treatment, y):
        X_aug = X.copy()
        X_aug["treatment"] = treatment
        self.model.fit(X_aug, y)
        return self

    def predict(self, X):
        X_t1 = X.copy(); X_t1["treatment"] = 1
        X_t0 = X.copy(); X_t0["treatment"] = 0
        return self.model.predict(X_t1) - self.model.predict(X_t0)


class TLearner:
    """T-Learner custom wrapper."""
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_t = RandomForestRegressor(
            n_estimators=100, max_depth=8,
            random_state=self.random_state, n_jobs=-1,
        )
        self.model_c = RandomForestRegressor(
            n_estimators=100, max_depth=8,
            random_state=self.random_state, n_jobs=-1,
        )

    def fit(self, X, treatment, y):
        mask_t = treatment == 1
        self.model_t.fit(X[mask_t], y[mask_t])
        self.model_c.fit(X[~mask_t], y[~mask_t])
        return self

    def predict(self, X):
        return self.model_t.predict(X) - self.model_c.predict(X)


class XLearner:
    """X-Learner custom wrapper."""
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_t = RandomForestRegressor(
            n_estimators=100, max_depth=8,
            random_state=self.random_state, n_jobs=-1,
        )
        self.model_c = RandomForestRegressor(
            n_estimators=100, max_depth=8,
            random_state=self.random_state, n_jobs=-1,
        )
        self.cate_model_t = RandomForestRegressor(
            n_estimators=100, max_depth=6,
            random_state=self.random_state, n_jobs=-1,
        )
        self.cate_model_c = RandomForestRegressor(
            n_estimators=100, max_depth=6,
            random_state=self.random_state, n_jobs=-1,
        )
        from sklearn.linear_model import LogisticRegression
        self.ps_model = LogisticRegression(max_iter=1000, random_state=self.random_state)

    def fit(self, X, treatment, y):
        mask_t = treatment == 1
        X_np = X.values if hasattr(X, 'values') else X

        # Stage 1: Response models
        self.model_t.fit(X_np[mask_t], y[mask_t])
        self.model_c.fit(X_np[~mask_t], y[~mask_t])

        # Stage 2: Imputed individual effects
        tau_treated = y[mask_t] - self.model_c.predict(X_np[mask_t])
        tau_control = self.model_t.predict(X_np[~mask_t]) - y[~mask_t]

        # Stage 3: CATE models
        self.cate_model_t.fit(X_np[mask_t], tau_treated)
        self.cate_model_c.fit(X_np[~mask_t], tau_control)

        # Propensity model
        self.ps_model.fit(X_np, treatment)
        return self

    def predict(self, X):
        X_np = X.values if hasattr(X, 'values') else X
        ps = self.ps_model.predict_proba(X_np)[:, 1]
        return ps * self.cate_model_c.predict(X_np) + (1 - ps) * self.cate_model_t.predict(X_np)
