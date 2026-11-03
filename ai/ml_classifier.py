"""Supervised Machine Learning Real-Bogus Classifier for transient NEO candidate vetting."""

import logging
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.models import CandidateRecord
from ai.feature_extractor import extract_candidate_features, features_to_vector, FEATURE_NAMES

logger = logging.getLogger(__name__)


class RealBogusClassifier:
    """Supervised ensemble ML model determining the probability that a candidate is a Real NEO vs Bogus artifact."""
    
    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=4,
                random_state=42
            ))
        ])
        self.is_trained = False
        self._bootstrap_default_model()

    def _generate_synthetic_training_data(self, n_samples: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate physically consistent synthetic training set reflecting survey characteristics:
        Real NEOs (Class 1) vs Bogus Artifacts (Class 0: dipoles, satellites, spikes, noise linkages).
        """
        rng = np.random.RandomState(42)
        half = n_samples // 2
        
        # Real Candidates (Class 1)
        # Rates: 0.5 to 15.0 deg/day; RMS: 0.10 to 0.45 arcsec; Elong: 90 to 175 deg; a: 0.8 to 2.8 AU; e: 0.1 to 0.7
        r_rate = rng.lognormal(mean=0.5, sigma=0.8, size=half)
        r_rate = np.clip(r_rate, 0.4, 20.0)
        r_log_rate = np.log10(r_rate)
        r_arc = rng.uniform(0.5, 48.0, size=half)
        r_nobs = rng.randint(3, 25, size=half)
        r_rms = rng.normal(loc=0.22, scale=0.08, size=half)
        r_rms = np.clip(r_rms, 0.08, 0.55)
        r_elong = rng.uniform(85.0, 178.0, size=half)
        r_vmag = rng.uniform(17.0, 21.8, size=half)
        r_hmag = rng.uniform(16.0, 27.0, size=half)
        r_accel = rng.exponential(scale=0.05, size=half)
        r_linearity = rng.uniform(0.96, 1.0, size=half)
        r_a = rng.uniform(0.8, 3.2, size=half)
        r_e = rng.uniform(0.1, 0.75, size=half)
        r_inc = rng.exponential(scale=12.0, size=half)
        r_tj = rng.uniform(3.0, 5.5, size=half)
        r_moid = rng.uniform(0.005, 0.25, size=half)
        r_twilight = (r_elong < 55.0).astype(float)
        r_extreme = (r_rate > 30.0).astype(float)
        r_stationary = (r_rate < 0.05).astype(float)
        r_leo = np.zeros(half)
        r_cadence = r_arc / r_nobs
        
        X_real = np.column_stack([
            r_rate, r_log_rate, r_arc, r_nobs, r_rms, r_elong, r_vmag, r_hmag,
            r_accel, r_linearity, r_a, r_e, r_inc, r_tj, r_moid,
            r_twilight, r_extreme, r_stationary, r_leo, r_cadence
        ])
        y_real = np.ones(half)
        
        # Bogus Candidates (Class 0)
        # Modes: Dipoles (high RMS), Satellites (extreme rate, LEO a), Spikes (low elong, low motion), Noise links (high accel, low linearity)
        b_rate = np.zeros(half)
        b_rms = np.zeros(half)
        b_accel = np.zeros(half)
        b_elong = np.zeros(half)
        b_vmag = np.zeros(half)
        b_hmag = np.zeros(half)
        b_a = np.zeros(half)
        b_e = np.zeros(half)
        b_inc = np.zeros(half)
        b_tj = np.zeros(half)
        b_moid = np.zeros(half)
        b_arc = rng.uniform(0.1, 3.0, size=half)
        b_nobs = rng.randint(2, 6, size=half)
        b_linearity = rng.uniform(0.40, 0.94, size=half)
        
        # Split bogus into 4 quadrants
        q = half // 4
        # 1. Satellites / Debris
        b_rate[:q] = rng.uniform(32.0, 95.0, size=q)
        b_rms[:q] = rng.uniform(0.2, 0.5, size=q)
        b_a[:q] = rng.uniform(0.005, 0.08, size=q)
        b_e[:q] = rng.uniform(0.001, 0.05, size=q)
        b_inc[:q] = rng.choice([53.0, 70.0, 97.0], size=q) + rng.normal(0, 1.0, size=q)
        b_hmag[:q] = rng.uniform(30.0, 40.0, size=q)
        b_vmag[:q] = rng.uniform(12.0, 16.0, size=q)
        b_elong[:q] = rng.uniform(60.0, 160.0, size=q)
        b_accel[:q] = rng.uniform(0.5, 3.0, size=q)
        
        # 2. Subtraction dipoles
        b_rate[q:2*q] = rng.uniform(0.1, 1.2, size=q)
        b_rms[q:2*q] = rng.uniform(0.85, 2.8, size=q)
        b_a[q:2*q] = rng.uniform(0.5, 4.0, size=q)
        b_e[q:2*q] = rng.uniform(0.1, 0.9, size=q)
        b_inc[q:2*q] = rng.uniform(0.0, 50.0, size=q)
        b_hmag[q:2*q] = rng.uniform(22.0, 32.0, size=q)
        b_vmag[q:2*q] = rng.uniform(20.5, 23.5, size=q)
        b_elong[q:2*q] = rng.uniform(70.0, 160.0, size=q)
        b_accel[q:2*q] = rng.uniform(0.1, 1.5, size=q)
        
        # 3. Bright star diffraction spikes / Ghosts (twilight glare)
        b_rate[2*q:3*q] = rng.uniform(0.08, 0.45, size=q)
        b_rms[2*q:3*q] = rng.uniform(0.60, 1.8, size=q)
        b_a[2*q:3*q] = rng.uniform(0.5, 3.0, size=q)
        b_e[2*q:3*q] = rng.uniform(0.2, 0.8, size=q)
        b_inc[2*q:3*q] = rng.uniform(5.0, 40.0, size=q)
        b_hmag[2*q:3*q] = rng.uniform(24.0, 30.0, size=q)
        b_vmag[2*q:3*q] = rng.uniform(17.0, 20.0, size=q)
        b_elong[2*q:3*q] = rng.uniform(30.0, 52.0, size=q)  # Twilight
        b_accel[2*q:3*q] = rng.uniform(0.0, 0.4, size=q)
        
        # 4. Spurious noise linkages & stationary variable stars
        b_rate[3*q:] = rng.choice([0.01, 0.02, 4.5, 8.2], size=half - 3*q)
        b_rms[3*q:] = rng.uniform(0.70, 2.5, size=half - 3*q)
        b_a[3*q:] = rng.uniform(0.1, 5.0, size=half - 3*q)
        b_e[3*q:] = rng.uniform(0.0, 0.95, size=half - 3*q)
        b_inc[3*q:] = rng.uniform(0.0, 80.0, size=half - 3*q)
        b_hmag[3*q:] = rng.uniform(22.0, 32.0, size=half - 3*q)
        b_vmag[3*q:] = rng.uniform(18.0, 22.0, size=half - 3*q)
        b_elong[3*q:] = rng.uniform(70.0, 170.0, size=half - 3*q)
        b_accel[3*q:] = rng.uniform(1.2, 6.0, size=half - 3*q)
        
        b_log_rate = np.log10(np.maximum(1e-4, b_rate))
        b_tj = rng.uniform(0.5, 3.2, size=half)
        b_moid = rng.uniform(0.05, 0.8, size=half)
        b_twilight = (b_elong < 55.0).astype(float)
        b_extreme = (b_rate > 30.0).astype(float)
        b_stationary = (b_rate < 0.05).astype(float)
        b_leo = (b_rate > 25.0).astype(float)
        b_cadence = b_arc / np.maximum(1, b_nobs)
        
        X_bogus = np.column_stack([
            b_rate, b_log_rate, b_arc, b_nobs, b_rms, b_elong, b_vmag, b_hmag,
            b_accel, b_linearity, b_a, b_e, b_inc, b_tj, b_moid,
            b_twilight, b_extreme, b_stationary, b_leo, b_cadence
        ])
        y_bogus = np.zeros(half)
        
        X = np.vstack([X_real, X_bogus])
        y = np.concatenate([y_real, y_bogus])
        return X, y

    def _bootstrap_default_model(self):
        """Train the classifier on physical survey simulations."""
        X, y = self._generate_synthetic_training_data(n_samples=2000)
        self.pipeline.fit(X, y)
        self.is_trained = True
        logger.info("Real-Bogus Classifier successfully trained and calibrated.")

    def predict_real_probability(self, candidate: CandidateRecord) -> float:
        """Compute the ML-predicted probability (0.0 to 1.0) that the candidate is a Real NEO."""
        feats = extract_candidate_features(candidate)
        vec = features_to_vector(feats).reshape(1, -1)
        prob_real = self.pipeline.predict_proba(vec)[0][1]
        return float(prob_real)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return the relative feature importances from the gradient boosting model."""
        clf = self.pipeline.named_steps["clf"]
        importances = clf.feature_importances_
        sorted_pairs = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
        return {k: float(v) for k, v in sorted_pairs}
