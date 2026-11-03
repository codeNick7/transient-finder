"""Unsupervised Anomaly Detection for discovering rare celestial interlopers and extreme NEOs."""

import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest

from core.models import CandidateRecord
from ai.feature_extractor import extract_candidate_features, features_to_vector

logger = logging.getLogger(__name__)


class CelestialAnomalyDetector:
    """Uses unsupervised isolation trees to detect astrophysically anomalous bodies
    (e.g. interstellar objects, retrograde NEOs, extreme Earth grazers).
    """
    
    def __init__(self, contamination: float = 0.05):
        self.detector = IsolationForest(
            n_estimators=80,
            contamination=contamination,
            random_state=42
        )
        self._bootstrap_normal_population()

    def _bootstrap_normal_population(self):
        """Fit on a representative population of standard main belt and typical Apollo/Amor NEOs."""
        rng = np.random.RandomState(42)
        n = 1000
        
        rate = rng.uniform(0.5, 4.0, size=n)
        log_rate = np.log10(rate)
        arc = rng.uniform(1.0, 30.0, size=n)
        nobs = rng.randint(4, 15, size=n)
        rms = rng.uniform(0.12, 0.35, size=n)
        elong = rng.uniform(100.0, 175.0, size=n)
        vmag = rng.uniform(18.0, 21.0, size=n)
        hmag = rng.uniform(19.0, 25.0, size=n)
        accel = rng.uniform(0.01, 0.1, size=n)
        linearity = rng.uniform(0.97, 1.0, size=n)
        a = rng.uniform(1.1, 2.6, size=n)
        e = rng.uniform(0.15, 0.55, size=n)
        inc = rng.uniform(2.0, 25.0, size=n)
        tj = rng.uniform(3.2, 5.0, size=n)
        moid = rng.uniform(0.01, 0.20, size=n)
        twilight = np.zeros(n)
        extreme = np.zeros(n)
        stationary = np.zeros(n)
        leo = np.zeros(n)
        cadence = arc / nobs
        
        X_normal = np.column_stack([
            rate, log_rate, arc, nobs, rms, elong, vmag, hmag,
            accel, linearity, a, e, inc, tj, moid,
            twilight, extreme, stationary, leo, cadence
        ])
        self.detector.fit(X_normal)

    def evaluate_anomaly(self, candidate: CandidateRecord) -> Tuple[bool, float, str]:
        """Determine if candidate is an outlier.
        
        Returns:
            (is_anomaly: bool, anomaly_score: float [0..1 where 1 is extreme outlier], reason: str)
        """
        feats = extract_candidate_features(candidate)
        vec = features_to_vector(feats).reshape(1, -1)
        
        raw_score = self.detector.score_samples(vec)[0]  # Lower is more anomalous
        # Normalize score roughly to 0..1 where > 0.6 is suspicious anomaly
        normalized_anomaly = float(np.clip(1.0 - (raw_score + 0.5), 0.0, 1.0))
        is_anomaly = normalized_anomaly > 0.62
        
        reasons = []
        if feats["eccentricity"] >= 1.0:
            reasons.append("Hyperbolic / Interstellar orbit trajectory (e >= 1.0)")
        if feats["inclination_deg"] > 90.0:
            reasons.append("Retrograde orbit (inclination > 90 deg)")
        if feats["moid_au"] < 0.005:
            reasons.append("Extreme ultra-close Earth approach (MOID < 0.005 AU / 2 LD)")
        if feats["motion_rate_deg_day"] > 25.0:
            reasons.append("Unusually high apparent motion rate")
            
        reason_str = "; ".join(reasons) if reasons else ("General kinematic distribution outlier" if is_anomaly else "Standard orbital distribution")
        return is_anomaly, normalized_anomaly, reason_str
