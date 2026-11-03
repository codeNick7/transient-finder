"""Multi-tier heuristic validation engine for astronomical NEO candidate vetting."""

import math
from typing import Dict, Any, List
from config import CONFIG
from core.models import CandidateRecord, HeuristicBreakdown
from ai.feature_extractor import extract_candidate_features


class HeuristicScoringEngine:
    """Computes transparent, physics-grounded heuristic scores across 5 scientific pillars."""
    
    def __init__(self, thresholds=CONFIG.thresholds):
        self.t = thresholds

    def evaluate_candidate(self, candidate: CandidateRecord) -> HeuristicBreakdown:
        feats = extract_candidate_features(candidate)
        
        s_astro = self._score_astrometric_consistency(feats)
        s_orbit = self._score_orbit_feasibility(feats)
        s_obs = self._score_observational_reliability(feats)
        s_sat = self._score_satellite_veto(feats)
        s_photo = self._score_photometric_stability(feats)
        
        # Weighted composite score
        w = self.t
        composite = (
            s_astro * w.weight_astrometric_consistency +
            s_orbit * w.weight_orbit_feasibility +
            s_obs * w.weight_observational_reliability +
            s_sat * w.weight_satellite_veto +
            s_photo * w.weight_photometric_stability
        )
        # If satellite veto triggered or extreme speed, cap composite score
        if s_sat <= 15.0 or feats["is_extreme_speed"]:
            composite = min(composite, 25.0)

        # If subtraction dipole / high astrometric residual, cap composite score
        if feats["rms_residual_arcsec"] > 0.90:
            composite = min(composite, 38.0)

        composite = float(max(0.0, min(100.0, composite)))
        
        return HeuristicBreakdown(
            astrometric_consistency_score=float(s_astro),
            orbit_feasibility_score=float(s_orbit),
            observational_reliability_score=float(s_obs),
            satellite_veto_score=float(s_sat),
            photometric_stability_score=float(s_photo),
            composite_ai_score=composite
        )

    def _score_astrometric_consistency(self, f: Dict[str, float]) -> float:
        """Evaluate astrometric positional accuracy, linearity, and acceleration stability."""
        score = 100.0
        rms = f["rms_residual_arcsec"]
        accel = f["acceleration_deg_day2"]
        linearity = f["linearity_r2"]
        
        # RMS penalties
        if rms <= 0.25:
            score -= 0.0
        elif rms <= 0.50:
            score -= (rms - 0.25) * 60.0  # up to -15
        elif rms <= 0.80:
            score -= 15.0 + (rms - 0.50) * 80.0  # up to -39
        else:
            score -= min(70.0, 39.0 + (rms - 0.80) * 50.0)
            
        # Non-linearity penalty
        if linearity < 0.95:
            score -= (0.95 - linearity) * 200.0
            
        # Physical acceleration check: topocentric curvature > 0.5 deg/day^2 is suspect
        if accel > 0.3:
            score -= min(40.0, (accel - 0.3) * 30.0)
            
        return float(max(0.0, min(100.0, score)))

    def _score_orbit_feasibility(self, f: Dict[str, float]) -> float:
        """Evaluate orbital parameters for natural celestial mechanics realism."""
        score = 90.0
        a = f["semi_major_axis_au"]
        e = f["eccentricity"]
        tj = f["tisserand_jupiter"]
        
        # Unrealistic low semi-major axis (e.g. within Earth's atmosphere / LEO)
        if a < 0.2:
            return 10.0
            
        # Parabolic/extreme hyperbolic without comet designation
        if e > 1.2:
            score -= 40.0
        elif e > 0.9:
            score -= 15.0
            
        # Tisserand parameter check
        if tj is not None:
            if tj >= 3.0:
                score += 10.0  # Classic asteroidal orbit
            elif 2.0 <= tj < 3.0:
                score += 5.0   # Jupiter-family comet type orbit
            elif tj < 1.5:
                score -= 20.0  # Highly anomalous or unbound
                
        return float(max(0.0, min(100.0, score)))

    def _score_observational_reliability(self, f: Dict[str, float]) -> float:
        """Evaluate observational cadence, number of data points, and arc coverage."""
        score = 50.0
        n_obs = f["n_observations"]
        arc_hours = f["arc_length_hours"]
        elong = f["solar_elongation_deg"]
        
        # Data points
        if n_obs >= 6:
            score += 25.0
        elif n_obs >= 4:
            score += 15.0
        elif n_obs == 3:
            score += 5.0
        else:
            score -= 20.0
            
        # Arc length
        if arc_hours >= 12.0:
            score += 25.0
        elif arc_hours >= 2.0:
            score += 15.0
        elif arc_hours >= 0.5:
            score += 5.0
        else:
            score -= 15.0
            
        # Twilight penalty: low solar elongation brings sky noise and diff-spike artifacts
        if elong < 55.0:
            score -= (55.0 - elong) * 0.8
            
        return float(max(0.0, min(100.0, score)))

    def _score_satellite_veto(self, f: Dict[str, float]) -> float:
        """High score means object is safe / natural; low score means probable satellite / space debris."""
        score = 100.0
        rate = f["motion_rate_deg_day"]
        is_extreme = f["is_extreme_speed"]
        is_leo = f["is_leo_orbit_band"]
        
        # Extreme apparent speed check (> 30 deg/day)
        if is_extreme:
            return 5.0
            
        if rate > 20.0:
            score -= (rate - 20.0) * 6.0
        elif rate > 10.0:
            score -= (rate - 10.0) * 2.0
            
        if is_leo:
            score -= 70.0
            
        # Stationary check: < 0.05 deg/day is not moving (likely variable star or bad pixel)
        if f["is_stationary"]:
            score -= 60.0
            
        return float(max(0.0, min(100.0, score)))

    def _score_photometric_stability(self, f: Dict[str, float]) -> float:
        """Evaluate photometric realism."""
        score = 85.0
        v_mag = f["v_mag"]
        h_mag = f["h_mag"]
        
        # Plausible survey detection magnitudes
        if 16.0 <= v_mag <= 22.5:
            score += 15.0
        elif v_mag < 14.0:
            # Suspiciously bright for a new NEO (unless very close flyby)
            score -= 20.0
        elif v_mag > 23.5:
            score -= 15.0
            
        # Physical diameter equivalent check (H)
        if h_mag > 30.0:
            # H > 30 implies diameter < 2 meters, typical of small artificial debris
            score -= 25.0
            
        return float(max(0.0, min(100.0, score)))
