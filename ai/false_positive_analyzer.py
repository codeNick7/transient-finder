"""Root-cause diagnostic false positive analyzer for sky survey transients and NEO candidates."""

from typing import List, Dict, Any
from core.models import CandidateRecord, FalsePositiveMode, FalsePositiveDiagnostic
from ai.feature_extractor import extract_candidate_features


class FalsePositiveAnalyzer:
    """Classifies root causes of candidate rejections to prevent false alarms and operator fatigue."""

    def diagnose(self, candidate: CandidateRecord, ml_real_prob: float, heuristic_score: float) -> FalsePositiveDiagnostic:
        feats = extract_candidate_features(candidate)
        triggers: List[str] = []
        mode = FalsePositiveMode.NONE_REAL_CANDIDATE
        
        # 1. Artificial Satellite / Space Debris Check
        if feats["is_extreme_speed"] or feats["motion_rate_deg_day"] > 30.0:
            triggers.append(f"Apparent motion rate ({feats['motion_rate_deg_day']:.2f} deg/day) exceeds maximum natural asteroid limit (30 deg/day)")
            mode = FalsePositiveMode.ARTIFICIAL_SATELLITE_OR_DEBRIS
        elif feats["is_leo_orbit_band"] or feats["semi_major_axis_au"] < 0.08:
            triggers.append(f"Orbital elements match Low Earth Orbit (LEO) debris band (a={feats['semi_major_axis_au']:.3f} AU, inc={feats['inclination_deg']:.1f} deg)")
            mode = FalsePositiveMode.ARTIFICIAL_SATELLITE_OR_DEBRIS
            
        # 2. Stationary / Variable Star Check
        elif feats["is_stationary"] or feats["motion_rate_deg_day"] < 0.04:
            triggers.append(f"Motion rate ({feats['motion_rate_deg_day']:.4f} deg/day) indicates stationary source, consistent with stellar flare or uncatalogued variable star")
            mode = FalsePositiveMode.STATIONARY_VARIABLE_STAR
            
        # 3. Spurious Tracklet Linkage Check
        elif feats["acceleration_deg_day2"] > 0.6 or feats["linearity_r2"] < 0.93:
            triggers.append(f"Apparent topocentric acceleration ({feats['acceleration_deg_day2']:.2f} deg/day^2) or linearity R^2 ({feats['linearity_r2']:.3f}) violates Keplerian tracklet smooth arc")
            mode = FalsePositiveMode.SPURIOUS_NOISE_LINKAGE
            
        # 4. Image Subtraction Dipole Artifact Check
        elif feats["rms_residual_arcsec"] > 0.85:
            triggers.append(f"Astrometric RMS residual ({feats['rms_residual_arcsec']:.2f} arcsec) exceeds optical PSF limit, indicative of difference image dipole misregistration")
            mode = FalsePositiveMode.DIPOLE_SUBTRACTION_ARTIFACT
            
        # 5. Bright Star Diffraction Spike / Twilight Glare Check
        elif feats["is_twilight_zone"] or feats["solar_elongation_deg"] < 55.0:
            if feats["rms_residual_arcsec"] > 0.55 or heuristic_score < 60.0:
                triggers.append(f"Low solar elongation ({feats['solar_elongation_deg']:.1f} deg) places candidate in twilight glare / diffraction ghost contamination region")
                mode = FalsePositiveMode.BRIGHT_STAR_DIFFRACTION_GHOST

        # Calculate bogus probability
        if mode == FalsePositiveMode.NONE_REAL_CANDIDATE:
            bogus_prob = max(0.0, 1.0 - ml_real_prob)
            if heuristic_score > 75.0 and bogus_prob < 0.25:
                explanation = "Candidate exhibits consistent astrometry, realistic Keplerian motion, and healthy observational coverage. No false-positive signatures detected."
                rejection_conf = 0.0
            else:
                explanation = "Candidate passed primary artifact filters, but requires additional observational follow-up to reduce ephemeris uncertainty."
                rejection_conf = float(bogus_prob * 100.0)
        else:
            bogus_prob = max(0.70, 1.0 - ml_real_prob)
            rejection_conf = float(min(99.5, max(75.0, (1.0 - heuristic_score / 100.0) * 100.0)))
            explanation = f"High-confidence false positive detected ({mode.value}). Triggers: " + "; ".join(triggers)

        return FalsePositiveDiagnostic(
            mode=mode,
            bogus_probability=float(bogus_prob),
            primary_triggers=triggers,
            explanation=explanation,
            rejection_confidence=rejection_conf
        )
