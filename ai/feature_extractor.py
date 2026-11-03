"""Feature engineering and representation extraction for ML and heuristic scoring."""

import math
from typing import Dict, Any, List
import numpy as np

from core.models import CandidateRecord
from core.orbit_evaluator import fit_tracklet_kinematics


FEATURE_NAMES = [
    "motion_rate_deg_day",
    "log_motion_rate",
    "arc_length_hours",
    "n_observations",
    "rms_residual_arcsec",
    "solar_elongation_deg",
    "v_mag",
    "h_mag",
    "acceleration_deg_day2",
    "linearity_r2",
    "semi_major_axis_au",
    "eccentricity",
    "inclination_deg",
    "tisserand_jupiter",
    "moid_au",
    "is_twilight_zone",
    "is_extreme_speed",
    "is_stationary",
    "is_leo_orbit_band",
    "cadence_hours_per_obs"
]


def extract_candidate_features(candidate: CandidateRecord) -> Dict[str, float]:
    """Extract 20 structured physical, astrometric, and observing geometry features from a CandidateRecord."""
    # Tracklet kinematics
    if candidate.observations and len(candidate.observations) >= 2:
        kin = fit_tracklet_kinematics(candidate.observations)
        rms = kin["rms_arcsec"]
        rate = kin["motion_rate_deg_day"]
        accel = kin["acceleration_deg_day2"]
        linearity = kin["linearity_r2"]
    else:
        rms = candidate.rms_residual_arcsec or 0.25
        rate = candidate.motion_rate_deg_day
        accel = 0.0
        linearity = 0.98

    arc_hours = max(0.01, candidate.arc_length_hours)
    n_obs = max(1, candidate.n_observations)
    cadence = arc_hours / n_obs
    elong = candidate.solar_elongation_deg or 130.0
    v_mag = candidate.v_mag or 20.0
    h_mag = candidate.h_mag or 22.0

    # Orbital elements
    elems = candidate.orbital_elements
    a = elems.semi_major_axis_au if elems and elems.semi_major_axis_au else 1.8
    e = elems.eccentricity if elems and elems.eccentricity is not None else 0.35
    inc = elems.inclination_deg if elems and elems.inclination_deg is not None else 10.0
    tj = elems.tisserand_jupiter if elems and elems.tisserand_jupiter is not None else 3.8
    moid = elems.moid_au if elems and elems.moid_au is not None else 0.08

    # Binary flags
    is_twilight = 1.0 if elong < 55.0 else 0.0
    is_extreme = 1.0 if rate > 30.0 else 0.0
    is_stationary = 1.0 if rate < 0.05 else 0.0
    # Satellites often have high inclinations (e.g. Starlink ~53 deg) and extreme speeds or LEO-like semi-major axis
    is_leo = 1.0 if (rate > 25.0 and (abs(inc - 53.0) < 5.0 or a < 0.1)) else 0.0

    log_rate = float(np.log10(max(1e-4, rate)))

    return {
        "motion_rate_deg_day": float(rate),
        "log_motion_rate": float(log_rate),
        "arc_length_hours": float(arc_hours),
        "n_observations": float(n_obs),
        "rms_residual_arcsec": float(rms),
        "solar_elongation_deg": float(elong),
        "v_mag": float(v_mag),
        "h_mag": float(h_mag),
        "acceleration_deg_day2": float(accel),
        "linearity_r2": float(linearity),
        "semi_major_axis_au": float(a),
        "eccentricity": float(e),
        "inclination_deg": float(inc),
        "tisserand_jupiter": float(tj),
        "moid_au": float(moid),
        "is_twilight_zone": float(is_twilight),
        "is_extreme_speed": float(is_extreme),
        "is_stationary": float(is_stationary),
        "is_leo_orbit_band": float(is_leo),
        "cadence_hours_per_obs": float(cadence)
    }


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dictionary to ordered numpy 1D vector."""
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
