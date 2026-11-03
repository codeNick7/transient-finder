"""Tracklet polynomial fitting, astrometric residual analysis, and kinematic feasibility."""

import math
from typing import List, Dict, Any, Tuple
import numpy as np
from core.models import AstrometricObservation
from core.astronomy import datetime_to_jd, great_circle_distance_deg


def fit_tracklet_kinematics(observations: List[AstrometricObservation]) -> Dict[str, Any]:
    """Fit 1st and 2nd degree polynomials to RA and Dec over time to evaluate astrometric consistency,
    residual RMS, velocity, and topocentric acceleration.
    """
    if len(observations) < 2:
        return {
            "n_points": len(observations),
            "rms_arcsec": 0.0,
            "motion_rate_deg_day": 0.0,
            "acceleration_deg_day2": 0.0,
            "linearity_r2": 1.0,
            "time_span_hours": 0.0
        }
    
    # Sort observations by time
    sorted_obs = sorted(observations, key=lambda o: o.obs_time_utc)
    t0_jd = datetime_to_jd(sorted_obs[0].obs_time_utc)
    
    times_days = np.array([datetime_to_jd(o.obs_time_utc) - t0_jd for o in sorted_obs])
    ras_deg = np.array([o.ra_deg for o in sorted_obs])
    decs_deg = np.array([o.dec_deg for o in sorted_obs])
    
    # Adjust RA for 360 wrap-around if traversing 0h
    if np.max(ras_deg) - np.min(ras_deg) > 180.0:
        ras_deg = np.where(ras_deg > 180.0, ras_deg - 360.0, ras_deg)
        
    time_span_hours = float((times_days[-1] - times_days[0]) * 24.0)
    
    # Linear fit
    if len(observations) == 2 or time_span_hours < 0.001:
        dt_days = max(1e-6, times_days[-1] - times_days[0])
        dist_deg = great_circle_distance_deg(ras_deg[0], decs_deg[0], ras_deg[-1], decs_deg[-1])
        rate_deg_day = dist_deg / dt_days
        return {
            "n_points": len(observations),
            "rms_arcsec": 0.15,
            "motion_rate_deg_day": float(rate_deg_day),
            "acceleration_deg_day2": 0.0,
            "linearity_r2": 1.0,
            "time_span_hours": float(time_span_hours)
        }
    
    # Degree 1 fit (velocity)
    p_ra = np.polyfit(times_days, ras_deg, deg=1)
    p_dec = np.polyfit(times_days, decs_deg, deg=1)
    
    pred_ra = np.polyval(p_ra, times_days)
    pred_dec = np.polyval(p_dec, times_days)
    
    # Cosine Dec factor for RA residuals
    mean_dec_rad = math.radians(float(np.mean(decs_deg)))
    ra_res_arcsec = (ras_deg - pred_ra) * math.cos(mean_dec_rad) * 3600.0
    dec_res_arcsec = (decs_deg - pred_dec) * 3600.0
    
    combined_res = np.sqrt(ra_res_arcsec ** 2 + dec_res_arcsec ** 2)
    rms_arcsec = float(np.sqrt(np.mean(combined_res ** 2)))
    
    # Apparent rate of motion (deg/day)
    v_ra = p_ra[0] * math.cos(mean_dec_rad)
    v_dec = p_dec[0]
    motion_rate_deg_day = float(math.sqrt(v_ra ** 2 + v_dec ** 2))
    
    # Acceleration check if 4+ points
    accel_deg_day2 = 0.0
    if len(observations) >= 4 and time_span_hours > 0.1:
        p2_ra = np.polyfit(times_days, ras_deg, deg=2)
        p2_dec = np.polyfit(times_days, decs_deg, deg=2)
        # Second derivative is 2 * a2
        a_ra = 2.0 * p2_ra[0] * math.cos(mean_dec_rad)
        a_dec = 2.0 * p2_dec[0]
        accel_deg_day2 = float(math.sqrt(a_ra ** 2 + a_dec ** 2))
    
    # Linearity R^2 comparing residual variance to total traversed angular motion
    total_motion_arcsec = great_circle_distance_deg(ras_deg[0], decs_deg[0], ras_deg[-1], decs_deg[-1]) * 3600.0
    ss_tot = max(1.0, total_motion_arcsec ** 2)
    ss_res = float(np.sum(ra_res_arcsec ** 2 + dec_res_arcsec ** 2))
    linearity_r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))

    return {
        "n_points": len(observations),
        "rms_arcsec": float(rms_arcsec),
        "motion_rate_deg_day": float(motion_rate_deg_day),
        "acceleration_deg_day2": float(accel_deg_day2),
        "linearity_r2": float(linearity_r2),
        "time_span_hours": float(time_span_hours)
    }


def estimate_apparent_ephemeris_drift(arc_hours: float, n_obs: int, initial_rms: float, elapsed_months: float) -> float:
    """Model the expected ephemeris uncertainty growth (drift) in arcseconds after elapsed_months.
    
    Short-arc discoveries (< 1 day) without follow-up undergo quadratic/cubic covariance growth over time.
    """
    if arc_hours <= 0:
        return 9999.0
        
    arc_days = arc_hours / 24.0
    # Observational arc leverage factor
    base_uncertainty = max(0.1, initial_rms) / math.sqrt(max(1, n_obs))
    
    # Uncertainty scales rapidly as (t / arc)^2 for unconstrained orbit
    time_ratio = (elapsed_months * 30.4) / max(0.05, arc_days)
    
    if arc_days < 0.2:  # very short tracklet (< 5 hours)
        drift_arcsec = base_uncertainty * (time_ratio ** 2.2) * 10.0
    elif arc_days < 1.0:  # 1 night
        drift_arcsec = base_uncertainty * (time_ratio ** 1.8) * 3.0
    elif arc_days < 7.0:  # 1 week arc
        drift_arcsec = base_uncertainty * (time_ratio ** 1.3) * 0.5
    else:  # multi-week arc
        drift_arcsec = base_uncertainty * (time_ratio ** 0.8) * 0.1
        
    return float(min(1e6, drift_arcsec))
