"""Unit tests for feature extraction and tracklet polynomial fitting."""

from datetime import datetime, timedelta
from core.models import CandidateRecord, AstrometricObservation
from core.orbit_evaluator import fit_tracklet_kinematics
from ai.feature_extractor import extract_candidate_features, features_to_vector, FEATURE_NAMES


def test_fit_tracklet_kinematics_linear():
    t0 = datetime(2026, 11, 3, 0, 0, 0)
    obs = [
        AstrometricObservation(obs_time_utc=t0, ra_deg=10.0, dec_deg=5.0),
        AstrometricObservation(obs_time_utc=t0 + timedelta(hours=1), ra_deg=10.1, dec_deg=5.05),
        AstrometricObservation(obs_time_utc=t0 + timedelta(hours=2), ra_deg=10.2, dec_deg=5.10),
    ]
    res = fit_tracklet_kinematics(obs)
    assert res["n_points"] == 3
    assert res["rms_arcsec"] < 0.20
    assert res["motion_rate_deg_day"] > 1.0
    assert res["linearity_r2"] > 0.99


def test_extract_candidate_features():
    c = CandidateRecord(
        candidate_id="TEST_001",
        source_survey="TEST_SURVEY",
        discovery_date=datetime(2026, 11, 3),
        apparent_ra_deg=45.0,
        apparent_dec_deg=12.0,
        motion_rate_deg_day=2.5,
        arc_length_hours=4.0,
        n_observations=5,
        rms_residual_arcsec=0.22,
        solar_elongation_deg=140.0
    )
    feats = extract_candidate_features(c)
    assert len(feats) == len(FEATURE_NAMES)
    assert feats["motion_rate_deg_day"] == 2.5
    assert feats["rms_residual_arcsec"] == 0.22
    assert feats["is_twilight_zone"] == 0.0

    vec = features_to_vector(feats)
    assert vec.shape == (len(FEATURE_NAMES),)
