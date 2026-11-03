"""Unit tests for multi-tier heuristic scoring."""

from datetime import datetime
from core.models import CandidateRecord, OrbitalElements
from ai.heuristic_engine import HeuristicScoringEngine


def test_heuristic_scoring_ideal_neo():
    engine = HeuristicScoringEngine()
    c = CandidateRecord(
        candidate_id="IDEAL_NEO",
        source_survey="CSS",
        discovery_date=datetime(2026, 11, 3),
        apparent_ra_deg=30.0,
        apparent_dec_deg=10.0,
        motion_rate_deg_day=2.1,
        solar_elongation_deg=155.0,
        arc_length_hours=12.0,
        n_observations=8,
        rms_residual_arcsec=0.18,
        v_mag=19.5,
        h_mag=21.0,
        orbital_elements=OrbitalElements(
            semi_major_axis_au=1.4,
            eccentricity=0.35,
            inclination_deg=8.0,
            tisserand_jupiter=4.6,
            moid_au=0.03
        )
    )
    b = engine.evaluate_candidate(c)
    assert b.astrometric_consistency_score >= 85.0
    assert b.orbit_feasibility_score >= 85.0
    assert b.satellite_veto_score >= 90.0
    assert b.composite_ai_score >= 80.0


def test_heuristic_scoring_satellite_debris():
    engine = HeuristicScoringEngine()
    c = CandidateRecord(
        candidate_id="SATELLITE_LEO",
        source_survey="SURVEY",
        discovery_date=datetime(2026, 11, 3),
        apparent_ra_deg=30.0,
        apparent_dec_deg=10.0,
        motion_rate_deg_day=45.0,  # Extreme speed
        solar_elongation_deg=90.0,
        arc_length_hours=0.2,
        n_observations=4,
        rms_residual_arcsec=0.30,
        v_mag=14.0,
        orbital_elements=OrbitalElements(
            semi_major_axis_au=0.005,
            eccentricity=0.01,
            inclination_deg=53.0
        )
    )
    b = engine.evaluate_candidate(c)
    assert b.satellite_veto_score <= 10.0
    assert b.composite_ai_score < 45.0
