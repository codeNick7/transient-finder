"""Unit tests for historical 6-month candidate re-validation."""

from datetime import datetime, timedelta
from ingestion.fixtures import get_ground_truth_fixtures
from ai.revalidation_engine import CandidateRevalidationEngine


def test_revalidation_historical_2026_hx3():
    engine = CandidateRevalidationEngine()
    fixtures = get_ground_truth_fixtures()
    hx3 = next(f for f in fixtures if f.candidate_id == "2026 HX3")
    
    rep = engine.revalidate_historical_candidate(hx3, current_time=datetime(2026, 11, 3))
    assert abs(rep.age_months - 6.0) < 0.5
    assert rep.final_disposition == "CONFIRMED_PERMANENT"
    assert rep.recovery_probability > 0.80


def test_revalidation_short_arc_drift():
    engine = CandidateRevalidationEngine()
    fixtures = get_ground_truth_fixtures()
    dipole = next(f for f in fixtures if f.candidate_id == "BOGUS_DIPOLE_001")
    
    rep = engine.revalidate_historical_candidate(dipole, current_time=datetime(2026, 11, 3))
    assert rep.final_disposition == "REJECTED_SPURIOUS"
