"""Unit tests for ML Real-Bogus classifier and false positive diagnostics."""

from core.models import FalsePositiveMode
from ingestion.fixtures import get_ground_truth_fixtures
from ai.validator import CandidateValidator


def test_validator_ground_truth_real_neo():
    val = CandidateValidator()
    fixtures = get_ground_truth_fixtures()
    real_apollo = next(f for f in fixtures if f.candidate_id == "NEO_APOLLO_REAL_01")
    
    res = val.validate(real_apollo)
    assert res.status.value == "VALIDATED_NEO"
    assert res.composite_confidence_score >= 75.0
    assert res.ml_real_probability >= 0.70
    assert res.false_positive_diagnostic.mode == FalsePositiveMode.NONE_REAL_CANDIDATE


def test_validator_potentially_hazardous():
    val = CandidateValidator()
    fixtures = get_ground_truth_fixtures()
    pha = next(f for f in fixtures if f.candidate_id == "NEO_PHA_HIGH_RISK_02")
    
    res = val.validate(pha)
    assert res.is_potentially_hazardous is True
    assert res.status.value == "POTENTIALLY_HAZARDOUS"


def test_validator_false_positive_satellite():
    val = CandidateValidator()
    fixtures = get_ground_truth_fixtures()
    sat = next(f for f in fixtures if f.candidate_id == "BOGUS_SATELLITE_LEO_002")
    
    res = val.validate(sat)
    assert res.status.value == "REJECTED_SPACE_DEBRIS"
    assert res.false_positive_diagnostic.mode == FalsePositiveMode.ARTIFICIAL_SATELLITE_OR_DEBRIS
    assert res.false_positive_diagnostic.rejection_confidence >= 75.0


def test_validator_false_positive_dipole():
    val = CandidateValidator()
    fixtures = get_ground_truth_fixtures()
    dipole = next(f for f in fixtures if f.candidate_id == "BOGUS_DIPOLE_001")
    
    res = val.validate(dipole)
    assert res.status.value == "BOGUS_FALSE_POSITIVE"
    assert res.false_positive_diagnostic.mode == FalsePositiveMode.DIPOLE_SUBTRACTION_ARTIFACT


def test_validator_stationary_variable_star():
    val = CandidateValidator()
    fixtures = get_ground_truth_fixtures()
    var_star = next(f for f in fixtures if f.candidate_id == "BOGUS_VAR_STAR_005")
    
    res = val.validate(var_star)
    assert res.false_positive_diagnostic.mode == FalsePositiveMode.STATIONARY_VARIABLE_STAR
