"""Core models and astronomical math functions."""

from core.models import (
    DetectionStatus,
    FalsePositiveMode,
    OrbitalElements,
    AstrometricObservation,
    HeuristicBreakdown,
    FalsePositiveDiagnostic,
    CandidateRecord,
    CandidateValidationResult,
    RevalidationReport
)

from core.astronomy import (
    datetime_to_jd,
    great_circle_distance_deg,
    calculate_solar_elongation_deg,
    calculate_tisserand_parameter,
    classify_neo_orbit,
    check_potentially_hazardous
)

from core.orbit_evaluator import (
    fit_tracklet_kinematics,
    estimate_apparent_ephemeris_drift
)

__all__ = [
    "DetectionStatus",
    "FalsePositiveMode",
    "OrbitalElements",
    "AstrometricObservation",
    "HeuristicBreakdown",
    "FalsePositiveDiagnostic",
    "CandidateRecord",
    "CandidateValidationResult",
    "RevalidationReport",
    "datetime_to_jd",
    "great_circle_distance_deg",
    "calculate_solar_elongation_deg",
    "calculate_tisserand_parameter",
    "classify_neo_orbit",
    "check_potentially_hazardous",
    "fit_tracklet_kinematics",
    "estimate_apparent_ephemeris_drift",
]
