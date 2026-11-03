"""Pydantic data models for NEO and transient sky survey candidates."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class DetectionStatus(str, Enum):
    VALIDATED_NEO = "VALIDATED_NEO"
    POTENTIALLY_HAZARDOUS = "POTENTIALLY_HAZARDOUS"
    PROBABLE_ASTEROID = "PROBABLE_ASTEROID"
    AMBIGUOUS_FOLLOW_UP_REQUIRED = "AMBIGUOUS_FOLLOW_UP_REQUIRED"
    BOGUS_FALSE_POSITIVE = "BOGUS_FALSE_POSITIVE"
    REJECTED_SPACE_DEBRIS = "REJECTED_SPACE_DEBRIS"


class FalsePositiveMode(str, Enum):
    NONE_REAL_CANDIDATE = "NONE_REAL_CANDIDATE"
    DIPOLE_SUBTRACTION_ARTIFACT = "DIPOLE_SUBTRACTION_ARTIFACT"
    BRIGHT_STAR_DIFFRACTION_GHOST = "BRIGHT_STAR_DIFFRACTION_GHOST"
    ARTIFICIAL_SATELLITE_OR_DEBRIS = "ARTIFICIAL_SATELLITE_OR_DEBRIS"
    SPURIOUS_NOISE_LINKAGE = "SPURIOUS_NOISE_LINKAGE"
    STATIONARY_VARIABLE_STAR = "STATIONARY_VARIABLE_STAR"
    COSMIC_RAY_STRIKE = "COSMIC_RAY_STRIKE"


class OrbitalElements(BaseModel):
    semi_major_axis_au: Optional[float] = Field(None, description="Semi-major axis (a) in AU")
    eccentricity: Optional[float] = Field(None, description="Orbital eccentricity (e)")
    inclination_deg: Optional[float] = Field(None, description="Orbital inclination (i) in degrees")
    perihelion_dist_au: Optional[float] = Field(None, description="Perihelion distance (q) in AU")
    aphelion_dist_au: Optional[float] = Field(None, description="Aphelion distance (Q) in AU")
    longitude_node_deg: Optional[float] = Field(None, description="Longitude of ascending node in degrees")
    arg_perihelion_deg: Optional[float] = Field(None, description="Argument of perihelion in degrees")
    mean_anomaly_deg: Optional[float] = Field(None, description="Mean anomaly in degrees")
    epoch_jd: Optional[float] = Field(None, description="Epoch in Julian Date")
    tisserand_jupiter: Optional[float] = Field(None, description="Tisserand parameter with respect to Jupiter (T_J)")
    moid_au: Optional[float] = Field(None, description="Minimum Orbit Intersection Distance with Earth in AU")
    neo_family: Optional[str] = Field(None, description="NEO family: Apollo, Amor, Aten, Atira, or Main Belt")


class AstrometricObservation(BaseModel):
    obs_time_utc: datetime
    ra_deg: float
    dec_deg: float
    mag: Optional[float] = None
    band: Optional[str] = "V"
    observatory_code: Optional[str] = None
    residual_ra_arcsec: Optional[float] = 0.0
    residual_dec_arcsec: Optional[float] = 0.0


class HeuristicBreakdown(BaseModel):
    astrometric_consistency_score: float = Field(..., ge=0.0, le=100.0)
    orbit_feasibility_score: float = Field(..., ge=0.0, le=100.0)
    observational_reliability_score: float = Field(..., ge=0.0, le=100.0)
    satellite_veto_score: float = Field(..., ge=0.0, le=100.0)
    photometric_stability_score: float = Field(..., ge=0.0, le=100.0)
    composite_ai_score: float = Field(..., ge=0.0, le=100.0)


class FalsePositiveDiagnostic(BaseModel):
    mode: FalsePositiveMode
    bogus_probability: float = Field(..., ge=0.0, le=1.0)
    primary_triggers: List[str]
    explanation: str
    rejection_confidence: float = Field(..., ge=0.0, le=100.0)


class CandidateRecord(BaseModel):
    candidate_id: str
    source_survey: str  # e.g. "JPL_SCOUT", "MPC_NEOCP", "FINK_ZTF", "CATALINA", "ATLAS"
    discovery_date: datetime
    apparent_ra_deg: float
    apparent_dec_deg: float
    v_mag: Optional[float] = None
    h_mag: Optional[float] = None
    motion_rate_deg_day: float  # Apparent rate of motion in deg/day
    solar_elongation_deg: Optional[float] = None
    arc_length_hours: float
    n_observations: int
    rms_residual_arcsec: Optional[float] = 0.25
    orbital_elements: Optional[OrbitalElements] = None
    observations: List[AstrometricObservation] = []
    raw_metadata: Dict[str, Any] = {}


class CandidateValidationResult(BaseModel):
    candidate_id: str
    source: str
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    status: DetectionStatus
    composite_confidence_score: float = Field(..., ge=0.0, le=100.0)
    heuristic_breakdown: HeuristicBreakdown
    false_positive_diagnostic: FalsePositiveDiagnostic
    ml_real_probability: float = Field(..., ge=0.0, le=1.0)
    is_potentially_hazardous: bool = False
    orbit_classification: Optional[str] = None
    recommendations: List[str] = []


class RevalidationReport(BaseModel):
    candidate_id: str
    original_discovery_date: datetime
    revalidation_epoch: datetime = Field(default_factory=datetime.utcnow)
    age_months: float
    baseline_orbit_quality: str
    ephemeris_drift_arcsec: float
    recovery_probability: float = Field(..., ge=0.0, le=1.0)
    final_disposition: str  # "CONFIRMED_PERMANENT", "REJECTED_SPURIOUS", "LOST_SHORT_ARC", "DESIGNATED"
    historical_notes: str
