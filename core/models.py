"""Pydantic data models for NEO and transient sky survey candidates."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class DetectionStatus(str, Enum):
    VALIDATED_NEO = "VALIDATED_NEO"
    IDENTIFIED_NEO = "IDENTIFIED_NEO"
    POTENTIALLY_HAZARDOUS = "POTENTIALLY_HAZARDOUS"
    PROBABLE_ASTEROID = "PROBABLE_ASTEROID"
    DISTANT_TRANSIENT = "DISTANT_TRANSIENT"
    AMBIGUOUS_FOLLOW_UP_REQUIRED = "AMBIGUOUS_FOLLOW_UP_REQUIRED"
    BOGUS_FALSE_POSITIVE = "BOGUS_FALSE_POSITIVE"
    REJECTED_SPACE_DEBRIS = "REJECTED_SPACE_DEBRIS"


class TransientCategory(str, Enum):
    IDENTIFIED_NEO = "IDENTIFIED_NEO"
    NEO_CANDIDATE = "NEO_CANDIDATE"
    MAIN_BELT_ASTEROID = "MAIN_BELT_ASTEROID"
    DISTANT_SUPERNOVA = "DISTANT_SUPERNOVA"
    DISTANT_AGN_FLARE = "DISTANT_AGN_FLARE"
    DISTANT_KILONOVA_OR_GRB = "DISTANT_KILONOVA_OR_GRB"
    CATACLYSMIC_VARIABLE = "CATACLYSMIC_VARIABLE"
    ARTIFICIAL_SATELLITE_OR_DEBRIS = "ARTIFICIAL_SATELLITE_OR_DEBRIS"
    IMAGE_ARTIFACT_OR_BOGUS = "IMAGE_ARTIFACT_OR_BOGUS"


class DistantTransientProperties(BaseModel):
    transient_type: Optional[str] = Field(None, description="Astrophysical class (e.g. SN Ia, SN II, AGN_FLARE, KILONOVA)")
    photometric_amplitude_mag: Optional[float] = Field(None, description="Brightening amplitude Delta-mag from quiescent state")
    decay_rate_mag_day: Optional[float] = Field(None, description="Apparent photometric decay or fading rate in mag/day")
    estimated_redshift_z: Optional[float] = Field(None, description="Estimated cosmological redshift z of host galaxy")
    host_galaxy_name: Optional[str] = Field(None, description="Associated host galaxy designation if resolved")
    host_galaxy_separation_arcsec: Optional[float] = Field(None, description="Angular offset from host galactic nucleus in arcsec")
    stationary_astrometric_sigma: Optional[float] = Field(0.05, description="Astrometric positional dispersion across epochs in arcsec")


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
    source_survey: str  # e.g. "JPL_SCOUT", "MPC_NEOCP", "FINK_ZTF", "NASA_JPL_HORIZONS", "CATALINA", "ATLAS"
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
    target_designation: Optional[str] = None
    is_known_object: bool = False
    transient_properties: Optional[DistantTransientProperties] = None
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
    transient_category: Optional[TransientCategory] = None
    transient_properties: Optional[DistantTransientProperties] = None
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


class SkyPatchQuery(BaseModel):
    ra_deg: float = Field(..., ge=0.0, lt=360.0, description="Central Right Ascension in degrees (0 to 360)")
    dec_deg: float = Field(..., ge=-90.0, le=90.0, description="Central Declination in degrees (-90 to +90)")
    radius_deg: float = Field(default=1.0, gt=0.0, le=15.0, description="Search cone radius in degrees (e.g. 0.5 to 5.0 deg)")
    target_id: Optional[str] = Field(None, description="Optional target identifier for ephemeris cross-match (e.g. 1998 KY26)")
    epoch: Optional[str] = Field(None, description="Observation epoch or date (e.g. 2024-06-01 or ISO format)")
    sources: List[str] = Field(default=["JPL_HORIZONS", "JPL_SCOUT", "MPC_NEOCP", "FINK_BROKER"], description="Observational data sources to query")


class SkyPatchObject(BaseModel):
    candidate: CandidateRecord
    angular_separation_deg: float = Field(..., description="Angular distance from search cone center in degrees")
    angular_separation_arcmin: float = Field(..., description="Angular distance from search cone center in arcminutes")
    validation: CandidateValidationResult
    category: TransientCategory


class SkyPatchResult(BaseModel):
    query: SkyPatchQuery
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    total_detected: int = 0
    neo_count: int = 0
    distant_transient_count: int = 0
    asteroid_count: int = 0
    artifact_count: int = 0
    objects: List[SkyPatchObject] = []
