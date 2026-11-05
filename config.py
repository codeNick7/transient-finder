"""Configuration settings and threshold parameters for the Transient-Finder NEO AI System."""

from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
FIXTURES_DIR = DATA_DIR / "fixtures"

for d in [DATA_DIR, CACHE_DIR, FIXTURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class ValidationThresholds(BaseModel):
    # Astrometry & Kinematics
    max_acceptable_rms_arcsec: float = 0.80  # Arcsec residual limit
    suspicious_rms_arcsec: float = 0.50
    min_observations_for_confirmation: int = 3
    min_arc_length_hours: float = 0.5
    
    # Motion rate thresholds (deg/day)
    min_neo_motion_deg_day: float = 0.50
    max_physical_natural_motion_deg_day: float = 30.0  # Speeds > 30 deg/day are almost always LEO/satellite
    
    # Solar elongation (deg)
    min_safe_solar_elongation_deg: float = 55.0
    
    # Orbital mechanics
    jupiter_semi_major_axis_au: float = 5.2044
    pha_max_moid_au: float = 0.05
    pha_max_h_mag: float = 22.0
    
    # AI Heuristic Weights (Sum = 1.0)
    weight_astrometric_consistency: float = 0.30
    weight_orbit_feasibility: float = 0.25
    weight_observational_reliability: float = 0.20
    weight_satellite_veto: float = 0.15
    weight_photometric_stability: float = 0.10
    
    # Decision thresholds
    confirmed_neo_threshold: float = 75.0
    candidate_review_threshold: float = 50.0
    bogus_reject_threshold: float = 40.0


class SystemConfig(BaseModel):
    app_name: str = "Transient-Finder AI"
    version: str = "1.0.0"
    author: str = "Advanced Astronomical Agentic Systems"
    
    # Endpoints
    jpl_scout_url: str = "https://ssd-api.jpl.nasa.gov/scout.api"
    jpl_cad_url: str = "https://ssd-api.jpl.nasa.gov/cad.api"
    jpl_sbdb_url: str = "https://ssd-api.jpl.nasa.gov/sbdb.api"
    mpc_neocp_url: str = "https://www.minorplanetcenter.net/iau/NEO/neocp.txt"
    fink_sso_cand_url: str = "https://api.ztf.fink-portal.org/api/v1/ssocand"
    
    # Networking
    request_timeout_seconds: int = 15
    user_agent: str = "TransientFinder/1.0 (Public Research Pipeline; contact: research@transient-finder.org)"
    cache_ttl_seconds: int = 3600  # 60 minutes
    
    # Thresholds
    thresholds: ValidationThresholds = ValidationThresholds()


CONFIG = SystemConfig()
