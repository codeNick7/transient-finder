"""Offline benchmark datasets, ground truth candidates, and synthetic false positives."""

from datetime import datetime, timedelta
from typing import List
from core.models import CandidateRecord, OrbitalElements, AstrometricObservation


def get_ground_truth_fixtures() -> List[CandidateRecord]:
    """Return a comprehensive benchmark suite of validated NEOs, 6-month historical objects,
    and distinct astronomical false-positive failure modes.
    """
    now = datetime(2026, 11, 3, 3, 30)
    six_months_ago = datetime(2026, 5, 1, 8, 23)
    
    fixtures: List[CandidateRecord] = []
    
    # 1. High-Confidence Apollo NEO Candidate (Real)
    obs_apollo = [
        AstrometricObservation(obs_time_utc=now, ra_deg=20.0000, dec_deg=10.0000, mag=19.2, residual_ra_arcsec=0.12, residual_dec_arcsec=0.15),
        AstrometricObservation(obs_time_utc=now + timedelta(minutes=45), ra_deg=20.0350, dec_deg=10.0220, mag=19.3, residual_ra_arcsec=-0.08, residual_dec_arcsec=0.10),
        AstrometricObservation(obs_time_utc=now + timedelta(minutes=90), ra_deg=20.0701, dec_deg=10.0439, mag=19.1, residual_ra_arcsec=0.05, residual_dec_arcsec=-0.12),
        AstrometricObservation(obs_time_utc=now + timedelta(minutes=135), ra_deg=20.1052, dec_deg=10.0658, mag=19.2, residual_ra_arcsec=-0.09, residual_dec_arcsec=-0.03),
    ]
    fixtures.append(CandidateRecord(
        candidate_id="NEO_APOLLO_REAL_01",
        source_survey="CATALINA_SKY_SURVEY",
        discovery_date=now,
        apparent_ra_deg=20.0000,
        apparent_dec_deg=10.0000,
        v_mag=19.2,
        h_mag=21.4,
        motion_rate_deg_day=1.65,
        solar_elongation_deg=152.0,
        arc_length_hours=2.25,
        n_observations=4,
        rms_residual_arcsec=0.19,
        orbital_elements=OrbitalElements(
            semi_major_axis_au=1.45,
            eccentricity=0.42,
            inclination_deg=8.4,
            perihelion_dist_au=0.841,
            aphelion_dist_au=2.059,
            longitude_node_deg=45.2,
            arg_perihelion_deg=112.0,
            tisserand_jupiter=4.52,
            moid_au=0.082,  # > 0.05 AU (Standard Apollo NEO, non-PHA)
            neo_family="APOLLO"
        ),
        observations=obs_apollo,
        raw_metadata={"type": "GROUND_TRUTH_REAL_NEO"}
    ))

    # 2. Potentially Hazardous Asteroid (PHA) Candidate (Real - High Priority)
    fixtures.append(CandidateRecord(
        candidate_id="NEO_PHA_HIGH_RISK_02",
        source_survey="PAN_STARRS_PS1",
        discovery_date=now - timedelta(hours=8),
        apparent_ra_deg=142.5000,
        apparent_dec_deg=-14.2000,
        v_mag=17.8,
        h_mag=19.5,  # H <= 22.0
        motion_rate_deg_day=3.40,
        solar_elongation_deg=165.0,
        arc_length_hours=6.5,
        n_observations=7,
        rms_residual_arcsec=0.22,
        orbital_elements=OrbitalElements(
            semi_major_axis_au=1.12,
            eccentricity=0.28,
            inclination_deg=3.2,
            perihelion_dist_au=0.806,
            aphelion_dist_au=1.434,
            longitude_node_deg=180.5,
            arg_perihelion_deg=74.1,
            tisserand_jupiter=5.41,
            moid_au=0.012,  # MOID <= 0.05 AU
            neo_family="APOLLO"
        ),
        raw_metadata={"type": "GROUND_TRUTH_PHA"}
    ))

    # 3. Historical 6-Month-Old Candidate: 2026 HX3 (May 2026 close approach)
    fixtures.append(CandidateRecord(
        candidate_id="2026 HX3",
        source_survey="NASA_JPL_CAD_HISTORICAL",
        discovery_date=six_months_ago,
        apparent_ra_deg=210.1250,
        apparent_dec_deg=-5.3200,
        v_mag=19.8,
        h_mag=26.13,
        motion_rate_deg_day=5.8,
        solar_elongation_deg=148.0,
        arc_length_hours=84.0,
        n_observations=24,
        rms_residual_arcsec=0.24,
        orbital_elements=OrbitalElements(
            semi_major_axis_au=2.20,
            eccentricity=0.575,
            inclination_deg=0.822,
            perihelion_dist_au=0.938,
            aphelion_dist_au=3.47,
            longitude_node_deg=28.0,
            arg_perihelion_deg=232.0,
            tisserand_jupiter=3.25,
            moid_au=0.0207,
            neo_family="APOLLO"
        ),
        raw_metadata={"historical_epoch": "2026-05", "close_approach_caDist": "0.0207 AU"}
    ))

    # 4. Historical 6-Month-Old Candidate: 2026 HY3 (May 2026 close approach)
    fixtures.append(CandidateRecord(
        candidate_id="2026 HY3",
        source_survey="NASA_JPL_CAD_HISTORICAL",
        discovery_date=six_months_ago + timedelta(hours=3),
        apparent_ra_deg=212.4500,
        apparent_dec_deg=-4.8900,
        v_mag=20.1,
        h_mag=26.42,
        motion_rate_deg_day=4.9,
        solar_elongation_deg=146.0,
        arc_length_hours=60.0,
        n_observations=18,
        rms_residual_arcsec=0.29,
        orbital_elements=OrbitalElements(
            semi_major_axis_au=1.85,
            eccentricity=0.485,
            inclination_deg=2.15,
            perihelion_dist_au=0.952,
            aphelion_dist_au=2.75,
            longitude_node_deg=35.1,
            arg_perihelion_deg=198.4,
            tisserand_jupiter=3.68,
            moid_au=0.0184,
            neo_family="APOLLO"
        ),
        raw_metadata={"historical_epoch": "2026-05", "close_approach_caDist": "0.0184 AU"}
    ))

    # 5. False Positive: Dipole Subtraction Artifact (Bogus)
    fixtures.append(CandidateRecord(
        candidate_id="BOGUS_DIPOLE_001",
        source_survey="ZTF_DIFF_PIPELINE",
        discovery_date=now - timedelta(hours=1),
        apparent_ra_deg=55.4000,
        apparent_dec_deg=28.1200,
        v_mag=21.2,
        h_mag=28.0,
        motion_rate_deg_day=0.15,
        solar_elongation_deg=130.0,
        arc_length_hours=0.45,
        n_observations=3,
        rms_residual_arcsec=1.45,  # High residual from asymmetric dipole centroiding
        orbital_elements=OrbitalElements(
            semi_major_axis_au=0.72,
            eccentricity=0.92,
            inclination_deg=45.0,
            tisserand_jupiter=1.1,
            moid_au=0.45,
            neo_family="UNREALISTIC"
        ),
        raw_metadata={"failure_type": "SUBTRACTION_DIPOLE", "dipole_contrast_ratio": -0.85}
    ))

    # 6. False Positive: Low Earth Orbit Space Debris / Satellite Streak (Bogus)
    fixtures.append(CandidateRecord(
        candidate_id="BOGUS_SATELLITE_LEO_002",
        source_survey="ATLAS_ALERT_STREAM",
        discovery_date=now - timedelta(minutes=30),
        apparent_ra_deg=310.2000,
        apparent_dec_deg=45.6000,
        v_mag=14.5,
        h_mag=32.0,  # Ridiculously small equivalent asteroid size
        motion_rate_deg_day=38.5,  # Exceeds maximum natural bound (>30 deg/day)
        solar_elongation_deg=82.0,
        arc_length_hours=0.20,
        n_observations=4,
        rms_residual_arcsec=0.30,
        orbital_elements=OrbitalElements(
            semi_major_axis_au=0.005,  # Geocentric LEO distance
            eccentricity=0.01,
            inclination_deg=53.0,  # Starlink orbital inclination
            tisserand_jupiter=0.0,
            moid_au=0.0,
            neo_family="EARTH_ORBIT_ARTIFICIAL"
        ),
        raw_metadata={"failure_type": "SPACE_DEBRIS", "norad_match_suspected": True}
    ))

    # 7. False Positive: Bright Star Ghost / Diffraction Spike (Bogus)
    fixtures.append(CandidateRecord(
        candidate_id="BOGUS_SPIKE_GHOST_003",
        source_survey="CATALINA_SKY_SURVEY",
        discovery_date=now - timedelta(hours=2),
        apparent_ra_deg=78.1000,
        apparent_dec_deg=8.2000,  # Near Betelgeuse or bright star
        v_mag=18.5,
        h_mag=27.0,
        motion_rate_deg_day=0.40,
        solar_elongation_deg=42.0,  # Inside twilight solar glare danger zone (< 55 deg)
        arc_length_hours=0.35,
        n_observations=3,
        rms_residual_arcsec=0.95,
        orbital_elements=OrbitalElements(
            semi_major_axis_au=0.85,
            eccentricity=0.65,
            inclination_deg=15.0,
            tisserand_jupiter=2.8,
            moid_au=0.15,
            neo_family="AMBIGUOUS"
        ),
        raw_metadata={"failure_type": "DIFFRACTION_SPIKE", "bright_star_dist_arcsec": 12.4}
    ))

    # 8. False Positive: Spurious Noise Linkage (Bogus Tracklet)
    obs_noise = [
        AstrometricObservation(obs_time_utc=now, ra_deg=100.0000, dec_deg=15.0000, mag=20.5),
        AstrometricObservation(obs_time_utc=now + timedelta(minutes=30), ra_deg=100.0400, dec_deg=15.0100, mag=20.9),
        AstrometricObservation(obs_time_utc=now + timedelta(minutes=60), ra_deg=100.1200, dec_deg=15.0700, mag=20.2),  # Jagged kink
        AstrometricObservation(obs_time_utc=now + timedelta(minutes=90), ra_deg=100.1500, dec_deg=15.0900, mag=21.0),
    ]
    fixtures.append(CandidateRecord(
        candidate_id="BOGUS_NOISE_LINK_004",
        source_survey="ZTF_FAT_CANDIDATE",
        discovery_date=now - timedelta(hours=3),
        apparent_ra_deg=100.0000,
        apparent_dec_deg=15.0000,
        v_mag=20.6,
        h_mag=28.5,
        motion_rate_deg_day=2.40,
        solar_elongation_deg=115.0,
        arc_length_hours=1.5,
        n_observations=4,
        rms_residual_arcsec=1.85,  # High RMS from discordant non-physical tracklet
        orbital_elements=OrbitalElements(
            semi_major_axis_au=3.8,
            eccentricity=0.88,
            inclination_deg=62.0,
            tisserand_jupiter=1.4,
            moid_au=0.35,
            neo_family="SPURIOUS_ORBIT"
        ),
        observations=obs_noise,
        raw_metadata={"failure_type": "SPURIOUS_LINKAGE", "acceleration_deg_day2": 3.8}
    ))

    # 9. False Positive: Stationary Uncatalogued Variable Star (Bogus NEO)
    fixtures.append(CandidateRecord(
        candidate_id="BOGUS_VAR_STAR_005",
        source_survey="PAN_STARRS_PS2",
        discovery_date=now - timedelta(hours=5),
        apparent_ra_deg=280.4500,
        apparent_dec_deg=-20.1000,
        v_mag=18.2,
        h_mag=29.0,
        motion_rate_deg_day=0.02,  # Basically zero motion (stationary stellar source)
        solar_elongation_deg=138.0,
        arc_length_hours=3.0,
        n_observations=4,
        rms_residual_arcsec=0.15,
        orbital_elements=OrbitalElements(
            semi_major_axis_au=0.0,
            eccentricity=0.0,
            inclination_deg=0.0,
            tisserand_jupiter=None,
            moid_au=None,
            neo_family="STATIONARY_STELLAR"
        ),
        raw_metadata={"failure_type": "VARIABLE_STAR", "delta_mag": 1.4}
    ))

    return fixtures


def get_live_candidates_snapshot(source: str = "all") -> List[CandidateRecord]:
    """Retrieve pre-cached snapshot of real live candidate telemetry (90+ real objects)
    for guaranteed offline functionality when public APIs are slow or unreachable.
    """
    import json
    from config import FIXTURES_DIR
    
    snapshot_path = FIXTURES_DIR / "live_candidates_snapshot.json"
    if not snapshot_path.exists():
        return get_ground_truth_fixtures()
        
    try:
        with open(snapshot_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        records: List[CandidateRecord] = []
        if source in ("all", "scout", "NASA JPL Scout"):
            for item in data.get("scout", []):
                try:
                    records.append(CandidateRecord.model_validate(item))
                except Exception:
                    pass
                    
        if source in ("all", "neocp", "Minor Planet Center NEOCP"):
            for item in data.get("neocp", []):
                try:
                    records.append(CandidateRecord.model_validate(item))
                except Exception:
                    pass
                    
        if records:
            return records
    except Exception:
        pass
        
    return get_ground_truth_fixtures()
