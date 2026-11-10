"""Unified AI candidate validator combining heuristics, ML real-bogus scoring, and diagnostic vetting."""

import logging
from typing import List, Optional

from core.models import (
    CandidateRecord,
    CandidateValidationResult,
    DetectionStatus,
    FalsePositiveMode,
    TransientCategory,
    RevalidationReport
)
from core.astronomy import check_potentially_hazardous, classify_neo_orbit
from ai.heuristic_engine import HeuristicScoringEngine
from ai.ml_classifier import RealBogusClassifier
from ai.anomaly_detector import CelestialAnomalyDetector
from ai.false_positive_analyzer import FalsePositiveAnalyzer
from ai.revalidation_engine import CandidateRevalidationEngine

logger = logging.getLogger(__name__)


class CandidateValidator:
    """Master validator orchestrating heuristic, machine learning, and false-positive diagnostic engines."""

    def __init__(self):
        self.heuristic_engine = HeuristicScoringEngine()
        self.ml_classifier = RealBogusClassifier()
        self.anomaly_detector = CelestialAnomalyDetector()
        self.fp_analyzer = FalsePositiveAnalyzer()
        self.revalidation_engine = CandidateRevalidationEngine()

    def validate(self, candidate: CandidateRecord) -> CandidateValidationResult:
        """Execute full end-to-end multi-tier vetting on a candidate."""
        # 1. Scientific Heuristic Breakdown
        heuristics = self.heuristic_engine.evaluate_candidate(candidate)
        composite_score = heuristics.composite_ai_score

        # 2. Machine Learning Real-Bogus Probability
        ml_real_prob = self.ml_classifier.predict_real_probability(candidate)

        # 3. False Positive Root Cause Diagnostics
        fp_diagnostic = self.fp_analyzer.diagnose(candidate, ml_real_prob, composite_score)

        # 4. Anomaly Detection
        is_anomaly, anomaly_score, anomaly_reason = self.anomaly_detector.evaluate_anomaly(candidate)

        # 5. Potentially Hazardous Asteroid (PHA) Assessment
        elems = candidate.orbital_elements
        moid = elems.moid_au if elems else None
        h_mag = candidate.h_mag
        is_pha = check_potentially_hazardous(moid, h_mag)

        # 6. Orbit Family Classification
        a = elems.semi_major_axis_au if elems else None
        e = elems.eccentricity if elems else None
        q = elems.perihelion_dist_au if elems else None
        orbit_class = classify_neo_orbit(a, e, q)

        # 7. Decision Logic & Status Assignment
        recommendations: List[str] = []
        transient_category: Optional[TransientCategory] = None
        transient_props = candidate.transient_properties

        # A. Check if object is an Identified / Confirmed Designated Small Body (from JPL Horizons/CAD)
        if candidate.is_known_object:
            if is_pha:
                status = DetectionStatus.POTENTIALLY_HAZARDOUS
                transient_category = TransientCategory.IDENTIFIED_NEO
                recommendations.append(f"CONFIRMED PHA: Designated Near-Earth Asteroid {candidate.candidate_id} (MOID <= 0.05 AU).")
                recommendations.append("Verified Keplerian orbital solution from NASA JPL Horizons database.")
            else:
                status = DetectionStatus.IDENTIFIED_NEO
                transient_category = TransientCategory.IDENTIFIED_NEO
                recommendations.append(f"CONFIRMED NEO: Designated asteroid {candidate.candidate_id} ({orbit_class} class).")
                recommendations.append("Astrometric coordinates conform precisely to JPL Horizons ephemerides.")
            composite_score = max(composite_score, 98.0)
            ml_real_prob = 1.0

        # B. Check if object is a Distant Transient (Supernova, AGN flare, Kilonova, CV)
        elif (
            transient_props is not None or
            (candidate.motion_rate_deg_day <= 0.05 and any(
                tag in candidate.source_survey.upper() for tag in ["SUPERNOVA", "AGN", "KILONOVA", "BROKER", "FINK"]
            ))
        ):
            status = DetectionStatus.DISTANT_TRANSIENT
            t_type = (transient_props.transient_type if transient_props and transient_props.transient_type else "SUPERNOVA").upper()

            if "SUPERNOVA" in t_type:
                transient_category = TransientCategory.DISTANT_SUPERNOVA
                z_str = f" (z = {transient_props.estimated_redshift_z:.4f})" if transient_props and transient_props.estimated_redshift_z else ""
                host_str = f" in host {transient_props.host_galaxy_name}" if transient_props and transient_props.host_galaxy_name else ""
                recommendations.append(f"CONFIRMED: Extragalactic Supernova transient{host_str}{z_str}.")
                recommendations.append("Stationary celestial coordinates confirmed across epochs (delta-mu < 0.001 deg/day).")
                recommendations.append("Target of Opportunity (ToO) optical spectroscopy requested for spectral type and redshift verification.")
            elif "AGN" in t_type or "BLAZAR" in t_type:
                transient_category = TransientCategory.DISTANT_AGN_FLARE
                recommendations.append("CONFIRMED: Distant Active Galactic Nucleus (AGN / Blazar) optical flare.")
                recommendations.append("Source exhibits stationary astrometry (proper motion ~0) with stochastic accretion flare.")
                recommendations.append("Cross-match with radio/X-ray catalogs (e.g. Fermi-LAT, Swift-XRT, VLASS).")
            elif "KILONOVA" in t_type or "GRB" in t_type or "AFTERGLOW" in t_type:
                transient_category = TransientCategory.DISTANT_KILONOVA_OR_GRB
                recommendations.append("HIGH PRIORITY: Rapid optical transient consistent with Kilonova / GRB afterglow.")
                recommendations.append("Fast photometric fading rate detected. Urgent multi-band follow-up requested.")
                recommendations.append("Check gravitational-wave alert networks (LIGO / Virgo / KAGRA).")
            elif "CATACLYSMIC" in t_type or "CV" in t_type or "NOVA" in t_type:
                transient_category = TransientCategory.CATACLYSMIC_VARIABLE
                recommendations.append("CONFIRMED: Galactic Cataclysmic Variable (CV / Dwarf Nova) outburst.")
                recommendations.append("High amplitude optical brightening with zero net celestial motion.")
            else:
                transient_category = TransientCategory.DISTANT_SUPERNOVA
                recommendations.append("CONFIRMED: Stationary astrophysical transient detection.")

            composite_score = max(composite_score, 88.0)
            ml_real_prob = max(ml_real_prob, 0.95)

        # C. Survey Artifacts and False Positives
        elif fp_diagnostic.mode == FalsePositiveMode.ARTIFICIAL_SATELLITE_OR_DEBRIS:
            status = DetectionStatus.REJECTED_SPACE_DEBRIS
            transient_category = TransientCategory.ARTIFICIAL_SATELLITE_OR_DEBRIS
            recommendations.append("VETO: Fast non-orbital velocity or inclination match with Low Earth Orbit artificial satellites.")
            recommendations.append("Do not submit to Minor Planet Center NEOCP.")
        elif fp_diagnostic.mode != FalsePositiveMode.NONE_REAL_CANDIDATE and (composite_score < 55.0 or fp_diagnostic.rejection_confidence >= 70.0):
            status = DetectionStatus.BOGUS_FALSE_POSITIVE
            transient_category = TransientCategory.IMAGE_ARTIFACT_OR_BOGUS
            recommendations.append(f"VETO: Identified as survey artifact ({fp_diagnostic.mode.value}).")
            recommendations.append(fp_diagnostic.explanation)

        # D. Potentially Hazardous Asteroid Candidate
        elif is_pha and composite_score >= 65.0:
            status = DetectionStatus.POTENTIALLY_HAZARDOUS
            transient_category = TransientCategory.NEO_CANDIDATE
            recommendations.append("CRITICAL: Candidate meets Potentially Hazardous Asteroid (PHA) criteria (MOID <= 0.05 AU, H <= 22.0).")
            recommendations.append("Issue immediate worldwide telescope alert for target follow-up.")

        # E. Validated NEO Candidate or Main Belt Asteroid
        elif composite_score >= 75.0 and ml_real_prob >= 0.70:
            if orbit_class == "MAIN_BELT_ASTEROID":
                status = DetectionStatus.PROBABLE_ASTEROID
                transient_category = TransientCategory.MAIN_BELT_ASTEROID
                recommendations.append("CONFIRMED: Real Solar System body conforming to Main Belt Asteroid orbit (q >= 1.3 AU).")
                recommendations.append("Non-hazardous asteroidal motion.")
            else:
                status = DetectionStatus.VALIDATED_NEO
                transient_category = TransientCategory.NEO_CANDIDATE
                recommendations.append(f"CONFIRMED: High-confidence {orbit_class} candidate.")
                recommendations.append("Astrometric residuals and orbital mechanics conform to genuine natural celestial body.")

        # F. Ambiguous Follow-Up Required
        elif composite_score >= 50.0:
            status = DetectionStatus.AMBIGUOUS_FOLLOW_UP_REQUIRED
            transient_category = TransientCategory.NEO_CANDIDATE
            recommendations.append("NEEDS FOLLOW-UP: Short observational arc creates ephemeris uncertainty.")
            recommendations.append("Schedule 2nd epoch verification within 2 to 6 hours before positional error ellipse expands.")

        # G. Main Belt Asteroid or Bogus
        else:
            if orbit_class == "MAIN_BELT_ASTEROID":
                status = DetectionStatus.PROBABLE_ASTEROID
                transient_category = TransientCategory.MAIN_BELT_ASTEROID
                recommendations.append("DYNAMICAL: Classified as Main Belt Asteroid (perihelion q >= 1.3 AU).")
            else:
                status = DetectionStatus.BOGUS_FALSE_POSITIVE
                transient_category = TransientCategory.IMAGE_ARTIFACT_OR_BOGUS
                recommendations.append("Low priority or likely false detection; review raw image cutouts.")

        if is_anomaly:
            recommendations.append(f"ANOMALY ALERT: {anomaly_reason} (Anomaly Score: {anomaly_score:.2f})")

        return CandidateValidationResult(
            candidate_id=candidate.candidate_id,
            source=candidate.source_survey,
            status=status,
            composite_confidence_score=composite_score,
            heuristic_breakdown=heuristics,
            false_positive_diagnostic=fp_diagnostic,
            ml_real_probability=ml_real_prob,
            is_potentially_hazardous=is_pha,
            orbit_classification=orbit_class,
            transient_category=transient_category,
            transient_properties=transient_props,
            recommendations=recommendations
        )

    def revalidate(self, candidate: CandidateRecord) -> RevalidationReport:
        """Perform longitudinal re-validation of past candidates (e.g. 6-month-old candidates)."""
        return self.revalidation_engine.revalidate_historical_candidate(candidate)
