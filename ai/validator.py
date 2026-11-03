"""Unified AI candidate validator combining heuristics, ML real-bogus scoring, and diagnostic vetting."""

import logging
from typing import List, Optional

from core.models import (
    CandidateRecord,
    CandidateValidationResult,
    DetectionStatus,
    FalsePositiveMode,
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

        if fp_diagnostic.mode == FalsePositiveMode.ARTIFICIAL_SATELLITE_OR_DEBRIS:
            status = DetectionStatus.REJECTED_SPACE_DEBRIS
            recommendations.append("VETO: Fast non-orbital velocity or inclination match with Low Earth Orbit artificial satellites.")
            recommendations.append("Do not submit to Minor Planet Center NEOCP.")
        elif fp_diagnostic.mode != FalsePositiveMode.NONE_REAL_CANDIDATE and (composite_score < 55.0 or fp_diagnostic.rejection_confidence >= 70.0):
            status = DetectionStatus.BOGUS_FALSE_POSITIVE
            recommendations.append(f"VETO: Identified as survey artifact ({fp_diagnostic.mode.value}).")
            recommendations.append(fp_diagnostic.explanation)
        elif is_pha and composite_score >= 65.0:
            status = DetectionStatus.POTENTIALLY_HAZARDOUS
            recommendations.append("CRITICAL: Candidate meets Potentially Hazardous Asteroid (PHA) criteria (MOID <= 0.05 AU, H <= 22.0).")
            recommendations.append("Issue immediate worldwide telescope alert for target follow-up.")
        elif composite_score >= 75.0 and ml_real_prob >= 0.70:
            status = DetectionStatus.VALIDATED_NEO
            recommendations.append(f"CONFIRMED: High-confidence {orbit_class} candidate.")
            recommendations.append("Astrometric residuals and orbital mechanics conform to genuine natural celestial body.")
        elif composite_score >= 50.0:
            status = DetectionStatus.AMBIGUOUS_FOLLOW_UP_REQUIRED
            recommendations.append("NEEDS FOLLOW-UP: Short observational arc creates ephemeris uncertainty.")
            recommendations.append("Schedule 2nd epoch verification within 2 to 6 hours before positional error ellipse expands.")
        else:
            status = DetectionStatus.PROBABLE_ASTEROID if orbit_class == "MAIN_BELT_ASTEROID" else DetectionStatus.BOGUS_FALSE_POSITIVE
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
            recommendations=recommendations
        )

    def revalidate(self, candidate: CandidateRecord) -> RevalidationReport:
        """Perform longitudinal re-validation of past candidates (e.g. 6-month-old candidates)."""
        return self.revalidation_engine.revalidate_historical_candidate(candidate)
