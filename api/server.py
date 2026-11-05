"""FastAPI REST API server for the Transient-Finder NEO AI System."""

import sys
from pathlib import Path

# Automatically ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config import CONFIG
from core.models import CandidateRecord, CandidateValidationResult, RevalidationReport
from ingestion.jpl_scout import JPLScoutClient
from ingestion.mpc_neocp import MPCNEOCPClient
from ingestion.jpl_cad import JPLHistoricalCADClient
from ingestion.fixtures import get_ground_truth_fixtures, get_live_candidates_snapshot
from ai.validator import CandidateValidator

app = FastAPI(
    title=CONFIG.app_name,
    version=CONFIG.version,
    description="AI-driven public survey data analysis, candidate validation, and false-positive diagnostic engine."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared validator instance
validator = CandidateValidator()
scout_client = JPLScoutClient()
neocp_client = MPCNEOCPClient()
cad_client = JPLHistoricalCADClient()


@app.get("/")
def read_root():
    return {
        "system": CONFIG.app_name,
        "version": CONFIG.version,
        "status": "OPERATIONAL",
        "endpoints": [
            "/api/candidates/live",
            "/api/candidates/fixtures",
            "/api/candidates/historical",
            "/api/validate",
            "/api/revalidate",
            "/api/model/importance"
        ]
    }


@app.get("/api/candidates/fixtures", response_model=List[CandidateRecord])
def get_fixtures():
    """Retrieve benchmark candidate fixtures including verified NEOs, 6-month historical objects, and false positive testcases."""
    return get_ground_truth_fixtures()


@app.get("/api/candidates/live", response_model=List[CandidateRecord])
def get_live_candidates(source: str = Query("all", enum=["all", "scout", "neocp"]), limit: int = 50):
    """Retrieve live candidate submissions from NASA JPL Scout and Minor Planet Center NEOCP."""
    results: List[CandidateRecord] = []
    
    if source in ("all", "scout"):
        scout_cands = scout_client.fetch_current_candidates()
        results.extend(scout_cands)
        
    if source in ("all", "neocp"):
        neocp_cands = neocp_client.fetch_current_candidates()
        results.extend(neocp_cands)
        
    if not results:
        # Fallback to persistent live snapshot if network offline
        return get_live_candidates_snapshot(source)[:limit]
        
    return results[:limit]


@app.get("/api/candidates/historical", response_model=List[CandidateRecord])
def get_historical_candidates(date_min: str = "2026-05-01", date_max: str = "2026-05-31", limit: int = 25):
    """Retrieve candidates submitted ~6 months ago (e.g., May 2026) for longitudinal re-validation."""
    cands = cad_client.fetch_close_approach_candidates(date_min=date_min, date_max=date_max)
    if not cands:
        fixtures = [f for f in get_ground_truth_fixtures() if "2026" in f.candidate_id]
        return fixtures
    return cands[:limit]


@app.post("/api/validate", response_model=CandidateValidationResult)
def validate_candidate(candidate: CandidateRecord):
    """Execute AI heuristic scoring, ML real-bogus classification, and false positive diagnostic analysis on candidate."""
    try:
        return validator.validate(candidate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.post("/api/revalidate", response_model=RevalidationReport)
def revalidate_candidate(candidate: CandidateRecord):
    """Perform historical re-validation on a candidate (e.g. 6-month-old submissions)."""
    try:
        return validator.revalidate(candidate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-validation failed: {str(e)}")


@app.get("/api/model/importance")
def get_feature_importance():
    """Return feature importances from the supervised Real-Bogus gradient boosting classifier."""
    return {
        "feature_importances": validator.ml_classifier.get_feature_importances()
    }
