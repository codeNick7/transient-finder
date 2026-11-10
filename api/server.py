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
from core.models import (
    CandidateRecord,
    CandidateValidationResult,
    RevalidationReport,
    SkyPatchQuery,
    SkyPatchResult
)
from ingestion.jpl_scout import JPLScoutClient
from ingestion.mpc_neocp import MPCNEOCPClient
from ingestion.jpl_cad import JPLHistoricalCADClient
from ingestion.jpl_horizons import JPLHorizonsClient
from ingestion.sky_patch import SkyPatchService
from ingestion.fixtures import get_ground_truth_fixtures, get_live_candidates_snapshot
from ai.validator import CandidateValidator

app = FastAPI(
    title=CONFIG.app_name,
    version=CONFIG.version,
    description="AI-driven public survey data analysis, candidate validation, on-demand sky patch query, and false-positive diagnostic engine."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared validator and service instances
validator = CandidateValidator()
scout_client = JPLScoutClient()
neocp_client = MPCNEOCPClient()
cad_client = JPLHistoricalCADClient()
horizons_client = JPLHorizonsClient()
sky_patch_service = SkyPatchService()


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
            "/api/sky-patch/query",
            "/api/sky-patch/search",
            "/api/horizons/ephemerides",
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


@app.post("/api/sky-patch/query", response_model=SkyPatchResult)
def query_sky_patch(query: SkyPatchQuery):
    """On-demand cone search query for an arbitrary patch of sky (RA & Dec) to detect and validate NEOs & distant transients."""
    try:
        return sky_patch_service.query_sky_patch(
            ra_deg=query.ra_deg,
            dec_deg=query.dec_deg,
            radius_deg=query.radius_deg,
            target_id=query.target_id,
            epoch=query.epoch,
            sources=query.sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sky patch query failed: {str(e)}")


@app.get("/api/sky-patch/search", response_model=SkyPatchResult)
def search_sky_patch(
    ra: float = Query(..., ge=0.0, lt=360.0, description="Central Right Ascension in degrees"),
    dec: float = Query(..., ge=-90.0, le=90.0, description="Central Declination in degrees"),
    radius: float = Query(1.0, gt=0.0, le=15.0, description="Search cone radius in degrees"),
    target_id: Optional[str] = Query(None, description="Optional target identifier (e.g. 1998 KY26)"),
    epoch: Optional[str] = Query(None, description="Observation epoch/date (e.g. 2024-06-01)")
):
    """GET endpoint to search and analyze an on-demand sky patch via query parameters."""
    try:
        return sky_patch_service.query_sky_patch(
            ra_deg=ra,
            dec_deg=dec,
            radius_deg=radius,
            target_id=target_id,
            epoch=epoch
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sky patch search failed: {str(e)}")


@app.get("/api/horizons/ephemerides", response_model=Optional[CandidateRecord])
def get_target_ephemerides(
    target_id: str = Query("1998 KY26", description="Small body target identifier"),
    start_date: str = Query("2024-06-01", description="Start date (YYYY-MM-DD)"),
    stop_date: str = Query("2024-06-02", description="Stop date (YYYY-MM-DD)"),
    step: str = Query("6h", description="Time step between ephemeris records")
):
    """Query NASA JPL Horizons for high-precision ephemerides and orbital state of a designated body."""
    res = horizons_client.fetch_target_ephemerides(
        target_id=target_id,
        start_date=start_date,
        stop_date=stop_date,
        step=step
    )
    if not res:
        raise HTTPException(status_code=404, detail=f"No Horizons ephemerides found for target '{target_id}'")
    return res


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
