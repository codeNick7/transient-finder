"""Unit and integration tests for On-Demand Sky Patch Search and JPL Horizons Ephemerides."""

import pytest
from fastapi.testclient import TestClient

from core.models import (
    TransientCategory,
    DetectionStatus,
    SkyPatchQuery,
    SkyPatchResult,
    CandidateRecord
)
from core.astronomy import great_circle_distance_deg
from ingestion.jpl_horizons import JPLHorizonsClient
from ingestion.sky_patch import SkyPatchService
from api.server import app


@pytest.fixture
def horizons_client():
    return JPLHorizonsClient()


@pytest.fixture
def patch_service():
    return SkyPatchService()


@pytest.fixture
def api_client():
    return TestClient(app)


def test_great_circle_distance():
    """Verify angular separation calculations on the celestial sphere."""
    # Distance between same coordinates is 0
    assert great_circle_distance_deg(180.0, 0.0, 180.0, 0.0) == 0.0

    # 1 degree separation along declination
    assert pytest.approx(great_circle_distance_deg(100.0, 10.0, 100.0, 11.0), abs=1e-4) == 1.0

    # 1 degree separation along celestial equator
    assert pytest.approx(great_circle_distance_deg(0.0, 0.0, 1.0, 0.0), abs=1e-4) == 1.0

    # 90 degrees separation
    assert pytest.approx(great_circle_distance_deg(0.0, 0.0, 90.0, 0.0), abs=1e-4) == 90.0


def test_jpl_horizons_1998_ky26(horizons_client):
    """Verify retrieval and parsing of 1998 KY26 ephemerides and orbital characteristics."""
    cand = horizons_client.fetch_target_ephemerides(
        target_id="1998 KY26",
        start_date="2024-06-01",
        stop_date="2024-06-02",
        step="6h"
    )
    assert cand is not None
    assert "1998 KY26" in cand.candidate_id
    assert cand.is_known_object is True

    # Coordinates near Earth close approach of June 2024
    assert 310.0 <= cand.apparent_ra_deg <= 325.0
    assert -10.0 <= cand.apparent_dec_deg <= 0.0

    # High apparent motion of close-approaching NEO (> 3 deg/day)
    assert cand.motion_rate_deg_day > 3.0

    # Orbital elements
    assert cand.orbital_elements is not None
    assert cand.orbital_elements.neo_family == "APOLLO"
    assert cand.orbital_elements.semi_major_axis_au > 1.0
    assert cand.orbital_elements.perihelion_dist_au < 1.05


def test_sky_patch_ky26_cone_search(patch_service):
    """Test on-demand cone search containing 1998 KY26 and Main Belt Asteroid."""
    result = patch_service.query_sky_patch(
        ra_deg=315.00,
        dec_deg=-6.27,
        radius_deg=1.50,
        target_id="1998 KY26",
        epoch="2024-06-01"
    )
    assert isinstance(result, SkyPatchResult)
    assert result.total_detected >= 1
    assert result.neo_count >= 1

    # Verify 1998 KY26 is the closest detection to the search center
    top_obj = result.objects[0]
    assert top_obj.candidate.candidate_id == "1998 KY26"
    assert top_obj.angular_separation_deg < 0.2
    assert top_obj.category == TransientCategory.IDENTIFIED_NEO
    assert top_obj.validation.status == DetectionStatus.IDENTIFIED_NEO
    assert top_obj.validation.composite_confidence_score >= 95.0


def test_sky_patch_distant_supernova(patch_service):
    """Test on-demand cone search around distant supernova SN 2024ggi."""
    result = patch_service.query_sky_patch(
        ra_deg=168.03,
        dec_deg=-10.52,
        radius_deg=2.00
    )
    assert result.total_detected >= 1
    assert result.distant_transient_count >= 1

    sn_obj = next((o for o in result.objects if o.candidate.candidate_id == "SN 2024ggi"), None)
    assert sn_obj is not None
    assert sn_obj.category == TransientCategory.DISTANT_SUPERNOVA
    assert sn_obj.validation.status == DetectionStatus.DISTANT_TRANSIENT
    assert sn_obj.candidate.motion_rate_deg_day < 0.01  # Stationary
    assert sn_obj.candidate.transient_properties is not None
    assert sn_obj.candidate.transient_properties.transient_type == "SUPERNOVA_TYPE_II"
    assert sn_obj.validation.composite_confidence_score >= 80.0


def test_sky_patch_custom_synthesis(patch_service):
    """Test field synthesis when an empty or arbitrary sky patch is queried."""
    # Isolated coordinates (RA 75.0, Dec +60.0)
    result = patch_service.query_sky_patch(
        ra_deg=75.0,
        dec_deg=60.0,
        radius_deg=1.0
    )
    assert result.total_detected >= 1
    # Check that angular separations are strictly within the 1.0 deg radius
    for obj in result.objects:
        assert obj.angular_separation_deg <= 1.0


def test_api_sky_patch_endpoints(api_client):
    """Test FastAPI REST endpoints for sky-patch query and search."""
    # 1. GET Horizons ephemerides
    res_horizons = api_client.get("/api/horizons/ephemerides?target_id=1998+KY26")
    assert res_horizons.status_code == 200
    assert res_horizons.json()["candidate_id"] == "1998 KY26"

    # 2. GET Sky Patch search
    res_get = api_client.get("/api/sky-patch/search?ra=315.0&dec=-6.27&radius=1.5&target_id=1998+KY26")
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert data_get["total_detected"] >= 1
    assert data_get["neo_count"] >= 1

    # 3. POST Sky Patch query
    payload = {
        "ra_deg": 168.03,
        "dec_deg": -10.52,
        "radius_deg": 2.0,
        "sources": ["FINK_BROKER"]
    }
    res_post = api_client.post("/api/sky-patch/query", json=payload)
    assert res_post.status_code == 200
    data_post = res_post.json()
    assert data_post["distant_transient_count"] >= 1


def test_cli_patch_command(capsys):
    """Test CLI patch subcommand execution."""
    from cli.main import main
    import sys

    # Simulate: python cli/main.py patch --ra 315.0 --dec -6.27 --radius 1.5 --target "1998 KY26"
    test_args = ["main.py", "patch", "--ra", "315.0", "--dec", "-6.27", "--radius", "1.5", "--target", "1998 KY26"]
    orig_argv = sys.argv
    try:
        sys.argv = test_args
        main()
    finally:
        sys.argv = orig_argv

    captured = capsys.readouterr()
    # Output should include target and detections summary
    assert "1998 KY26" in captured.out or "Detections Summary" in captured.out
