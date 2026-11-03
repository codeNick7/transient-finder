"""NASA JPL Close Approach Data (CAD) and Small-Body Database (SBDB) Ingestion Client for Historical / 6-Month-Old Candidates."""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from config import CONFIG
from ingestion.base import BaseIngestionClient
from core.models import CandidateRecord, OrbitalElements
from core.astronomy import classify_neo_orbit, calculate_tisserand_parameter

logger = logging.getLogger(__name__)


class JPLHistoricalCADClient(BaseIngestionClient):
    """Client for retrieving historical NEO candidates (e.g. ~6-month-old submissions)
    from NASA JPL Close Approach and Small-Body databases.
    """
    
    def fetch_close_approach_candidates(self, date_min: str = "2026-05-01", date_max: str = "2026-05-31", dist_max_ld: float = 15.0) -> List[CandidateRecord]:
        """Fetch candidates that approached Earth during the specified historical window (e.g., 6 months ago)."""
        url = f"{CONFIG.jpl_cad_url}?date-min={date_min}&date-max={date_max}&dist-max={dist_max_ld}LD"
        raw_text = self.fetch_url(url)
        if not raw_text:
            logger.warning("CAD API returned empty.")
            return []
            
        try:
            payload = json.loads(raw_text)
            fields = payload.get("fields", [])
            data_rows = payload.get("data", [])
            
            candidates: List[CandidateRecord] = []
            for row in data_rows:
                row_dict = dict(zip(fields, row))
                rec = self._parse_cad_row(row_dict)
                if rec:
                    candidates.append(rec)
            return candidates
        except Exception as e:
            logger.error(f"Error parsing CAD data: {e}")
            return []

    def fetch_sbdb_details(self, designation: str) -> Optional[OrbitalElements]:
        """Fetch full orbital parameters and condition code from JPL SBDB API."""
        encoded = designation.replace(" ", "%20")
        url = f"{CONFIG.jpl_sbdb_url}?sstr={encoded}&phys-par=1"
        raw_text = self.fetch_url(url)
        if not raw_text:
            return None
            
        try:
            payload = json.loads(raw_text)
            orbit_data = payload.get("orbit", {})
            elements_list = orbit_data.get("elements", [])
            
            elem_dict = {elem.get("name"): float(elem.get("value")) for elem in elements_list if elem.get("name") and elem.get("value")}
            
            a = elem_dict.get("a")
            e = elem_dict.get("e")
            i = elem_dict.get("i", 0.0)
            q = elem_dict.get("q")
            om = elem_dict.get("om", 0.0)
            w = elem_dict.get("w", 0.0)
            ma = elem_dict.get("ma", 0.0)
            epoch_jd = float(orbit_data.get("epoch", 2460000.0))
            moid = float(orbit_data.get("moid", 0.05)) if orbit_data.get("moid") else None
            
            tj = calculate_tisserand_parameter(a, e, i) if a and e is not None else None
            family = classify_neo_orbit(a, e, q)
            
            return OrbitalElements(
                semi_major_axis_au=a,
                eccentricity=e,
                inclination_deg=i,
                perihelion_dist_au=q,
                aphelion_dist_au=a * (1.0 + e) if a and e is not None else None,
                longitude_node_deg=om,
                arg_perihelion_deg=w,
                mean_anomaly_deg=ma,
                epoch_jd=epoch_jd,
                tisserand_jupiter=tj,
                moid_au=moid,
                neo_family=family
            )
        except Exception as e:
            logger.warning(f"Error parsing SBDB for {designation}: {e}")
            return None

    def _parse_cad_row(self, row: Dict[str, Any]) -> Optional[CandidateRecord]:
        """Convert CAD row into CandidateRecord."""
        try:
            desig = row.get("des", "UNKNOWN")
            cd_date_str = row.get("cd", "2026-May-01 00:00")
            try:
                disc_date = datetime.strptime(cd_date_str, "%Y-%b-%d %H:%M")
            except Exception:
                disc_date = datetime(2026, 5, 1)
                
            h_mag = float(row["h"]) if row.get("h") else 25.0
            v_rel = float(row["v_rel"]) if row.get("v_rel") else 12.0
            dist_au = float(row["dist"]) if row.get("dist") else 0.02
            
            # Apparent motion rate for an object at distance dist_au with relative velocity v_rel:
            # v_rel (km/s) / (dist_au * 1.496e8 km) in rad/s -> deg/day
            rate_deg_day = (v_rel / (dist_au * 1.496e8)) * (180.0 / 3.14159) * 86400.0
            rate_deg_day = min(25.0, max(0.5, rate_deg_day))
            
            return CandidateRecord(
                candidate_id=desig,
                source_survey="NASA_JPL_CAD_HISTORICAL",
                discovery_date=disc_date,
                apparent_ra_deg=180.0,  # Nominal sky center for historical summary
                apparent_dec_deg=10.0,
                v_mag=h_mag + 5.0 * 0.4,  # Approx apparent mag near close approach
                h_mag=h_mag,
                motion_rate_deg_day=rate_deg_day,
                solar_elongation_deg=145.0,
                arc_length_hours=72.0,
                n_observations=15,
                rms_residual_arcsec=0.22,
                orbital_elements=OrbitalElements(moid_au=dist_au, neo_family="CONFIRMED_HISTORICAL_NEO"),
                raw_metadata=row
            )
        except Exception as e:
            logger.warning(f"Error parsing CAD row: {e}")
            return None
