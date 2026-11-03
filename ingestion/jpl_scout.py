"""NASA JPL Center for Near Earth Object Studies (CNEOS) Scout API Ingestion Client."""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from config import CONFIG
from ingestion.base import BaseIngestionClient
from core.models import CandidateRecord, OrbitalElements
from core.astronomy import classify_neo_orbit, calculate_tisserand_parameter

logger = logging.getLogger(__name__)


def parse_ra_hms_to_deg(ra_str: str) -> float:
    """Parse 'HH:MM' or 'HH:MM:SS' string into RA degrees."""
    try:
        parts = [float(p) for p in ra_str.strip().split(":")]
        if len(parts) == 2:
            h, m = parts
            s = 0.0
        elif len(parts) >= 3:
            h, m, s = parts[:3]
        else:
            return float(ra_str)
        return (h + m / 60.0 + s / 3600.0) * 15.0
    except Exception:
        return 0.0


def parse_dec_dms_to_deg(dec_str: str) -> float:
    """Parse '+/-DD:MM' or '+/-DD:MM' into Dec degrees."""
    try:
        sign = -1.0 if dec_str.strip().startswith("-") else 1.0
        cleaned = dec_str.strip().lstrip("+-")
        parts = [float(p) for p in cleaned.split(":")]
        if len(parts) == 1:
            d = parts[0]
            m = 0.0
        else:
            d = parts[0]
            m = parts[1]
        return sign * (d + m / 60.0)
    except Exception:
        return 0.0


class JPLScoutClient(BaseIngestionClient):
    """Client for querying the NASA JPL Scout API for unconfirmed NEOCP candidates."""
    
    def fetch_current_candidates(self) -> List[CandidateRecord]:
        """Fetch summary of all unconfirmed Near-Earth Object candidates from JPL Scout."""
        raw_text = self.fetch_url(CONFIG.jpl_scout_url)
        if not raw_text:
            logger.warning("Scout API returned empty or failed to connect.")
            return []
            
        try:
            payload = json.loads(raw_text)
            data_rows = payload.get("data", [])
            records: List[CandidateRecord] = []
            
            for row in data_rows:
                rec = self._parse_scout_row(row)
                if rec:
                    records.append(rec)
            return records
        except Exception as e:
            logger.error(f"Error parsing JPL Scout response: {e}")
            return []

    def fetch_candidate_orbits(self, tdes: str) -> Optional[OrbitalElements]:
        """Fetch detailed Monte Carlo orbit samples for a specific Scout candidate."""
        url = f"{CONFIG.jpl_scout_url}?tdes={tdes}&orbits=1"
        raw_text = self.fetch_url(url)
        if not raw_text:
            return None
            
        try:
            payload = json.loads(raw_text)
            orbits_obj = payload.get("orbits", {})
            data_rows = orbits_obj.get("data", [])
            fields = orbits_obj.get("fields", [])
            
            if not data_rows or not fields:
                return None
                
            # Sample median from top orbit solution
            first_row = dict(zip(fields, data_rows[0]))
            
            ec = float(first_row.get("ec", 0.0))
            qr = float(first_row.get("qr", 1.0))
            inc = float(first_row.get("inc", 0.0))
            semi_major = qr / (1.0 - ec) if (1.0 - ec) > 1e-4 else None
            moid = float(first_row.get("moid", 0.1))
            
            t_jup = calculate_tisserand_parameter(semi_major, ec, inc) if semi_major else None
            family = classify_neo_orbit(semi_major, ec, qr)
            
            return OrbitalElements(
                semi_major_axis_au=semi_major,
                eccentricity=ec,
                inclination_deg=inc,
                perihelion_dist_au=qr,
                aphelion_dist_au=semi_major * (1.0 + ec) if semi_major else None,
                longitude_node_deg=float(first_row.get("om", 0.0)),
                arg_perihelion_deg=float(first_row.get("w", 0.0)),
                epoch_jd=float(first_row.get("epoch", 2460000.0)),
                tisserand_jupiter=t_jup,
                moid_au=moid,
                neo_family=family
            )
        except Exception as e:
            logger.warning(f"Error parsing orbits for {tdes}: {e}")
            return None

    def _parse_scout_row(self, row: Dict[str, Any]) -> Optional[CandidateRecord]:
        """Convert a Scout summary data row into a CandidateRecord."""
        try:
            candidate_id = row.get("objectName", "UNKNOWN")
            ra_deg = parse_ra_hms_to_deg(str(row.get("ra", "00:00")))
            dec_deg = parse_dec_dms_to_deg(str(row.get("dec", "+00")))
            
            v_mag = float(row["Vmag"]) if row.get("Vmag") else None
            h_mag = float(row["H"]) if row.get("H") else None
            
            # Rate in Scout is typically arcsec/min -> convert to deg/day
            rate_arcsec_min = float(row.get("rate", 1.0))
            rate_deg_day = (rate_arcsec_min * 60.0 * 24.0) / 3600.0
            
            arc_hours = float(row.get("arc", 1.0))
            n_obs = int(row.get("nObs", 3))
            rms = float(row.get("rmsN", 0.35))
            elong = float(row.get("elong", 140.0))
            
            last_run_str = row.get("lastRun")
            try:
                disc_date = datetime.strptime(last_run_str, "%Y-%m-%d %H:%M") if last_run_str else datetime.utcnow()
            except Exception:
                disc_date = datetime.utcnow()
                
            moid_val = float(row["moid"]) if row.get("moid") is not None else None
            
            orbital_elems = OrbitalElements(
                moid_au=moid_val,
                neo_family="NEO_CANDIDATE" if float(row.get("neoScore", 0)) > 50 else "AMBIGUOUS"
            )
            
            return CandidateRecord(
                candidate_id=candidate_id,
                source_survey="NASA_JPL_SCOUT",
                discovery_date=disc_date,
                apparent_ra_deg=ra_deg,
                apparent_dec_deg=dec_deg,
                v_mag=v_mag,
                h_mag=h_mag,
                motion_rate_deg_day=rate_deg_day,
                solar_elongation_deg=elong,
                arc_length_hours=arc_hours,
                n_observations=n_obs,
                rms_residual_arcsec=rms,
                orbital_elements=orbital_elems,
                raw_metadata=row
            )
        except Exception as e:
            logger.warning(f"Error parsing Scout row: {e}")
            return None
