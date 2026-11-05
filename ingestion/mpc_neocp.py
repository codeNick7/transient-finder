"""Minor Planet Center (MPC) NEO Confirmation Page (NEOCP) Ingestion Client."""

import logging
from datetime import datetime
from typing import List, Optional

from config import CONFIG
from ingestion.base import BaseIngestionClient
from core.models import CandidateRecord, OrbitalElements
from core.astronomy import calculate_solar_elongation_deg

logger = logging.getLogger(__name__)


class MPCNEOCPClient(BaseIngestionClient):
    """Client for parsing the live Minor Planet Center NEOCP bulletin feed."""
    
    def fetch_current_candidates(self, force_refresh: bool = False) -> List[CandidateRecord]:
        """Fetch candidates from the Minor Planet Center NEOCP feed."""
        raw_text = self.fetch_url(CONFIG.mpc_neocp_url, force_refresh=force_refresh)
        if not raw_text:
            logger.warning("MPC NEOCP feed returned empty.")
            return []
            
        candidates: List[CandidateRecord] = []
        for line in raw_text.splitlines():
            if not line.strip() or len(line) < 30:
                continue
            rec = self._parse_neocp_line(line)
            if rec:
                candidates.append(rec)
        return candidates

    def _parse_neocp_line(self, line: str) -> Optional[CandidateRecord]:
        """Parse a single line from the MPC NEOCP text format.
        
        Example line:
        CEXN6R2 100 2026 11 03.1   0.2512 +19.2102 20.5 Added Nov. 3.14 UT               4   0.02 24.3  0.015
        """
        try:
            tokens = line.split()
            if len(tokens) < 7:
                return None
                
            tdes = tokens[0]
            # score = int(tokens[1]) if tokens[1].isdigit() else 0
            
            # Date tokens: year, month, day
            year = int(tokens[2])
            month = int(tokens[3])
            day_float = float(tokens[4])
            day_int = int(day_float)
            fraction_day = day_float - day_int
            hours_val = fraction_day * 24.0
            h_int = int(hours_val)
            m_int = int((hours_val - h_int) * 60.0)
            
            try:
                disc_dt = datetime(year, month, day_int, h_int, m_int)
            except Exception:
                disc_dt = datetime.utcnow()
                
            # RA in hours -> deg
            ra_hours = float(tokens[5])
            ra_deg = (ra_hours * 15.0) % 360.0
            
            # Dec in deg
            dec_deg = float(tokens[6])
            
            # Magnitude
            v_mag = float(tokens[7]) if len(tokens) > 7 and tokens[7].replace(".", "").isdigit() else 20.0
            
            # Parse tail tokens for nObs, Arc, H
            n_obs = 3
            arc_hours = 1.0
            h_mag = None
            rate_deg_day = 1.5
            
            # Look for arc and nobs in tail
            for i, tok in enumerate(tokens[8:]):
                if tok.isdigit() and int(tok) >= 3 and n_obs == 3:
                    n_obs = int(tok)
                    # Next token is often arc in days
                    if i + 1 < len(tokens[8:]):
                        try:
                            arc_days = float(tokens[8:][i + 1])
                            arc_hours = max(0.1, arc_days * 24.0)
                        except Exception:
                            pass
                    # Following token is often H
                    if i + 2 < len(tokens[8:]):
                        try:
                            h_mag = float(tokens[8:][i + 2])
                        except Exception:
                            pass
                    break
                    
            solar_elong = calculate_solar_elongation_deg(ra_deg, dec_deg, disc_dt)
            
            return CandidateRecord(
                candidate_id=tdes,
                source_survey="MPC_NEOCP",
                discovery_date=disc_dt,
                apparent_ra_deg=ra_deg,
                apparent_dec_deg=dec_deg,
                v_mag=v_mag,
                h_mag=h_mag,
                motion_rate_deg_day=rate_deg_day,
                solar_elongation_deg=solar_elong,
                arc_length_hours=arc_hours,
                n_observations=n_obs,
                rms_residual_arcsec=0.28,
                orbital_elements=OrbitalElements(neo_family="NEOCP_UNCONFIRMED"),
                raw_metadata={"raw_line": line}
            )
        except Exception as e:
            logger.debug(f"Failed parsing line '{line}': {e}")
            return None
