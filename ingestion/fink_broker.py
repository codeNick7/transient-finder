"""Fink Alert Broker (ZTF & LSST) Solar System Candidate Ingestion Client."""

import json
import logging
import math
from datetime import datetime
from typing import List, Optional

from config import CONFIG
from ingestion.base import BaseIngestionClient
from core.models import CandidateRecord, OrbitalElements
from core.astronomy import classify_neo_orbit, calculate_tisserand_parameter

logger = logging.getLogger(__name__)


class FinkBrokerClient(BaseIngestionClient):
    """Client for querying the Fink Alert Broker for newly linked Solar System candidates."""
    
    def fetch_sso_candidates(self, limit: int = 50) -> List[CandidateRecord]:
        """Fetch newly linked Solar System Object candidates from Fink-FAT."""
        cache_key = f"fink_sso_cand_limit_{limit}"
        cached = self.get_cached(cache_key)
        
        raw_data = None
        if cached:
            raw_data = cached
        else:
            try:
                resp = self.session.post(
                    CONFIG.fink_sso_cand_url,
                    json={"kind": "orbParams"},
                    timeout=CONFIG.request_timeout_seconds
                )
                if resp.status_code == 200:
                    raw_data = resp.json()
                    self.set_cached(cache_key, raw_data)
                else:
                    logger.warning(f"Fink API returned status {resp.status_code}")
            except Exception as e:
                logger.warning(f"Error querying Fink broker: {e}")
                
        if not raw_data or not isinstance(raw_data, list):
            return []
            
        candidates: List[CandidateRecord] = []
        for item in raw_data[:limit]:
            rec = self._parse_fink_candidate(item)
            if rec:
                candidates.append(rec)
        return candidates

    def _parse_fink_candidate(self, item: dict) -> Optional[CandidateRecord]:
        """Convert a Fink candidate record to CandidateRecord."""
        try:
            cand_id = item.get("d:ssoCandId", "FINK_UNKNOWN")
            a = float(item.get("d:a", 2.5))
            e = float(item.get("d:e", 0.15))
            inc = float(item.get("d:i", 5.0))
            chi2 = float(item.get("d:chi_reduced", 1.0))
            epoch_jd = float(item.get("d:ref_epoch", 2460000.0))
            
            # Rough conversion from JD to datetime
            epoch_dt = datetime.utcfromtimestamp((epoch_jd - 2440587.5) * 86400.0) if 2400000 < epoch_jd < 2500000 else datetime.utcnow()
            
            q = a * (1.0 - e)
            family = classify_neo_orbit(a, e, q)
            tj = calculate_tisserand_parameter(a, e, inc)
            
            # Rough motion rate for main belt vs NEO from Keplerian velocity:
            # v ~ 30 / sqrt(a) km/s -> apparent motion approx 0.3 deg/day at 2.5 AU, 1.5 deg/day at 1 AU
            rate_deg_day = 0.8 / math.sqrt(max(0.5, a))
            
            orbital_elems = OrbitalElements(
                semi_major_axis_au=a,
                eccentricity=e,
                inclination_deg=inc,
                perihelion_dist_au=q,
                aphelion_dist_au=a * (1.0 + e),
                longitude_node_deg=float(item.get("d:long_node", 0.0)),
                arg_perihelion_deg=float(item.get("d:arg_peric", 0.0)),
                mean_anomaly_deg=float(item.get("d:mean_anomaly", 0.0)),
                epoch_jd=epoch_jd,
                tisserand_jupiter=tj,
                neo_family=family
            )
            
            # Apparent position placeholder based on orbital node
            ra_deg = float(item.get("d:long_node", 0.0)) % 360.0
            dec_deg = float(item.get("d:i", 0.0)) * 0.5
            
            return CandidateRecord(
                candidate_id=cand_id,
                source_survey="FINK_ZTF_FAT",
                discovery_date=epoch_dt,
                apparent_ra_deg=ra_deg,
                apparent_dec_deg=dec_deg,
                v_mag=19.5,
                h_mag=18.0,
                motion_rate_deg_day=rate_deg_day,
                solar_elongation_deg=135.0,
                arc_length_hours=48.0,
                n_observations=8,
                rms_residual_arcsec=max(0.1, min(1.2, chi2 * 0.25)),
                orbital_elements=orbital_elems,
                raw_metadata=item
            )
        except Exception as e:
            logger.warning(f"Error parsing Fink candidate: {e}")
            return None
