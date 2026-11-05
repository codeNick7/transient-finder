"""Ingestion package providing access to JPL Scout, MPC NEOCP, Fink Broker, and JPL CAD."""

from ingestion.base import BaseIngestionClient
from ingestion.jpl_scout import JPLScoutClient
from ingestion.mpc_neocp import MPCNEOCPClient
from ingestion.fink_broker import FinkBrokerClient
from ingestion.jpl_cad import JPLHistoricalCADClient
from ingestion.fixtures import get_ground_truth_fixtures, get_live_candidates_snapshot

__all__ = [
    "BaseIngestionClient",
    "JPLScoutClient",
    "MPCNEOCPClient",
    "FinkBrokerClient",
    "JPLHistoricalCADClient",
    "get_ground_truth_fixtures",
    "get_live_candidates_snapshot",
]
