"""Base ingestion client with disk caching, retries, and offline failover."""

import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import CONFIG, CACHE_DIR

logger = logging.getLogger(__name__)


class BaseIngestionClient:
    """Base HTTP client for astronomical public APIs with transparent disk caching and stale-while-revalidate failover."""
    
    def __init__(self, cache_dir: Optional[Path] = None, cache_ttl_seconds: int = CONFIG.cache_ttl_seconds):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_ttl_seconds = cache_ttl_seconds
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": CONFIG.user_agent,
            "Accept": "application/json, text/plain, */*"
        })
        
        # Configure automatic connection retry with exponential backoff
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _cache_path(self, key: str) -> Path:
        hashed = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{hashed}.json"
        
    def get_cached(self, key: str, allow_stale: bool = False) -> Optional[Any]:
        """Retrieve payload from disk cache. If allow_stale is True, returns payload regardless of age."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_time = datetime.fromisoformat(data["_cached_at"])
            age_seconds = (datetime.utcnow() - cached_time).total_seconds()
            if not allow_stale and age_seconds > self.cache_ttl_seconds:
                return None
            return data["payload"]
        except Exception as e:
            logger.warning(f"Failed reading cache for {key}: {e}")
            return None

    def get_cache_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cache timestamp and age in seconds."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_time = datetime.fromisoformat(data["_cached_at"])
            age_seconds = (datetime.utcnow() - cached_time).total_seconds()
            return {
                "cached_at": cached_time,
                "age_seconds": age_seconds,
                "age_minutes": age_seconds / 60.0
            }
        except Exception:
            return None

    def set_cached(self, key: str, payload: Any) -> None:
        path = self._cache_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "_cached_at": datetime.utcnow().isoformat(),
                    "payload": payload
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed writing cache for {key}: {e}")

    def fetch_url(self, url: str, params: Optional[Dict[str, Any]] = None, use_cache: bool = True, force_refresh: bool = False) -> Optional[str]:
        """Fetch URL with transparent disk caching and stale-while-revalidate failover."""
        cache_key = f"{url}?{json.dumps(params or {}, sort_keys=True)}"
        
        # 1. Check fresh cache if not forcing refresh
        if use_cache and not force_refresh:
            cached = self.get_cached(cache_key, allow_stale=False)
            if cached is not None:
                return cached
                
        # 2. Attempt network request
        try:
            resp = self.session.get(url, params=params, timeout=CONFIG.request_timeout_seconds)
            if resp.status_code == 200:
                text = resp.text
                if use_cache:
                    self.set_cached(cache_key, text)
                return text
            else:
                logger.warning(f"HTTP {resp.status_code} fetching {url}")
        except Exception as e:
            logger.warning(f"Network error or timeout connecting to {url}: {e}")

        # 3. Failover: If network failed or timed out, fall back to stale cache rather than failing
        if use_cache:
            stale_cached = self.get_cached(cache_key, allow_stale=True)
            if stale_cached is not None:
                meta = self.get_cache_metadata(cache_key)
                age_m = meta["age_minutes"] if meta else 0
                logger.info(f"Using stale cached telemetry for {url} ({age_m:.1f} mins old) following network failure.")
                return stale_cached

        return None
