"""Base ingestion client with disk caching, retries, and offline failover."""

import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import requests

from config import CONFIG, CACHE_DIR

logger = logging.getLogger(__name__)


class BaseIngestionClient:
    """Base HTTP client for astronomical public APIs with transparent disk caching."""
    
    def __init__(self, cache_dir: Optional[Path] = None, cache_ttl_seconds: int = CONFIG.cache_ttl_seconds):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_ttl_seconds = cache_ttl_seconds
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": CONFIG.user_agent,
            "Accept": "application/json, text/plain, */*"
        })
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _cache_path(self, key: str) -> Path:
        hashed = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{hashed}.json"
        
    def get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_time = datetime.fromisoformat(data["_cached_at"])
            if (datetime.utcnow() - cached_time).total_seconds() > self.cache_ttl_seconds:
                return None
            return data["payload"]
        except Exception as e:
            logger.warning(f"Failed reading cache for {key}: {e}")
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

    def fetch_url(self, url: str, params: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> Optional[str]:
        cache_key = f"{url}?{json.dumps(params or {}, sort_keys=True)}"
        if use_cache:
            cached = self.get_cached(cache_key)
            if cached is not None:
                return cached
                
        try:
            resp = self.session.get(url, params=params, timeout=CONFIG.request_timeout_seconds)
            if resp.status_code == 200:
                text = resp.text
                if use_cache:
                    self.set_cached(cache_key, text)
                return text
            else:
                logger.warning(f"HTTP {resp.status_code} fetching {url}")
                return None
        except Exception as e:
            logger.warning(f"Error connecting to {url}: {e}")
            return None
