"""Astronomical kinematics, orbital mechanics, coordinate conversions, and geometry."""

import math
from datetime import datetime, timezone
from typing import Tuple, Optional


def datetime_to_jd(dt: datetime) -> float:
    """Convert datetime to Julian Date (JD)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    
    year = dt.year
    month = dt.month
    day = dt.day + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + dt.microsecond / 3.6e9) / 24.0

    if month <= 2:
        year -= 1
        month += 12

    A = int(year / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    return jd


def great_circle_distance_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    """Calculate the great-circle angular distance between two celestial points in degrees (Haversine formula)."""
    r1, d1 = math.radians(ra1_deg), math.radians(dec1_deg)
    r2, d2 = math.radians(ra2_deg), math.radians(dec2_deg)
    
    dlat = d2 - d1
    dlon = r2 - r1
    
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(d1) * math.cos(d2) * (math.sin(dlon / 2.0) ** 2))
    a = max(0.0, min(1.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return math.degrees(c)


def get_sun_coordinates(jd: float) -> Tuple[float, float]:
    """Calculate low-precision geocentric ecliptic coordinates of the Sun in (RA deg, Dec deg) for solar elongation."""
    # Days since J2000.0
    n = jd - 2451545.0
    # Mean longitude of Sun
    L = (280.460 + 0.9856474 * n) % 360.0
    # Mean anomaly of Sun
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    # Ecliptic longitude
    lambda_sun = math.radians((L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360.0)
    # Obliquity of the ecliptic
    epsilon = math.radians(23.439 - 0.0000004 * n)
    
    # Right Ascension & Declination
    sin_delta = math.sin(epsilon) * math.sin(lambda_sun)
    delta = math.asin(sin_delta)
    
    y = math.cos(epsilon) * math.sin(lambda_sun)
    x = math.cos(lambda_sun)
    alpha = math.atan2(y, x)
    
    ra_deg = (math.degrees(alpha) + 360.0) % 360.0
    dec_deg = math.degrees(delta)
    return ra_deg, dec_deg


def calculate_solar_elongation_deg(ra_deg: float, dec_deg: float, obs_time: datetime) -> float:
    """Compute the solar elongation angle between the candidate object and the Sun in degrees."""
    jd = datetime_to_jd(obs_time)
    sun_ra, sun_dec = get_sun_coordinates(jd)
    return great_circle_distance_deg(ra_deg, dec_deg, sun_ra, sun_dec)


def calculate_tisserand_parameter(semi_major_axis_au: float, eccentricity: float, inclination_deg: float, a_jupiter: float = 5.2044) -> Optional[float]:
    """Calculate the Tisserand parameter of an orbit with respect to Jupiter (T_J).
    
    T_J = (a_J / a) + 2 * cos(i) * sqrt((a / a_J) * (1 - e^2))
    """
    if semi_major_axis_au is None or semi_major_axis_au <= 0:
        return None
    if eccentricity is None or eccentricity < 0 or eccentricity >= 1.0:
        # Parabolic / hyperbolic or invalid
        return None
    
    inc_rad = math.radians(inclination_deg or 0.0)
    term1 = a_jupiter / semi_major_axis_au
    sqrt_val = (semi_major_axis_au / a_jupiter) * (1.0 - eccentricity ** 2)
    if sqrt_val < 0:
        return None
    term2 = 2.0 * math.cos(inc_rad) * math.sqrt(sqrt_val)
    return float(term1 + term2)


def classify_neo_orbit(semi_major_axis_au: Optional[float], eccentricity: Optional[float], perihelion_au: Optional[float] = None) -> str:
    """Classify small body orbital family based on standard dynamical definitions.
    
    - Atira: a < 1.0 AU, Q < 0.983 AU (interior to Earth's orbit)
    - Aten: a < 1.0 AU, Q >= 0.983 AU (Earth-crossing, semi-major axis smaller than Earth)
    - Apollo: a >= 1.0 AU, q <= 1.017 AU (Earth-crossing, semi-major axis larger than Earth)
    - Amor: a > 1.0 AU, 1.017 AU < q < 1.3 AU (Earth-approaching, outside Earth's orbit)
    - Main Belt Asteroid (MBA): q >= 1.3 AU
    - Comet / Hyperbolic: e >= 1.0 or high inclination cometary
    """
    if semi_major_axis_au is None or eccentricity is None:
        return "UNKNOWN_SHORT_ARC"
    
    if eccentricity >= 1.0:
        return "HYPERBOLIC_OR_INTERSTELLAR"
        
    a = semi_major_axis_au
    e = eccentricity
    q = perihelion_au if perihelion_au is not None else a * (1.0 - e)
    Q = a * (1.0 + e)
    
    if a < 1.0 and Q < 0.983:
        return "ATIRA"
    elif a < 1.0 and Q >= 0.983:
        return "ATEN"
    elif a >= 1.0 and q <= 1.017:
        return "APOLLO"
    elif a > 1.0 and 1.017 < q < 1.3:
        return "AMOR"
    elif q >= 1.3:
        return "MAIN_BELT_ASTEROID"
    else:
        return "NON_NEO_OUTER"


def check_potentially_hazardous(moid_au: Optional[float], h_mag: Optional[float]) -> bool:
    """Check if object meets NASA/JPL Potentially Hazardous Asteroid (PHA) criteria:
    MOID <= 0.05 AU and H <= 22.0 (approx. diameter > 140 meters).
    """
    if moid_au is None or h_mag is None:
        return False
    return (moid_au <= 0.05) and (h_mag <= 22.0)
