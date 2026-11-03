"""Unit tests for astronomical coordinates, geometry, and orbital mechanics."""

import math
from datetime import datetime
from core.astronomy import (
    datetime_to_jd,
    great_circle_distance_deg,
    get_sun_coordinates,
    calculate_solar_elongation_deg,
    calculate_tisserand_parameter,
    classify_neo_orbit,
    check_potentially_hazardous
)


def test_datetime_to_jd():
    # J2000 epoch: 2000-01-01 12:00:00 UTC = JD 2451545.0
    dt = datetime(2000, 1, 1, 12, 0, 0)
    jd = datetime_to_jd(dt)
    assert abs(jd - 2451545.0) < 1e-4


def test_great_circle_distance():
    # Distance between (0, 0) and (0, 90) is 90 deg
    d = great_circle_distance_deg(0.0, 0.0, 0.0, 90.0)
    assert abs(d - 90.0) < 1e-4

    # Distance between (0, 0) and (180, 0) is 180 deg
    d2 = great_circle_distance_deg(0.0, 0.0, 180.0, 0.0)
    assert abs(d2 - 180.0) < 1e-4


def test_solar_elongation():
    # Solar elongation should be between 0 and 180 degrees
    dt = datetime(2026, 11, 3, 12, 0, 0)
    elong = calculate_solar_elongation_deg(180.0, 0.0, dt)
    assert 0.0 <= elong <= 180.0


def test_tisserand_parameter():
    # Typical asteroid in inner solar system: a = 1.5, e = 0.3, i = 5 deg
    tj = calculate_tisserand_parameter(1.5, 0.3, 5.0)
    assert tj is not None
    assert tj > 3.0  # Asteroidal criterion

    # Cometary orbit (Jupiter family)
    tj_comet = calculate_tisserand_parameter(3.5, 0.6, 15.0)
    assert tj_comet is not None
    assert 2.0 < tj_comet < 3.0


def test_classify_neo_orbit():
    # Apollo: a >= 1.0, q <= 1.017
    assert classify_neo_orbit(1.5, 0.4, 0.9) == "APOLLO"
    # Aten: a < 1.0, Q >= 0.983
    assert classify_neo_orbit(0.9, 0.2, 0.72) == "ATEN"
    # Amor: a > 1.0, 1.017 < q < 1.3
    assert classify_neo_orbit(1.8, 0.35, 1.17) == "AMOR"
    # Atira: a < 1.0, Q < 0.983
    assert classify_neo_orbit(0.8, 0.1, 0.72) == "ATIRA"
    # Main belt: q >= 1.3
    assert classify_neo_orbit(2.8, 0.15, 2.38) == "MAIN_BELT_ASTEROID"


def test_check_potentially_hazardous():
    # MOID <= 0.05 and H <= 22.0
    assert check_potentially_hazardous(0.03, 20.5) is True
    # Non-hazardous: large MOID
    assert check_potentially_hazardous(0.12, 19.0) is False
    # Non-hazardous: too small (H > 22.0)
    assert check_potentially_hazardous(0.02, 25.0) is False
