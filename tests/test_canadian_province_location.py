"""FW17 (2026-10-07): bare Canadian province names in the location field."""

import pytest

from applypilot.scoring.scorer import _check_ineligible


def _loc(location):
    return _check_ineligible({"title": "IT Support Specialist", "location": location, "full_description": ""})


@pytest.mark.parametrize(
    "location",
    [
        "Ontario - Remote",
        "Ontario Remote Work",
        "remote, Ontario - Remote",
        "Toronto, Ontario",
        "Montreal, Quebec",
        "Montréal, Québec",
        "Calgary, Alberta",
        "Vancouver, British Columbia",
        "Halifax, Nova Scotia",
        "Moncton, New Brunswick",
        "Winnipeg, Manitoba",
    ],
)
def test_canadian_province_rejected(location):
    assert _loc(location) is not None


@pytest.mark.parametrize(
    "location",
    [
        "Ontario, CA",
        "Ontario, California",
        "Ontario, OR",
        "Ontario, NY 14519",
        "New Brunswick, NJ",
        "New Brunswick, New Jersey",
        "Alberta, VA",
        "Ontario County, NY",
        "Greensboro, NC",
        # Multi-location strings that also list the US stay eligible (#108 guard).
        "Ontario - Remote; United States - Remote",
    ],
)
def test_us_places_and_us_also_listed_not_rejected(location):
    assert _loc(location) is None
