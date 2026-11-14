from datetime import date

from download_data import current_season_start, season_code, season_name


def test_season_code():
    assert season_code(2000) == "0001"
    assert season_code(2026) == "2627"
    assert season_code(2099) == "9900"


def test_season_name():
    assert season_name(2024) == "2024-25"
    assert season_name(2099) == "2099-00"


def test_current_season_starts_in_august():
    assert current_season_start(date(2026, 11, 14)) == 2026
    assert current_season_start(date(2027, 5, 20)) == 2026
    assert current_season_start(date(2027, 7, 31)) == 2026
    assert current_season_start(date(2027, 8, 1)) == 2027
