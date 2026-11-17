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


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def fake_server(monkeypatch, body):
    import download_data
    monkeypatch.setattr(download_data.urllib.request, "urlopen",
                        lambda request, timeout: FakeResponse(body))
    monkeypatch.setattr(download_data.time, "sleep", lambda seconds: None)


def test_a_results_csv_is_saved(monkeypatch, tmp_path):
    import download_data
    fake_server(monkeypatch, b"\xef\xbb\xbfDiv,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,01/08/25,A,B,1,0,H\n")
    monkeypatch.setattr(download_data, "RAW_DIR", tmp_path)
    assert download_data.download_results("E0", 2025) == "downloaded"
    assert (tmp_path / "E0_2526.csv").read_bytes().startswith(b"\xef\xbb\xbfDiv")
    assert list(tmp_path.glob("*.part")) == []


def test_an_error_page_is_never_cached(monkeypatch, tmp_path):
    import pytest
    import download_data
    fake_server(monkeypatch, b"<html><body>Service unavailable</body></html>")
    monkeypatch.setattr(download_data, "RAW_DIR", tmp_path)
    with pytest.raises(download_data.BadDownload, match="not a results CSV"):
        download_data.download_results("E0", 2007)
    assert list(tmp_path.iterdir()) == []


def test_fixtures_must_be_a_list_of_matches(monkeypatch, tmp_path):
    import pytest
    import download_data
    monkeypatch.setattr(download_data, "RAW_DIR", tmp_path)
    fake_server(monkeypatch, b'{"name": "x", "matches": []}')
    download_data.download_fixtures(2026)
    assert (tmp_path / "fixtures_2026-27.json").exists()
    fake_server(monkeypatch, b"Not Found")
    with pytest.raises(download_data.BadDownload):
        download_data.download_fixtures(2027)
    assert not (tmp_path / "fixtures_2027-28.json").exists()
