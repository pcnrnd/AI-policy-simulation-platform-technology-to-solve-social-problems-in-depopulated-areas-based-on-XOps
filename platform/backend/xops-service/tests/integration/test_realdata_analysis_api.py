"""The retired diagnosis route keeps auth and never invokes its historical engine."""
from src.realdata import analysis

URL = "/api/v3/realdata/analysis"


def test_retired_diagnosis_keeps_authentication(client):
    assert client.get(URL).status_code == 401


def test_retired_diagnosis_returns_410_without_reading_data(client, auth_headers, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("퇴역한 진단 엔진이 호출됨")
    monkeypatch.setattr(analysis, "analyze", forbidden)
    response = client.get(URL, params={"region":"namwon","dong_code":"all","base_ym":202312}, headers={**auth_headers,"X-Data-Mode":"real"})
    assert response.status_code == 410
    assert "종료" in response.json()["detail"]
    assert client.get(URL, headers=auth_headers).status_code == 410


def test_mock_cannot_enter_retired_realdata_route(client, auth_headers):
    assert client.get(URL,headers={**auth_headers,"X-Data-Mode":"mock"}).status_code == 404


def test_retired_route_is_marked_in_openapi(client):
    operation = client.app.openapi()["paths"][URL]["get"]
    assert operation["deprecated"] is True
    assert "410" in operation["responses"]
