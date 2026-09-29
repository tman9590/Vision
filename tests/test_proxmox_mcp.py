"""Unit tests for the Proxmox MCP server (no network access required)."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp")

from proxmox_mcp import server  # noqa: E402


def _response(data, ok=True, status=200):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status
    resp.reason = "OK" if ok else "Forbidden"
    resp.text = ""
    resp.json.return_value = {"data": data}
    return resp


@pytest.fixture
def fake_client(monkeypatch):
    c = server.ProxmoxClient("192.168.0.67", 8006, "root@pam!claude", "secret")
    c.session = MagicMock()
    monkeypatch.setattr(server, "_client", c)
    return c


def test_auth_header_and_base_url():
    c = server.ProxmoxClient("192.168.0.67", 8006, "root@pam!claude", "abc")
    assert c.base_url == "https://192.168.0.67:8006/api2/json"
    assert c.session.headers["Authorization"] == "PVEAPIToken=root@pam!claude=abc"


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("", False), ("false", False), ("true", True), ("/ca.pem", "/ca.pem")],
)
def test_parse_verify(value, expected):
    assert server._parse_verify(value) == expected


def test_client_from_env_requires_token(monkeypatch):
    monkeypatch.delenv("PROXMOX_TOKEN_ID", raising=False)
    monkeypatch.delenv("PROXMOX_TOKEN_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="PROXMOX_TOKEN_ID"):
        server.client_from_env()


def test_list_guests_filters_by_kind(fake_client):
    fake_client.session.request.return_value = _response(
        [{"vmid": 100, "type": "qemu"}, {"vmid": 101, "type": "lxc"}]
    )
    assert server.list_guests("lxc") == [{"vmid": 101, "type": "lxc"}]
    method, url = fake_client.session.request.call_args.args
    assert method == "GET" and url.endswith("/cluster/resources")


def test_api_error_is_raised(fake_client):
    fake_client.session.request.return_value = _response(None, ok=False, status=403)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        server.list_nodes()


def test_power_blocked_without_write_flag(fake_client, monkeypatch):
    monkeypatch.delenv("PROXMOX_ALLOW_WRITE", raising=False)
    with pytest.raises(PermissionError):
        server.guest_power("pve", 100, "start")
    fake_client.session.request.assert_not_called()


def test_power_allowed_with_write_flag(fake_client, monkeypatch):
    monkeypatch.setenv("PROXMOX_ALLOW_WRITE", "true")
    fake_client.session.request.return_value = _response("UPID:pve:...")
    assert server.guest_power("pve", 100, "start") == "UPID:pve:..."
    method, url = fake_client.session.request.call_args.args
    assert method == "POST" and url.endswith("/nodes/pve/qemu/100/status/start")


def test_invalid_kind_rejected(fake_client):
    with pytest.raises(ValueError):
        server.get_guest_status("pve", 100, kind="docker")
