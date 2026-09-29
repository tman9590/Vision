#!/usr/bin/env python3
"""
Proxmox VE MCP server.

Exposes a Proxmox VE host (default https://192.168.0.67:8006) to MCP clients
such as Claude Code / Claude Desktop over stdio.

Configuration (environment variables):
    PROXMOX_HOST          Host or IP of the Proxmox node   (default: 192.168.0.67)
    PROXMOX_PORT          API port                          (default: 8006)
    PROXMOX_TOKEN_ID      API token id, e.g. root@pam!claude (required)
    PROXMOX_TOKEN_SECRET  API token secret (UUID)            (required)
    PROXMOX_VERIFY_SSL    true | false | /path/to/ca.pem     (default: false,
                          since Proxmox ships a self-signed certificate)
    PROXMOX_ALLOW_WRITE   Set to "true" to enable power/snapshot actions
                          (default: false — read-only)

Run:
    python -m proxmox_mcp.server
"""

from __future__ import annotations

import os
from typing import Any

import requests
import urllib3
from mcp.server.fastmcp import FastMCP


class ProxmoxClient:
    """Minimal Proxmox VE REST client using API-token auth."""

    def __init__(
        self,
        host: str,
        port: int,
        token_id: str,
        token_secret: str,
        verify: bool | str = False,
        timeout: float = 15.0,
    ):
        self.base_url = f"https://{host}:{port}/api2/json"
        self.verify = verify
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"PVEAPIToken={token_id}={token_secret}"
        if verify is False:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def request(self, method: str, path: str, **params: Any) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        kwargs: dict[str, Any] = {"verify": self.verify, "timeout": self.timeout}
        if method == "GET":
            kwargs["params"] = params
        else:
            kwargs["data"] = params
        resp = self.session.request(method, url, **kwargs)
        if not resp.ok:
            raise RuntimeError(
                f"Proxmox API {method} /{path.lstrip('/')} failed: "
                f"HTTP {resp.status_code} {resp.reason} {resp.text.strip()}"
            )
        return resp.json().get("data")

    def get(self, path: str, **params: Any) -> Any:
        return self.request("GET", path, **params)

    def post(self, path: str, **params: Any) -> Any:
        return self.request("POST", path, **params)


def _parse_verify(value: str | None) -> bool | str:
    if value is None or value.strip() == "":
        return False
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    return value  # path to a CA bundle


def client_from_env() -> ProxmoxClient:
    token_id = os.environ.get("PROXMOX_TOKEN_ID", "")
    token_secret = os.environ.get("PROXMOX_TOKEN_SECRET", "")
    if not token_id or not token_secret:
        raise RuntimeError(
            "PROXMOX_TOKEN_ID and PROXMOX_TOKEN_SECRET must be set. "
            "Create a token in the Proxmox UI: Datacenter → Permissions → API Tokens."
        )
    return ProxmoxClient(
        host=os.environ.get("PROXMOX_HOST", "192.168.0.67"),
        port=int(os.environ.get("PROXMOX_PORT", "8006")),
        token_id=token_id,
        token_secret=token_secret,
        verify=_parse_verify(os.environ.get("PROXMOX_VERIFY_SSL")),
    )


def _writes_enabled() -> bool:
    return os.environ.get("PROXMOX_ALLOW_WRITE", "").strip().lower() in ("1", "true", "yes", "on")


def _require_write() -> None:
    if not _writes_enabled():
        raise PermissionError(
            "Write actions are disabled. Set PROXMOX_ALLOW_WRITE=true to enable them."
        )


def _check_kind(kind: str) -> str:
    if kind not in ("qemu", "lxc"):
        raise ValueError("kind must be 'qemu' (VM) or 'lxc' (container)")
    return kind


mcp = FastMCP("proxmox")
_client: ProxmoxClient | None = None


def client() -> ProxmoxClient:
    global _client
    if _client is None:
        _client = client_from_env()
    return _client


# ── Read-only tools ──────────────────────────────────────────────────────────


@mcp.tool()
def get_version() -> dict:
    """Return the Proxmox VE version of the connected host."""
    return client().get("version")


@mcp.tool()
def list_nodes() -> list:
    """List cluster nodes with status, CPU, memory and uptime."""
    return client().get("nodes")


@mcp.tool()
def get_node_status(node: str) -> dict:
    """Detailed status of one node (CPU, memory, load, kernel, PVE version)."""
    return client().get(f"nodes/{node}/status")


@mcp.tool()
def list_guests(kind: str | None = None) -> list:
    """
    List all VMs and LXC containers across the cluster.

    Args:
        kind: optional filter, 'qemu' for VMs or 'lxc' for containers.
    """
    guests = client().get("cluster/resources", type="vm")
    if kind:
        _check_kind(kind)
        guests = [g for g in guests if g.get("type") == kind]
    return guests


@mcp.tool()
def get_guest_status(node: str, vmid: int, kind: str = "qemu") -> dict:
    """Current status of a VM ('qemu') or container ('lxc')."""
    return client().get(f"nodes/{node}/{_check_kind(kind)}/{vmid}/status/current")


@mcp.tool()
def get_guest_config(node: str, vmid: int, kind: str = "qemu") -> dict:
    """Configuration (CPU, memory, disks, network) of a VM or container."""
    return client().get(f"nodes/{node}/{_check_kind(kind)}/{vmid}/config")


@mcp.tool()
def list_storage(node: str) -> list:
    """List storage pools on a node with usage."""
    return client().get(f"nodes/{node}/storage")


@mcp.tool()
def list_snapshots(node: str, vmid: int, kind: str = "qemu") -> list:
    """List snapshots of a VM or container."""
    return client().get(f"nodes/{node}/{_check_kind(kind)}/{vmid}/snapshot")


@mcp.tool()
def list_tasks(node: str, limit: int = 20) -> list:
    """Recent tasks on a node (backups, starts, migrations, ...)."""
    return client().get(f"nodes/{node}/tasks", limit=limit)


@mcp.tool()
def get_task_status(node: str, upid: str) -> dict:
    """Status of a task by its UPID (returned by start/stop/snapshot actions)."""
    return client().get(f"nodes/{node}/tasks/{upid}/status")


# ── Write tools (require PROXMOX_ALLOW_WRITE=true) ───────────────────────────

_POWER_ACTIONS = ("start", "shutdown", "stop", "reboot", "suspend", "resume")


@mcp.tool()
def guest_power(node: str, vmid: int, action: str, kind: str = "qemu") -> str:
    """
    Change the power state of a VM or container. Requires PROXMOX_ALLOW_WRITE=true.

    Args:
        action: one of start, shutdown (graceful), stop (hard), reboot, suspend, resume.
    Returns the task UPID.
    """
    _require_write()
    if action not in _POWER_ACTIONS:
        raise ValueError(f"action must be one of {', '.join(_POWER_ACTIONS)}")
    return client().post(f"nodes/{node}/{_check_kind(kind)}/{vmid}/status/{action}")


@mcp.tool()
def create_snapshot(
    node: str, vmid: int, snapname: str, description: str = "", kind: str = "qemu"
) -> str:
    """Create a snapshot of a VM or container. Requires PROXMOX_ALLOW_WRITE=true."""
    _require_write()
    params = {"snapname": snapname}
    if description:
        params["description"] = description
    return client().post(f"nodes/{node}/{_check_kind(kind)}/{vmid}/snapshot", **params)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
