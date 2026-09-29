# Proxmox MCP server

This MCP server lets Claude inspect and manage the Proxmox VE host at
`https://192.168.0.67:8006` over the Proxmox REST API.

> The server has to run on a machine that can reach `192.168.0.67`: your laptop
> or desktop on the home LAN, or a VM/LXC on the Proxmox box itself.

## 1. Create an API token in Proxmox

1. Open `https://192.168.0.67:8006`, then go to **Datacenter → Permissions → Users**
   and add a user such as `claude@pve`. You can skip this and use `root@pam`,
   but a dedicated user is safer.
2. Go to **Datacenter → Permissions → API Tokens → Add**. Choose the user, set
   Token ID `mcp`, and untick **Privilege Separation** so the token inherits the
   user's permissions. Copy the secret, because Proxmox shows it only once.
3. Go to **Datacenter → Permissions → Add → User Permission**, set path `/`, and
   pick a role for the user:
   - `PVEAuditor` gives read-only access.
   - `PVEVMAdmin` also allows power and snapshot actions.

## 2. Install

```bash
git clone https://github.com/tman9590/Vision && cd Vision
pip install -r proxmox_mcp/requirements.txt
```

## 3. Configure credentials

Put these variables in your shell profile (`~/.zshrc` / `~/.bashrc`). Don't commit them:

```bash
export PROXMOX_TOKEN_ID='claude@pve!mcp'
export PROXMOX_TOKEN_SECRET='xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'
# optional
export PROXMOX_HOST=192.168.0.67        # default
export PROXMOX_VERIFY_SSL=false         # default: Proxmox uses a self-signed cert
                                        # (or /path/to/pve-root-ca.pem to verify it)
export PROXMOX_ALLOW_WRITE=false        # set to true to enable start/stop/snapshots
```

## 4. Use it

**Claude Code:** the repo's `.mcp.json` registers the server. Run `claude` from
the repo root, approve the `proxmox` server, and check it with `/mcp`.

**Claude Code, from any directory:**

```bash
claude mcp add proxmox --scope user \
  -e PROXMOX_TOKEN_ID='claude@pve!mcp' -e PROXMOX_TOKEN_SECRET='...' \
  -- python3 /absolute/path/to/Vision/proxmox_mcp/server.py
```

**Claude Desktop:** add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "proxmox": {
      "command": "python3",
      "args": ["/absolute/path/to/Vision/proxmox_mcp/server.py"],
      "env": {
        "PROXMOX_HOST": "192.168.0.67",
        "PROXMOX_TOKEN_ID": "claude@pve!mcp",
        "PROXMOX_TOKEN_SECRET": "..."
      }
    }
  }
}
```

## Tools

| Tool | What it does |
|---|---|
| `get_version`, `list_nodes`, `get_node_status` | Host and node information |
| `list_guests`, `get_guest_status`, `get_guest_config` | VMs (`qemu`) and containers (`lxc`) |
| `list_storage`, `list_snapshots`, `list_tasks`, `get_task_status` | Storage, snapshots, and task history |
| `guest_power` ✏️ | start / shutdown / stop / reboot / suspend / resume |
| `create_snapshot` ✏️ | Snapshot a VM or container |

The tools marked ✏️ work only when `PROXMOX_ALLOW_WRITE=true` is set and the
token's role allows the action.

## Quick check without Claude

```bash
curl -k -H "Authorization: PVEAPIToken=$PROXMOX_TOKEN_ID=$PROXMOX_TOKEN_SECRET" \
  https://192.168.0.67:8006/api2/json/version
```
