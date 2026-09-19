#!/usr/bin/env bash
#
# Tests connectivity between WSL2 and Microsoft Foundry Local running on the Windows 11 host.
#

set -euo pipefail

PORT="${FOUNDRY_PORT:-5272}"
HOST_IP=$(ip route show default | awk '{print $3}' || echo "127.0.0.1")
TARGET_URL="http://${HOST_IP}:${PORT}/v1/models"

echo "============================================================"
echo " Testing WSL2 -> Windows Host Foundry Local Connectivity"
echo "============================================================"
echo "Host IP (detected): ${HOST_IP}"
echo "Target URL:         ${TARGET_URL}"
echo "------------------------------------------------------------"

if curl -s -m 5 "${TARGET_URL}" >/dev/null; then
    echo "✅ SUCCESS: Successfully reached Microsoft Foundry Local on Windows Host!"
    echo ""
    curl -s "${TARGET_URL}" | jq . || curl -s "${TARGET_URL}"
else
    echo "❌ FAILED: Unable to connect to ${TARGET_URL}"
    echo "Checklist:"
    echo " 1. Ensure 'foundry server start' is running on the Windows host."
    echo " 2. Ensure Windows Firewall rule allows inbound TCP port ${PORT}."
    echo " 3. Check that Windows Foundry server is bound to 0.0.0.0, not solely 127.0.0.1."
fi
echo "============================================================"
