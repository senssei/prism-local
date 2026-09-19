<#
.SYNOPSIS
    Configures Microsoft Foundry Local on Windows 11 host to accept remote connections from WSL2.
.DESCRIPTION
    Runs on the Windows 11 host (PowerShell as Administrator).
    Configures Windows Defender firewall and starts Foundry Local listening on all interfaces.
#>

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Microsoft Foundry Local - Windows Host Bridge Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$PORT = 5272
$RULE_NAME = "FoundryLocal-WSL-Inbound"

# 1. Check Firewall Rule
$existing = Get-NetFirewallRule -DisplayName $RULE_NAME -ErrorAction SilentlyContinue
if ($null -eq $existing) {
    Write-Host "Creating Windows Firewall rule for TCP port $PORT..." -ForegroundColor Yellow
    New-NetFirewallRule -DisplayName $RULE_NAME `
                        -Direction Inbound `
                        -Protocol TCP `
                        -LocalPort $PORT `
                        -Action Allow `
                        -Description "Allow WSL2 instances to access local Foundry Local daemon" | Out-Null
    Write-Host "✅ Firewall rule created successfully." -ForegroundColor Green
} else {
    Write-Host "✅ Firewall rule '$RULE_NAME' already exists." -ForegroundColor Green
}

# 2. Display WSL2 Connection Command
$ip = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "vEthernet (WSL)" -ErrorAction SilentlyContinue).IPAddress
if ($null -eq $ip) {
    $ip = "127.0.0.1"
}

Write-Host "`nTo connect from WSL2, export the host base URL:" -ForegroundColor Yellow
Write-Host "export FOUNDRY_BASE_URL=`"http://$($ip):$PORT/v1`"" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
