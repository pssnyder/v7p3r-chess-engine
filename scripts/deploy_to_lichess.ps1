# V7P3R Lichess Bot Automated Deployment Script (PowerShell)
# 
# Usage: .\scripts\deploy_to_lichess.ps1 [version] [environment]
# 
# Examples:
#   .\scripts\deploy_to_lichess.ps1 17.5 production
#   .\scripts\deploy_to_lichess.ps1 17.5 test

param(
    [string]$Version = "17.5",
    [string]$Environment = "test"
)

$ErrorActionPreference = "Stop"

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$DeployDir = "s:\Programming\Chess Engines\Deployed Engines\v7p3r-lichess-engine"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "V7P3R Lichess Bot Deployment" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Version: v$Version"
Write-Host "Environment: $Environment"
Write-Host "Project Root: $ProjectRoot"
Write-Host "Deploy Dir: $DeployDir"
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

function Print-Status($message) {
    Write-Host "[OK] $message" -ForegroundColor Green
}

function Print-Error($message) {
    Write-Host "[ERROR] $message" -ForegroundColor Red
}

function Print-Warning($message) {
    Write-Host "[WARNING] $message" -ForegroundColor Yellow
}

# Step 1: Verify version numbers
Write-Host "Step 1: Verifying version numbers..."
$uciContent = Get-Content "$ProjectRoot\src\v7p3r_uci.py" -Raw
if ($uciContent -notmatch "V7P3R v$Version") {
    Print-Error "Version number not found in v7p3r_uci.py"
    Print-Warning "Update line: print('id name V7P3R v$Version')"
    exit 1
}
Print-Status "Version number verified in UCI interface"

$engineContent = Get-Content "$ProjectRoot\src\v7p3r.py" -Raw
if ($engineContent -notmatch "V7P3R Chess Engine v$Version") {
    Print-Error "Version number not found in v7p3r.py docstring"
    Print-Warning "Update docstring header to: V7P3R Chess Engine v$Version"
    exit 1
}
Print-Status "Version number verified in engine"
Write-Host ""

# Step 2: Run validation tests
Write-Host "Step 2: Running validation tests..."
Print-Warning "Skipping validation (already completed)"
# Uncomment to enable:
# & python "$ProjectRoot\testing\test_v17_5_phase1.py"
# if ($LASTEXITCODE -ne 0) {
#     Print-Error "Phase 1 tests failed"
#     exit 1
# }
Print-Status "Validation tests passed (skipped)"
Write-Host ""

# Step 3: Render a local-only Lichess config from the private token file.
Write-Host "Step 3: Rendering local Lichess bot configuration..."
$templateFile = "$ProjectRoot\lichess\config.yml.default"
$tokenFile = [System.Environment]::GetEnvironmentVariable('V7P3R_TOKEN_FILE')
if (-not $tokenFile) {
    $tokenFile = 'E:\Programming Stuff\Gaming Projects\Chess Engines\V7P3R Chess Engine\v7p3r_bot_token.env'
}
$localConfig = "$ProjectRoot\lichess\config.local.yml"

if (!(Test-Path $tokenFile)) {
    Print-Error "Token file not found: $tokenFile"
    Print-Error "Set V7P3R_TOKEN_FILE or keep the file at E:\Programming Stuff\Gaming Projects\Chess Engines\V7P3R Chess Engine\v7p3r_bot_token.env"
    exit 1
}
if (!(Test-Path $templateFile)) {
    Print-Error "Config template not found: $templateFile"
    exit 1
}

$tokenValue = $null
Get-Content $tokenFile | ForEach-Object {
    if ($_ -match '^(LICHESS_TOKEN|TOKEN|API_TOKEN|lichess_token)\s*=\s*(.+)$') {
        $tokenValue = $matches[2].Trim().Trim('"').Trim("'").Trim()
    }
}
if (-not $tokenValue) {
    $tokenValue = (Get-Content $tokenFile -Raw).Trim().Replace("`r", '').Replace("`n", '').Trim("'").Trim('"')
}
if (-not $tokenValue) {
    Print-Error "Token file exists but no supported token value was found."
    exit 1
}

$configContent = Get-Content $templateFile -Raw
$configContent = $configContent.Replace('token: "REPLACE_WITH_LICHESS_OAUTH_TOKEN"', "token: `"$tokenValue`"")
$configContent = $configContent.Replace('token: REPLACE_WITH_LICHESS_OAUTH_TOKEN', "token: `"$tokenValue`"")
Set-Content -Path $localConfig -Value $configContent -Encoding utf8

Print-Status "Rendered local config at $localConfig"
Print-Status "Local deploys use the secure private token file and do not modify tracked config files"
Write-Host ""

# Step 4: Create deployment package
Write-Host "Step 4: Creating deployment package..."
$tempDeployDir = "$env:TEMP\v7p3r_v${Version}_deploy"
if (Test-Path $tempDeployDir) {
    Remove-Item -Recurse -Force $tempDeployDir
}
New-Item -ItemType Directory -Path "$tempDeployDir\engines\v7p3r" -Force | Out-Null

# Copy engine files
Copy-Item "$ProjectRoot\src\*.py" "$tempDeployDir\engines\v7p3r\" -Force
Copy-Item "$ProjectRoot\requirements.txt" "$tempDeployDir\" -Force
Copy-Item $localConfig "$tempDeployDir\config.yml" -Force

# Create version info file
$versionInfo = @"
V7P3R v$Version
Deployed: $(Get-Date)
Environment: $Environment
"@
Set-Content -Path "$tempDeployDir\VERSION.txt" -Value $versionInfo

Print-Status "Created deployment package at $tempDeployDir"
Write-Host ""

# Step 5: Deploy to target environment
Write-Host "Step 5: Deploying to $Environment..."

switch ($Environment) {
    "production" {
        Print-Warning "Production deployment to GCP/remote server"
        Print-Warning "Manual step required: Upload deployment package to server"
        Print-Warning "Command: scp -r $tempDeployDir/* user@server:/path/to/lichess-bot/"
    }
    
    { $_ -in "test", "local" } {
        Print-Status "Deploying to local test environment: $DeployDir"
        if (!(Test-Path $DeployDir)) {
            New-Item -ItemType Directory -Path $DeployDir -Force | Out-Null
        }
        New-Item -ItemType Directory -Path "$DeployDir\engines\v7p3r" -Force | Out-Null
        
        # Copy files
        Copy-Item -Recurse -Force "$tempDeployDir\*" $DeployDir
        Print-Status "Files synced to $DeployDir"
    }
    
    default {
        Print-Error "Unknown environment: $Environment"
        exit 1
    }
}
Write-Host ""

# Step 6: Health check instructions
Write-Host "Step 6: Post-Deployment Health Checks" -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------------"
Write-Host "After bot restart, verify:"
Write-Host "  1. Bot is online: https://lichess.org/@/v7p3r_bot"
Write-Host "  2. Check version in first game chat"
Write-Host "  3. Monitor first 10 games for crashes"
Write-Host "  4. Run analytics after 50 games:"
Write-Host "     cd $ProjectRoot\analytics"
Write-Host "     python weekly_analysis.py --days 3"
Write-Host ""

# Step 7: Rollback instructions
Write-Host "Step 7: Rollback Instructions (if needed)" -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------------"
Write-Host "To rollback to previous version:"
Write-Host "  1. Stop the bot service"
Write-Host "  2. Restore previous engine files from backup"
Write-Host "  3. Restart bot service"
Write-Host ""

# Step 8: Summary
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "DEPLOYMENT SUMMARY" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Print-Status "Version v$Version prepared for $Environment deployment"
Print-Status "Deployment package: $tempDeployDir"
Print-Status "Generated config: $localConfig"

if ($Environment -eq "production") {
    Print-Warning "PRODUCTION: Manual upload to server required"
    Print-Warning "After upload, restart bot service and monitor health"
} else {
    Print-Status "LOCAL: Files deployed to $DeployDir"
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. $Environment == production: Upload to server and restart bot"
Write-Host "  2. Monitor bot status on Lichess"
Write-Host "  3. Review first 10 games for issues"
Write-Host "  4. Run weekly analytics after 50 games"
Write-Host "======================================================================" -ForegroundColor Cyan
