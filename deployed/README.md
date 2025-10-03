# Deployed V7P3R Engine Versions

This directory contains the actual deployed versions of the V7P3R chess engine that are currently running in production environments.

## Current Production Deployment

### V7P3R v12.2 - Lichess Deployment
**Status**: 🟢 **LIVE ON LICHESS**  
**Deployed**: October 2025  
**Performance**: Tournament verified, 23.5/40 vs regression suite  
**Location**: `/v12.2/` folder  

**Deployment Notes**:
- Stable in production tournaments
- Tournament-grade UCI compliance  
- Consistent performance vs v10.8, v12.0, v12.1
- Source code and executable backed up in deployed/v12.2/

## Version History

| Version | Status | Deployment | Notes |
|---------|---------|------------|-------|
| **v12.2** | 🟢 **LIVE** | Lichess Production | Stable, tournament-ready (23.5/40 vs regression) |
| v12.3 | 🔴 **FAILED** | Severe Performance Issues | 0.0/10 vs v12.2 - DO NOT DEPLOY |

## Deployment Protocol

1. **Never modify deployed engine files directly**
2. **Test all changes in main development branch first**
3. **Only promote to deployed after comprehensive validation**
4. **Maintain deployment logs for rollback capability**

## Quick Deployment Commands

```powershell
# Copy current stable version to deployed
Copy-Item -Path "v7p3r*.py" -Destination "deployed/v12.2/"

# Backup before deployment
Copy-Item -Path "deployed/v12.2" -Destination "deployed/v12.2_backup_$(Get-Date -Format 'yyyyMMdd')" -Recurse

# Deploy new version (example)
Copy-Item -Path "v7p3r*.py" -Destination "deployed/v12.3/"
```

## Current Lichess Bot Status
- **Account**: V7P3R-v12.2 (or current bot name)
- **Engine**: V7P3R v12.2
- **Performance**: Monitoring via game records
- **Health**: Check `/game_records/` for recent performance data
