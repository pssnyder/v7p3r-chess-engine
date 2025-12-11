---
applyTo: '**'
---
Coding standards, domain knowledge, and preferences that AI should follow.

When updating code files, be cautious not to remove functional code that is needed in order for files to run.
You're aim with code updates, enhancements, and even refactors should be to preserve existing functionality.
Unless instructed to remove features, all code should return the same outputs and function the same when you are done.

## Version History Preservation
- When modifying engine code, preserve version lineage documentation in file headers
- Update VERSION_LINEAGE in `src/v7p3r.py` to include new version with description
- Maintain CHANGELOG.md entries for historical record - NEVER delete old entries
- Keep deployment_log.json complete - it's the authoritative deployment history
- Document WHY changes were made, not just WHAT changed (rationale in CHANGELOG)

## Regression Prevention
- Before removing or changing evaluation code, check CHANGELOG.md for historical context
- Review deployment_log.json to see if similar changes caused past rollbacks
- Known failure patterns documented in version_management.instructions.md - consult before major changes
- If previous version had a feature, preserve it unless explicitly instructed to remove