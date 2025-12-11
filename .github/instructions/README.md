# V7P3R Chess Engine - AI Instructions

This directory contains instruction files that guide AI assistants (like GitHub Copilot) when working on the V7P3R chess engine codebase.

## Instruction Files

### 1. version_management.instructions.md
**Purpose**: Define version management, testing, and deployment workflow  
**Applies To**: All files (`**`)  
**Use When**: Creating versions, deploying, or managing releases

**Key Topics**:
- Semantic versioning (MAJOR.MINOR.PATCH)
- Git workflow (main/develop/feature branches)
- Testing requirements (regression, performance, time control)
- Production deployment to GCP/Lichess
- Rollback procedures
- Documentation standards

**Critical Rules**:
- Never deploy without regression tests passing (100%)
- Never reuse version numbers
- Always create git tags for deployments
- Update CHANGELOG.md and deployment_log.json
- Test across all time controls (bullet, blitz, rapid)

---

### 2. code_preservation_instructions.instructions.md
**Purpose**: Prevent removal of functional code and regression  
**Applies To**: All files (`**`)  
**Use When**: Modifying engine code, evaluation functions, or algorithms

**Key Topics**:
- Preserving version lineage documentation
- Maintaining CHANGELOG.md history
- Consulting deployment_log.json before changes
- Understanding known failure patterns

**Critical Rules**:
- Never delete CHANGELOG.md entries
- Check deployment_log.json for rollback history
- Preserve features unless explicitly instructed to remove
- Document WHY changes were made, not just WHAT

---

### 3. error_prevention_instructions.instructions.md
**Purpose**: Prevent common PowerShell and command execution errors  
**Applies To**: All files (`**`)  
**Use When**: Running commands, executing scripts, or using terminal

**Key Topics**:
- PowerShell syntax compatibility
- Command separator usage (`;` not `&&`)
- Proper script execution in Windows

**Critical Rules**:
- Use semicolon (`;`) to chain PowerShell commands, NOT `&&`
- Use correct PowerShell syntax for version compatibility
- Avoid bash-style command separators in Windows PowerShell

---

### 4. idea_implementation_guardrails.instructions.md
**Purpose**: Guide implementation approach, complexity, and change management  
**Applies To**: All files (`**`)  
**Use When**: Implementing new features, refactoring, or making substantial changes

**Key Topics**:
- General coding guidelines (PEP 8, clean code)
- Avoiding unnecessary complexity
- Change management preparation
- Version management workflow
- Housekeeping and file organization

**Critical Rules**:
- Start with simplest solution
- Create project document before substantial changes
- Follow version management instructions
- Test incrementally with user feedback
- Store test files in `testing/` directory
- Document new functionality in `docs/`

---

## How AI Assistants Use These Instructions

When you (an AI assistant) are working on this codebase:

1. **Before ANY implementation**:
   - Read current version from `src/v7p3r.py` and `deployment_log.json`
   - Check CHANGELOG.md for recent changes and known issues
   - Review version_management.instructions.md for workflow

2. **During implementation**:
   - Follow code_preservation_instructions.md to avoid regressions
   - Use error_prevention_instructions.md for correct command syntax
   - Apply idea_implementation_guardrails.md for complexity management

3. **After implementation**:
   - Update version number following semantic versioning
   - Update CHANGELOG.md with detailed entry
   - Update deployment_log.json with testing status
   - Create git tag if approved by user

4. **Before deployment**:
   - Verify regression tests pass (100%)
   - Confirm performance benchmark meets criteria
   - Update all documentation
   - Follow deployment procedure in version_management.instructions.md

## Workflow Integration

These instructions integrate with:

- **CHANGELOG.md**: Version history and rationale
- **deployment_log.json**: Machine-readable deployment tracking
- **docs/TESTING_GUIDE.md**: Testing procedures and regression prevention
- **docs/QUICK_REFERENCE.md**: Quick commands and common tasks
- **scripts/deploy_to_production.sh**: Automated deployment script

## File Organization

```
.github/instructions/
├── README.md (this file)
├── version_management.instructions.md (workflow & deployment)
├── code_preservation_instructions.instructions.md (prevent regressions)
├── error_prevention_instructions.instructions.md (PowerShell syntax)
└── idea_implementation_guardrails.instructions.md (complexity & change mgmt)

Related files:
├── CHANGELOG.md (version history)
├── deployment_log.json (deployment tracking)
├── docs/
│   ├── TESTING_GUIDE.md (testing procedures)
│   └── QUICK_REFERENCE.md (quick commands)
└── scripts/
    └── deploy_to_production.sh (deployment automation)
```

## Key Principles

1. **Version Control First**: Always follow version management workflow
2. **Test Before Deploy**: 100% regression tests, performance benchmarks
3. **Document Everything**: CHANGELOG, deployment_log, git tags
4. **Preserve History**: Never delete old entries, learn from failures
5. **Incremental Changes**: Small steps, user validation, rollback ready

## Quick Decision Tree

**"Should I deploy this change?"**
```
Does it change engine behavior?
  YES → Follow full version management workflow
  NO → Is it a critical bug fix?
    YES → Hotfix workflow (still requires testing)
    NO → Documentation/comment change only
```

**"What version number should I use?"**
```
Breaking change or major rewrite?
  YES → Increment MAJOR (18.0.0)
  NO → New feature or evaluation improvement?
    YES → Increment MINOR (17.9.0)
    NO → Bug fix or parameter tuning?
      YES → Increment PATCH (17.8.1)
```

**"When do I update documentation?"**
```
ALWAYS:
- CHANGELOG.md (every version)
- deployment_log.json (every deployment)
- Version in v7p3r.py and v7p3r_uci.py
- Git tag (after user approval)

SOMETIMES:
- README.md (major features)
- docs/*.md (new functionality)
- Testing files (new regression tests)
```

## Getting Started as an AI Assistant

1. Read this README first
2. Review version_management.instructions.md for workflow
3. Check current version in deployment_log.json
4. Read last 3-5 CHANGELOG.md entries for context
5. Follow the instructions relevant to your task

## Updates and Maintenance

These instruction files should be updated when:
- New deployment procedures are established
- New known failure patterns are discovered
- Workflow improvements are identified
- New tools or scripts are added

Always document WHY instructions changed in CHANGELOG.md.

---

**Last Updated**: December 10, 2025  
**Current Production Version**: v17.7.0  
**Next Version**: v17.8.0 (testing)
