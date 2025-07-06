---
applyTo: '*.py*'
---
Coding standards, domain knowledge, and preferences that AI should follow.

# Implementation Guardrails
## General Guidelines
- Ensure code is clean, readable, and maintainable.
- Follow PEP 8 style guide for Python code.
- Use meaningful variable and function names.
- Write comments and docstrings to explain complex logic.
- Use "v7p3r_*.py" custom implemented modules and local configuration implementations where applicable.

## Creativity and Complexity
- Stick to provided instructions and project outlines.
- Avoid unnecessary complexity; start with the simplest solution that meets requirements.
- Begin with small incremental changes and test with the user to verify acceptance.
- Avoid over-engineered solutions that involve multiple code edits and file changes.
- Instead of a focus on the smartest over thought out or "best" solution for chess decision-making, focus on modular components that can be easily swapped out or improved over time.
- Modular components are defined as those that have low couplings and high cohesion, they are performant, and each iteration is minimally impactful.
- Prior to the introduction of any fundamentally new functionality that diverts from the users current code, always verify the changes in a project document that you will follow so the user can provide feedbakc priorit to implementation.

## Housekeeping
- Any and all "test_*.py" files should be stored in the "testing/" directory.
- Any tests that need to be run that will similate a game can be done so with existing funtionality.
    - New test config files can be created in the same directory as current default_conifg.json and custom_config.josn files.
    - speed_config.json is a good example of a test config that should make moves within 10-15 seconds.
- Any new functionality should be documented in the project documents directory "docs/".
- Verify with the user if any other files are created, as they should be stored in an appropriate directory or placed in a temp location to be cleaned up once implementation is completed.
