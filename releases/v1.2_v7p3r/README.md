# V7P3R Chess Engine v1.2 - Consolidated Version

A clean, consolidated version of V7P3R Chess Engine built from v1.0 and v1.1 foundations for C++ conversion reference.

## Overview

This is a simplified, single-file chess engine implementation that consolidates the core functionality from v1.0 and v1.1 without the complexity of later versions. It maintains the original coding style and architecture patterns for easier C++ translation.

## Features

- Static position evaluation (no ML/GA/NN)
- Simple YAML configuration
- Alpha-beta search with move ordering
- Piece-square tables
- Opening book support
- Basic time management
- Single-file architecture (like original v1.0/v1.1)

## Files

- `v7p3r.py` - Main engine file (consolidated from v1.0/v1.1)
- `v7p3r.yaml` - Simple configuration file
- `engine_utilities/` - Supporting utility modules (PST, scoring, etc.)

## Usage

```python
python v7p3r.py
```

## Configuration

Engine settings are in `v7p3r.yaml` following the original v1.0/v1.1 format.

## Notes

This version specifically excludes:
- Complex multi-file modular architecture
- JSON configuration systems
- Advanced logging/metrics
- ML/Neural network components
- Complex dependency management

The goal is to provide a clean, simple reference for C++ conversion that maintains the original v1.* coding patterns and functionality.
