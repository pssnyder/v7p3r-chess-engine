# Log Management and Cleanup System

## Overview

The V7P3R Chess Engine now includes a log management system that automatically clears log files before starting a new game. This helps prevent confusion with old logs and errors, and keeps the logging directory clean.

## Features

1. **Automatic Log Clearing**: When a new game starts, all log files are automatically cleared
2. **Selective Clearing**: Can clear logs for specific modules only
3. **Handles Rotated Logs**: Cleans up both main log files (.log) and rotated log files (.log.1, .log.2, etc.)
4. **Safe Operation**: Preserves log directory structure, only clears file contents

## Usage

The log clearing system is integrated into the game initialization process and works automatically, but you can also use it manually:

```python
# Import the logger
from v7p3r_debug import v7p3rLogger

# Clear all logs
v7p3rLogger.clear_logs()

# Clear logs for a specific module only
v7p3rLogger.clear_logs("v7p3r_search")
```

## Implementation Details

The log clearing functionality is implemented in the `v7p3rLogger.clear_logs()` method in `v7p3r_debug.py`. It works by:

1. Finding the logging directory
2. Identifying all log files (both main and rotated)
3. Truncating each file to zero bytes (preserving the file but removing all content)
4. Returning the count of cleared files

## Benefits

- **Clearer Debugging**: Each game session starts with fresh, empty logs
- **Reduced Confusion**: No need to wonder if an error message is from a previous run
- **Disk Space Management**: Prevents log files from growing indefinitely
- **Better Focus**: When analyzing logs, you only see relevant information from the current session

The log clearing system is designed to be unobtrusive and automatic, requiring no user intervention during normal operation.
