#!/usr/bin/env python3
"""
Fetch games from Lichess API and run analytics
Downloads games for a specific date range and analyzes them
"""
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

def fetch_games_from_lichess(username, since_date, until_date=None, output_file=None):
    """
    Fetch games from Lichess API.
    
    Args:
        username: Lichess username
        since_date: Start date (datetime object or timestamp)
        until_date: End date (datetime object or timestamp), defaults to now
        output_file: Path to save PGN file
    
    Returns:
        Path to saved PGN file
    """
    # Convert dates to timestamps (milliseconds)
    if isinstance(since_date, datetime):
        since_ms = int(since_date.timestamp() * 1000)
    else:
        since_ms = since_date
    
    if until_date is None:
        until_ms = int(datetime.now().timestamp() * 1000)
    elif isinstance(until_date, datetime):
        until_ms = int(until_date.timestamp() * 1000)
    else:
        until_ms = until_date
    
    # Build API URL
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "tags": "true",
        "clocks": "true",
        "evals": "false",  # We'll do our own analysis with Stockfish
        "opening": "true",
        "literate": "true",
        "since": since_ms,
        "until": until_ms,
        "perfType": "rapid,blitz,classical",  # Include common time controls
    }
    
    print(f"Fetching games for {username}...")
    print(f"  From: {datetime.fromtimestamp(since_ms/1000).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  To:   {datetime.fromtimestamp(until_ms/1000).strftime('%Y-%m-%d %H:%M:%S')}")
    
    headers = {
        "Accept": "application/x-chess-pgn"
    }
    
    # Make request
    response = requests.get(url, params=params, headers=headers, stream=True)
    
    if response.status_code != 200:
        print(f"Error: API returned status {response.status_code}")
        print(response.text)
        return None
    
    # Save to file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"lichess_{username}_{timestamp}.pgn"
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Stream download to file
    game_count = 0
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                # Count games (rough estimate by counting [Event headers)
                game_count += chunk.decode('utf-8', errors='ignore').count('[Event ')
    
    file_size = output_path.stat().st_size
    print(f"\n✓ Downloaded {game_count} games (~{file_size / 1024 / 1024:.1f} MB)")
    print(f"✓ Saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch v7p3r_bot games from Lichess")
    parser.add_argument("--username", default="v7p3r_bot", help="Lichess username")
    parser.add_argument("--since", required=True, help="Start date (YYYY-MM-DD or timestamp)")
    parser.add_argument("--until", help="End date (YYYY-MM-DD or timestamp), defaults to now")
    parser.add_argument("--output", help="Output PGN file path")
    
    args = parser.parse_args()
    
    # Parse dates
    try:
        # Try as timestamp first
        since_ms = int(args.since)
    except ValueError:
        # Try as date string
        try:
            since_date = datetime.strptime(args.since, "%Y-%m-%d")
            since_ms = int(since_date.timestamp() * 1000)
        except ValueError:
            print(f"Error: Invalid date format '{args.since}'. Use YYYY-MM-DD or timestamp.")
            sys.exit(1)
    
    until_ms = None
    if args.until:
        try:
            until_ms = int(args.until)
        except ValueError:
            try:
                until_date = datetime.strptime(args.until, "%Y-%m-%d")
                until_ms = int(until_date.timestamp() * 1000)
            except ValueError:
                print(f"Error: Invalid date format '{args.until}'. Use YYYY-MM-DD or timestamp.")
                sys.exit(1)
    
    # Fetch games
    output_file = fetch_games_from_lichess(
        args.username,
        since_ms,
        until_ms,
        args.output
    )
    
    if output_file:
        print(f"\nReady for analysis: python full_analysis.py {output_file}")
