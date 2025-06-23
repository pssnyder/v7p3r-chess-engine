# engine_utilities/engine_monitor.app.py
"""
v7p3r Chess Engine Analysis Dashboard

This module provides a Streamlit-based dashboard for analyzing chess engine 
performance metrics over time, supporting both local and cloud-based data sources
with caching for improved performance.

Features:
- Configurable data source (local or cloud)
- Performance metrics visualization
- Configuration parameter analysis
- Game outcome tracking
- Data freshness indicators
- Caching for improved performance
"""

import streamlit as st
import psutil
import os
import time
import glob
import pandas as pd
import numpy as np
import yaml
import re
import json
import sqlite3
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
import chess
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, cast
from pathlib import Path
from functools import lru_cache
from io import StringIO
from matplotlib.axes import Axes

GAMES_DIR = "games"

# --- Configuration and Setup ---

# Cache duration in seconds
CACHE_TTL = 300  # 5 minutes

# Add a function to load config
def load_config(config_path="config/engine_utilities.yaml"):
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    st.warning(f"Config file not found at {config_path}, using defaults")
    return {}

# Load config
config = load_config()

# Constants
METRICS_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../metrics/chess_analytics.db"))
CLOUD_ENABLED = config.get('cloud', {}).get('enabled', False)
BUCKET_NAME = config.get('cloud', {}).get('bucket_name', "viper-chess-engine-data")
FIRESTORE_COLLECTION = config.get('cloud', {}).get('firestore_collection', "games")

# Dynamically import OpeningBook from engine_utilities/opening_book.py
def load_opening_fens():
    """Load opening FENs from the OpeningBook class."""
    opening_book_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "opening_book.py")
    )
    spec = importlib.util.spec_from_file_location("opening_book", opening_book_path)
    if spec is None or spec.loader is None:
        st.warning(f"Cannot load module. Ensure the file exists at: {opening_book_path}")
        return set()
    opening_book_mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(opening_book_mod)
    except Exception as e:
        st.warning(f"Failed to execute module: {e}")
        return set()
    book = opening_book_mod.OpeningBook()
    return set(book.book.keys())

OPENING_FENS = load_opening_fens()

# --- Data Source Management ---

# Configure logger
logger = logging.getLogger(__name__)

class DataSource:
    """Abstract base class for data sources."""
    
    def __init__(self):
        self.last_refresh = None
    
    def fetch_game_data(self) -> pd.DataFrame:
        """Fetch game data and update last_refresh timestamp."""
        self.last_refresh = datetime.datetime.now()
        return pd.DataFrame()
    
    def get_data_freshness(self) -> str:
        """Return human-readable string indicating data freshness."""
        if not self.last_refresh:
            return "Data not yet loaded"
        
        delta = datetime.datetime.now() - self.last_refresh
        if delta.total_seconds() < 60:
            return f"Updated {int(delta.total_seconds())} seconds ago"
        elif delta.total_seconds() < 3600:
            return f"Updated {int(delta.total_seconds() / 60)} minutes ago"
        else:
            return f"Updated {int(delta.total_seconds() / 3600)} hours ago"

class LocalDataSource(DataSource):
    """Data source that fetches from local SQLite database."""
    
    def __init__(self, db_path=METRICS_DB_PATH):
        super().__init__()
        self.db_path = db_path
    
    def get_connection(self):
        """Get a connection to the local SQLite database."""
        if not os.path.exists(self.db_path):
            st.error(f"Database not found at {self.db_path}")
            return None
        return sqlite3.connect(self.db_path)
    
    @lru_cache(maxsize=32)
    def fetch_game_data(self) -> pd.DataFrame:
        """Fetch game data from local SQLite database."""
        conn = self.get_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            # Query for game analytics data
            query = """
            SELECT 
                ga.game_id, ga.timestamp, ga.result as winner, ga.game_length,
                ga.avg_move_time_white, ga.avg_move_time_black, ga.opening,
                ga.white_engine, ga.black_engine, ga.etl_job_id,
                ga.white_material_advantage, ga.black_material_advantage,
                ga.move_count, ga.depth_reached, ga.nodes_searched
            FROM game_analytics ga
            ORDER BY ga.timestamp DESC
            LIMIT 1000
            """
            
            df = pd.read_sql_query(query, conn)
            
            # Query for engine performance metrics
            query_perf = """
            SELECT 
                ep.game_id, ep.engine_name, ep.avg_search_depth, 
                ep.avg_nodes_searched, ep.avg_time_per_move_ms,
                ep.effective_branching_factor, ep.search_efficiency,
                ep.nodes_per_second
            FROM engine_performance ep
            JOIN game_analytics ga ON ep.game_id = ga.game_id
            ORDER BY ga.timestamp DESC
            LIMIT 2000
            """
            
            df_perf = pd.read_sql_query(query_perf, conn)
            
            # Join the dataframes
            if not df.empty and not df_perf.empty:
                df = df.merge(df_perf, on='game_id', how='left')
            
            # Convert timestamps
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Update last refresh time
            self.last_refresh = datetime.datetime.now()
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching data from local database: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

class CloudDataSource(DataSource):
    """Data source that fetches from Google Cloud (Firestore and GCS)."""
    
    def __init__(self, bucket_name=BUCKET_NAME, collection=FIRESTORE_COLLECTION):
        super().__init__()
        self.bucket_name = bucket_name
        self.collection = collection
        self.cloud_store = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize CloudStore client."""
        try:
            # Import locally to avoid early dependency on cloud modules
            from engine_utilities.cloud_store import CloudStore
            
            self.cloud_store = CloudStore(
                bucket_name=self.bucket_name, 
                firestore_collection=self.collection,
                max_retries=3,
                retry_delay=1
            )
            logger.info(f"Cloud data source initialized with collection {self.collection}")
        except Exception as e:            
            st.error(f"Error initializing cloud connection: {e}")
            self.cloud_store = None
    
    @lru_cache(maxsize=32)
    def fetch_game_data(self) -> pd.DataFrame:
        """Fetch game data from Firestore with resilient error handling."""
        if not self.cloud_store:
            st.warning("Cloud connection not available. Attempting to reconnect...")
            self._initialize_client()
            
            if not self.cloud_store:
                st.error("Could not establish cloud connection. Please check credentials and network.")
                return pd.DataFrame()
        
        try:
            # Import locally to avoid early dependency on cloud modules
            from google.cloud import firestore
            
            # Use batch approach for better performance
            game_data = []
            batch_size = 100  # Process in batches of 100 games
            
            # Query games collection
            try:
                games_ref = self.cloud_store.db.collection(self.collection)
                query = games_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(batch_size)
                
                # Fetch games with retries built into CloudStore
                games = list(query.stream())
                
                logger.info(f"Retrieved {len(games)} games from cloud")
                
                # Process each game
                for game in games:
                    if game is None:
                        continue
                        
                    data = game.to_dict()
                    if not data:
                        continue
                        
                    data['game_id'] = game.id
                    
                    # Fetch moves subcollection for this game (with error handling)
                    try:
                        moves_ref = games_ref.document(game.id).collection('moves')
                        moves = list(moves_ref.stream())
                        moves_data = [move.to_dict() for move in moves if move is not None]
                        
                        # Calculate aggregates from moves
                        if moves_data:
                            data['move_count'] = len(moves_data)
                            
                            # Safe aggregations with error handling
                            try:
                                depths = [m.get('depth', 0) for m in moves_data if 'depth' in m]
                                if depths:
                                    data['avg_search_depth'] = np.mean(depths)
                                    
                                nodes = [m.get('nodes', 0) for m in moves_data if 'nodes' in m]
                                if nodes:
                                    data['avg_nodes_searched'] = np.mean(nodes)
                                    
                                times = [m.get('time_ms', 0) for m in moves_data if 'time_ms' in m]
                                if times:
                                    data['avg_time_per_move_ms'] = np.mean(times)
                                
                                # Calculate derived metrics
                                if nodes and times:
                                    # Nodes per second calculation
                                    total_nodes = sum(nodes)
                                    total_time_s = sum(times) / 1000
                                    if total_time_s > 0:
                                        data['nodes_per_second'] = total_nodes / total_time_s
                            except Exception as calc_error:
                                logger.warning(f"Error calculating metrics for game {game.id}: {calc_error}")
                            
                    except Exception as e:
                        logger.warning(f"Error fetching moves for game {game.id}: {e}")
                        # Continue with the game data we have
                    
                    game_data.append(data)
                
            except Exception as e:
                logger.error(f"Error querying Firestore: {e}")
                st.error(f"Error fetching data from cloud: {e}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(game_data)
            
            # Convert timestamps
            if 'timestamp' in df.columns and not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Rename columns to match local schema
            column_mapping = {
                'result': 'winner',
                'num_moves': 'game_length',
                'white_ai': 'white_engine',
                'black_ai': 'black_engine'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df[new_col] = df[old_col]
            
            # Update last refresh time
            self.last_refresh = datetime.datetime.now()
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching data from cloud: {e}")
            logger.error(f"Cloud data fetch error: {str(e)}")
            return pd.DataFrame()

# --- Data processing functions ---

# Extract FENs from PGN moves
def extract_fens_from_moves(moves_str, max_depth=12):
    """Given a moves string, return a list of FENs after each ply up to max_depth."""
    board = chess.Board()
    fens = []
    moves = re.findall(r"\d+\.\s*([^\s]+)(?:\s+([^\s]+))?", moves_str)
    ply = 0
    for move_pair in moves:
        for move in move_pair:
            if move and ply < max_depth:
                try:
                    board.push_san(move)
                    fens.append(board.fen())
                    ply += 1
                except Exception:
                    continue
    return fens

# Annotate dataframe with detected opening FEN
def annotate_openings(df):
    """Add detected opening information to the dataframe."""
    detected_openings = []
    
    for idx, row in df.iterrows():
        moves_str = row.get("moves", "")
        fens = extract_fens_from_moves(moves_str)
        found = None
        for fen in fens:
            # Only match up to the first space (piece placement, not castling/en passant/halfmove/fullmove)
            fen_key = " ".join(fen.split(" ")[:4])
            for opening_fen in OPENING_FENS:
                opening_fen_key = " ".join(opening_fen.split(" ")[:4])
                if fen_key == opening_fen_key:
                    found = opening_fen
                    break
            if found:
                break
        detected_openings.append(found)
    
    df["detected_opening_fen"] = detected_openings
    return df

# Helper function to load YAML files
def load_yaml(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

# --- Main Streamlit App ---

def main():
    st.set_page_config(
        page_title="v7p3r Chess Engine Analytics Dashboard",
        page_icon="♟️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("v7p3r Chess Engine Analytics Dashboard")
    
    # Initialize variables to avoid undefined references
    refresh_data = False
    auto_refresh = False
    refresh_interval = 5
    date_filter = [datetime.datetime.now() - datetime.timedelta(days=30), datetime.datetime.now()]
    selected_engines = None
      # Initialize variables to default values to avoid undefined variable errors
    refresh_data = False
    auto_refresh = False
    refresh_interval = 5
    date_filter = (datetime.datetime.now() - datetime.timedelta(days=30), datetime.datetime.now())
    selected_engines = None
    data_source_type = "Local Database"
    
    # Sidebar for data source configuration
    with st.sidebar:
        st.header("Dashboard Configuration")
        
        # Data source selection
        data_source_type = st.radio(
            "Select Data Source",
            ["Local Database", "Cloud Storage"],
            disabled=not CLOUD_ENABLED,
            help="Select where to fetch analytics data from"
        )
        
        # Cache control
        st.subheader("Cache Settings")
        st.write(f"Cache TTL: {CACHE_TTL} seconds")
        refresh_data = st.button("Refresh Data Now")
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh Interval (minutes)",
                min_value=1,
                max_value=60,
                value=5
            )
        
        # Filter settings
        st.subheader("Data Filters")
        date_filter = st.date_input(
            "Filter by Date Range",
            value=(datetime.datetime.now() - datetime.timedelta(days=30), datetime.datetime.now()),
            help="Filter data by date range"
        )
        if len(date_filter) == 2:
            start_date, end_date = date_filter
            st.write(f"Showing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Engine filter
        if 'df' in st.session_state and not st.session_state.df.empty:
            df = st.session_state.df
            engines = []
            if 'white_engine' in df.columns:
                engines.extend(df['white_engine'].unique())
            if 'black_engine' in df.columns:
                engines.extend(df['black_engine'].unique())
            engines = sorted(list(set(engines)))
            
            if engines:
                selected_engines = st.multiselect(
                    "Filter by Engine",
                    options=engines,
                    default=engines,
                    help="Select engines to include in analysis"
                )

# Create a placeholder for data freshness indicator with a more prominent display
    freshness_container = st.container()
    with freshness_container:
        freshness_col1, freshness_col2 = st.columns([3, 1])
        freshness_indicator = freshness_col1.empty()
        refresh_button = freshness_col2.empty()
    
    # Fetch and cache data
    @st.cache_data(ttl=CACHE_TTL)
    def get_data(source_type):
        if source_type == "Local Database" or not CLOUD_ENABLED:
            return LocalDataSource().fetch_game_data()
        else:
            return CloudDataSource().fetch_game_data()
    
    # Get data based on cache or refresh
    if refresh_data or 'df' not in st.session_state:
        with st.spinner("Fetching chess analytics data..."):
            if data_source_type == "Local Database" or not CLOUD_ENABLED:
                data_source = LocalDataSource()
            else:
                data_source = CloudDataSource()
                
            df = data_source.fetch_game_data()
            st.session_state.df = df
            st.session_state.data_source = data_source
    else:
        df = st.session_state.df
        data_source = st.session_state.data_source

    # Apply date filter if available
    if len(date_filter) == 2 and 'timestamp' in df.columns and not df.empty:
        start_date, end_date = date_filter
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # End of day
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

    # Apply engine filter if available
    if selected_engines:
        df = df[
            (df['white_engine'].isin(selected_engines)) | 
            (df['black_engine'].isin(selected_engines))
        ]

    # Show data freshness with a more prominent display
    freshness_status = data_source.get_data_freshness()
    freshness_seconds = 0
    if data_source.last_refresh:
        freshness_seconds = (datetime.datetime.now() - data_source.last_refresh).total_seconds()

    # Color-coded freshness indicator
    if freshness_seconds < 300:  # Less than 5 minutes
        freshness_indicator.success(f"🟢 Data Freshness: {freshness_status}")
    elif freshness_seconds < 3600:  # Less than 1 hour
        freshness_indicator.info(f"🟡 Data Freshness: {freshness_status}")
    else:
        freshness_indicator.warning(f"🟠 Data Freshness: {freshness_status}")

    # Refresh button in the freshness indicator area
    if refresh_button.button("🔄 Refresh Now"):
        with st.spinner("Refreshing data..."):
            if data_source_type == "Local Database" or not CLOUD_ENABLED:
                data_source = LocalDataSource()
            else:
                data_source = CloudDataSource()
                
            df = data_source.fetch_game_data()
            st.session_state.df = df
            st.session_state.data_source = data_source
            st.rerun()

    # Auto-refresh mechanism
    if auto_refresh:
        time_placeholder = st.empty()
        if 'last_auto_refresh' not in st.session_state:
            st.session_state.last_auto_refresh = datetime.datetime.now()
        
        # Check if it's time to refresh
        time_since_refresh = (datetime.datetime.now() - st.session_state.last_auto_refresh).total_seconds()
        if time_since_refresh >= refresh_interval * 60:
            with st.spinner("Auto-refreshing data..."):
                df = data_source.fetch_game_data()
                st.session_state.df = df
                st.session_state.last_auto_refresh = datetime.datetime.now()
                st.rerun()
    
    # Stop if no data
    if df.empty:
        st.warning("No chess games found in the selected data source.")
        st.stop()
    
    # Summary metrics section
    st.header("Summary Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Games", len(df))
    
    with col2:
        if 'winner' in df.columns:
            white_win_rate = 100 * (df['winner'] == '1-0').mean()
            st.metric("Win Rate as White", f"{white_win_rate:.1f}%")
    
    with col3:
        if 'winner' in df.columns:
            black_win_rate = 100 * (df['winner'] == '0-1').mean()
            st.metric("Win Rate as Black", f"{black_win_rate:.1f}%")
    
    with col4:
        if 'winner' in df.columns:
            draw_rate = 100 * (df['winner'] == '1/2-1/2').mean()
            st.metric("Draw Rate", f"{draw_rate:.1f}%")
    
    with col5:
        if 'game_length' in df.columns:
            avg_length = df['game_length'].mean()
            st.metric("Avg Game Length", f"{avg_length:.1f} moves")
    
    # Engine Performance Trends Section
    st.header("Engine Performance Over Time")
    
    time_metrics_tab, node_metrics_tab, move_time_tab = st.tabs([
        "Win Rate & Search Depth", 
        "Node Analysis", 
        "Move Time Analysis"
    ])
    
    with time_metrics_tab:
        st.subheader("Win Rate and Search Depth Trends")
        
        if df['timestamp'].nunique() > 5:
            # Prepare time series data
            df_sorted = df.sort_values('timestamp')
            df_ts = df_sorted.set_index('timestamp')
            
            # Create visualization
            fig, ax1 = plt.subplots(figsize=(12, 6))
                  # Plot win rate
        if 'winner' in df.columns:
            rolling_window = min(10, len(df_ts) // 2) if len(df_ts) > 0 else 1
            df_ts['win_rate'] = df_ts['winner'].eq('1-0').rolling(rolling_window).mean() * 100
            ax1.plot(df_ts.index, df_ts['win_rate'], 'b-', label='Win Rate %', marker='o')
            ax1.set_ylabel('Win Rate %', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_ylim(0, 100)
          # Plot search depth on second y-axis
        if 'avg_search_depth' in df.columns:
            ax2 = ax1.twinx()
            # Use cast to correctly type the axes for Pylance
            ax2_typed = cast(Axes, ax2)
            ax2_typed.plot(df_ts.index, df_ts['avg_search_depth'].to_numpy(), 'r-', label='Avg Search Depth', marker='s')
            ax2.set_ylabel('Average Search Depth', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
                
            # Formatting
            ax1.set_xlabel('Date')
            ax1.grid(True, alpha=0.3)
            plt.title('Win Rate and Search Depth Over Time')
            fig.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
        else:
            st.info("Need more time-series data to create meaningful visualizations.")

    with node_metrics_tab:
        st.subheader("Node Search Analysis")
        
        if 'avg_nodes_searched' in df.columns and df['timestamp'].nunique() > 5:
            # Prepare data
            df_sorted = df.sort_values('timestamp')
            df_ts = df_sorted.set_index('timestamp')
            
            # Create visualization
            fig, ax1 = plt.subplots(figsize=(12, 6))
                  # Plot nodes searched
        ax1.plot(df_ts.index, df_ts['avg_nodes_searched'], 'g-', 
                label='Avg Nodes Searched', marker='^')
        ax1.set_ylabel('Avg Nodes Searched', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
          # Plot nodes per second on second y-axis if available
        if 'nodes_per_second' in df.columns:
            ax2 = ax1.twinx()
            # Use cast to correctly type the axes for Pylance
            ax2_typed = cast(Axes, ax2)
            ax2_typed.plot(df_ts.index, df_ts['nodes_per_second'].to_numpy(), 'm-', 
                    label='Nodes per Second', marker='x')
            ax2.set_ylabel('Nodes per Second', color='m')
            ax2.tick_params(axis='y', labelcolor='m')
            
            # Formatting
            ax1.set_xlabel('Date')
            ax1.grid(True, alpha=0.3)
            plt.title('Node Search Performance Over Time')
            fig.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
        else:
            st.info("Node metrics not available in the data.")

    with move_time_tab:
        st.subheader("Move Time Analysis")
        
        if 'avg_time_per_move_ms' in df.columns and df['timestamp'].nunique() > 5:
            # Prepare data
            df_sorted = df.sort_values('timestamp')
            df_ts = df_sorted.set_index('timestamp')
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Convert to seconds for better readability
            df_ts['avg_time_per_move_s'] = df_ts['avg_time_per_move_ms'] / 1000
          # Plot move time
            ax.plot(df_ts.index, df_ts['avg_time_per_move_s'], 'c-', 
                    label='Avg Time per Move (s)', marker='d')
            
            # Formatting
            ax.set_xlabel('Date')
            ax.set_ylabel('Average Time per Move (seconds)')
            ax.grid(True, alpha=0.3)
            plt.title('Move Time Analysis Over Time')
            fig.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
        else:
            st.info("Move time metrics not available in the data.")
        
    # Material Advantage Analysis
    if 'white_material_advantage' in df.columns and 'black_material_advantage' in df.columns:
        st.header("Material Advantage Analysis")
        
        material_df = df[['white_material_advantage', 'black_material_advantage', 'winner']].copy()
        material_df['net_advantage'] = material_df['white_material_advantage'] - material_df['black_material_advantage']
        
        fig, ax = plt.subplots(figsize=(10, 6))
          # Create bins for material advantage
        bin_edges = list(range(-10, 11, 2))
        material_df['advantage_bin'] = pd.cut(material_df['net_advantage'], bin_edges)
        
        # Group by advantage bin and calculate win rates
        grouped = material_df.groupby('advantage_bin')['winner'].value_counts(normalize=True).unstack().fillna(0)
        
        # Plot
        grouped.plot(kind='bar', ax=ax)
        ax.set_xlabel('Material Advantage (White - Black)')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate by Material Advantage')
        ax.legend(title='Result')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Engine Comparison
    if 'white_engine' in df.columns and 'black_engine' in df.columns and 'winner' in df.columns:
        st.header("Engine Comparison")
        
        # Prepare data
        engines = pd.concat([
            df['white_engine'].rename('engine'),
            df['black_engine'].rename('engine')
        ])
        unique_engines = engines.unique()
        
        if len(unique_engines) > 1:
            # Create matchup matrix
            matchups = []
            
            for engine1 in unique_engines:
                for engine2 in unique_engines:
                    if engine1 != engine2:
                        # Games where engine1 is white and engine2 is black
                        white_games = df[(df['white_engine'] == engine1) & (df['black_engine'] == engine2)]
                        white_wins = (white_games['winner'] == '1-0').sum()
                        white_draws = (white_games['winner'] == '1/2-1/2').sum()
                        white_losses = (white_games['winner'] == '0-1').sum()
                        white_total = len(white_games)
                        
                        # Games where engine1 is black and engine2 is white
                        black_games = df[(df['white_engine'] == engine2) & (df['black_engine'] == engine1)]
                        black_wins = (black_games['winner'] == '0-1').sum()
                        black_draws = (black_games['winner'] == '1/2-1/2').sum()
                        black_losses = (black_games['winner'] == '1-0').sum()
                        black_total = len(black_games)
                        
                        # Total stats
                        total_games = white_total + black_total
                        total_wins = white_wins + black_wins
                        total_draws = white_draws + black_draws
                        total_losses = white_losses + black_losses
                        
                        if total_games > 0:
                            win_rate = (total_wins / total_games) * 100
                            draw_rate = (total_draws / total_games) * 100
                            loss_rate = (total_losses / total_games) * 100
                            
                            matchups.append({
                                'Engine': engine1,
                                'Opponent': engine2,
                                'Games': total_games,
                                'Wins': total_wins,
                                'Draws': total_draws,
                                'Losses': total_losses,
                                'Win Rate': f"{win_rate:.1f}%",
                                'Draw Rate': f"{draw_rate:.1f}%",
                                'Loss Rate': f"{loss_rate:.1f}%",
                                'Score': f"{total_wins + (total_draws * 0.5)}/{total_games}"
                            })
            
            if matchups:
                matchup_df = pd.DataFrame(matchups)
                st.dataframe(matchup_df, use_container_width=True)
                
                # Create a win rate heatmap
                engines_list = sorted(unique_engines)
                matrix = np.zeros((len(engines_list), len(engines_list)))
                
                for i, engine1 in enumerate(engines_list):
                    for j, engine2 in enumerate(engines_list):
                        if engine1 != engine2:
                            # Find the matchup
                            matchup = matchup_df[(matchup_df['Engine'] == engine1) & (matchup_df['Opponent'] == engine2)]
                            if not matchup.empty:
                                win_rate = float(matchup['Win Rate'].iloc[0].strip('%'))
                                matrix[i, j] = win_rate
                            else:
                                matrix[i, j] = 0
                
                # Plot heatmap - try with seaborn if available, otherwise use matplotlib
                try:
                    import seaborn as sns
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="RdYlGn", 
                               xticklabels=engines_list, yticklabels=engines_list, ax=ax)
                    ax.set_title("Win Rate (%) - Engine (row) vs Opponent (column)")
                    plt.tight_layout()
                    st.pyplot(fig)
                except ImportError:
                    # Fallback to matplotlib if seaborn not available
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(matrix, cmap="RdYlGn")
                    ax.set_xticks(np.arange(len(engines_list)))
                    ax.set_yticks(np.arange(len(engines_list)))
                    ax.set_xticklabels(engines_list)
                    ax.set_yticklabels(engines_list)
                    ax.set_title("Win Rate (%) - Engine (row) vs Opponent (column)")
                    
                    # Add text annotations
                    for i in range(len(engines_list)):
                        for j in range(len(engines_list)):
                            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="black")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("No engine matchups found in the data.")
        else:
            st.info("Only one engine found in the data, no comparison possible.")
    
    # Performance by Opening Section
    if 'detected_opening_fen' in df.columns and df['detected_opening_fen'].notna().any():
        st.header("Performance by Opening")
        
        # Filter to games with detected openings
        opening_df = df[df['detected_opening_fen'].notna()].copy()
        
        if not opening_df.empty:
            # Get counts of each opening
            opening_counts = opening_df['detected_opening_fen'].value_counts()
            
            # Only analyze openings with at least 5 games
            common_openings = opening_counts[opening_counts >= 5].index.tolist()
            
            if common_openings:
                common_openings_df = opening_df[opening_df['detected_opening_fen'].isin(common_openings)]
                
                # Calculate win rates by opening
                opening_results = []
                
                for opening in common_openings:
                    opening_games = common_openings_df[common_openings_df['detected_opening_fen'] == opening]
                    total_games = len(opening_games)
                    white_wins = (opening_games['winner'] == '1-0').sum()
                    black_wins = (opening_games['winner'] == '0-1').sum()
                    draws = (opening_games['winner'] == '1/2-1/2').sum()
                    
                    # Calculate percentages
                    white_win_pct = (white_wins / total_games) * 100 if total_games > 0 else 0
                    black_win_pct = (black_wins / total_games) * 100 if total_games > 0 else 0
                    draw_pct = (draws / total_games) * 100 if total_games > 0 else 0
                    
                    # Get the first few moves as a readable name
                    opening_name = f"Opening {common_openings.index(opening) + 1}"
                    
                    opening_results.append({
                        'Opening': opening_name,
                        'FEN': opening,
                        'Games': total_games,
                        'White Win %': white_win_pct,
                        'Black Win %': black_win_pct,
                        'Draw %': draw_pct
                    })
                
                if opening_results:
                    # Convert to DataFrame and display
                    results_df = pd.DataFrame(opening_results)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Create stacked bar chart
                    results_df.plot(
                        x='Opening',
                        y=['White Win %', 'Draw %', 'Black Win %'],
                        kind='bar',
                        stacked=True,
                        ax=ax,
                        color=['green', 'gray', 'blue']
                    )
                    
                    ax.set_title('Win Rates by Opening')
                    ax.set_xlabel('Opening')
                    ax.set_ylabel('Percentage')
                    ax.legend(title='Result')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Display plot and data
                    st.pyplot(fig)
                    
                    with st.expander("View Opening Details"):
                        st.dataframe(results_df)
            else:
                st.info("Not enough games with the same opening for meaningful analysis.")
        else:
            st.info("No games with detected openings found in the data.")
        
    # Engine Performance Trends
    st.header("Engine Performance Trends")

    if 'white_engine' in df.columns and 'black_engine' in df.columns and df['timestamp'].nunique() > 5:
        # Get unique engines
        engines = []
        if 'white_engine' in df.columns:
            engines.extend(df['white_engine'].unique())
        if 'black_engine' in df.columns:
            engines.extend(df['black_engine'].unique())
        engines = sorted(list(set([e for e in engines if e])))
        
        if len(engines) > 1:
            # Create a DataFrame with win rates over time for each engine
            engine_perf_data = []
            
            # Group by month for better visualization
            df['month'] = df['timestamp'].dt.to_period('M')
            
            for engine in engines:
                # Games where this engine played as white
                white_games_by_month = df[df['white_engine'] == engine].groupby('month')
                white_wins_by_month = white_games_by_month['winner'].apply(lambda x: (x == '1-0').mean() * 100)
                
                # Games where this engine played as black
                black_games_by_month = df[df['black_engine'] == engine].groupby('month')
                black_wins_by_month = black_games_by_month['winner'].apply(lambda x: (x == '0-1').mean() * 100)
                
                # Combine data
                for month in sorted(df['month'].unique()):
                    month_str = month.strftime('%Y-%m')
                    white_wr = white_wins_by_month.get(month, 0)
                    black_wr = black_wins_by_month.get(month, 0)
                    
                    # Only add if there's data
                    if not pd.isna(white_wr) or not pd.isna(black_wr):
                        engine_perf_data.append({
                            'Engine': engine,
                            'Month': month_str,
                            'White Win Rate': 0 if pd.isna(white_wr) else white_wr,
                            'Black Win Rate': 0 if pd.isna(black_wr) else black_wr,
                            'Combined Win Rate': (0 if pd.isna(white_wr) else white_wr + 0 if pd.isna(black_wr) else black_wr) / 2
                        })
            
            if engine_perf_data:
                perf_df = pd.DataFrame(engine_perf_data)
                
                # Create line chart
                fig, ax = plt.subplots(figsize=(12, 8))
                
                for engine in engines:
                    engine_data = perf_df[perf_df['Engine'] == engine]
                    if not engine_data.empty:
                        ax.plot(
                            engine_data['Month'], 
                            engine_data['Combined Win Rate'],
                            marker='o',
                            linewidth=2,
                            label=engine
                        )
                
                ax.set_title('Engine Win Rate Trends')
                ax.set_xlabel('Month')
                ax.set_ylabel('Win Rate %')
                ax.legend(title='Engine')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Show detailed view in tabs
                white_tab, black_tab = st.tabs(["Performance as White", "Performance as Black"])
                
                with white_tab:
                    # White performance
                    st.subheader("Engine Performance as White")
                    fig_white, ax_white = plt.subplots(figsize=(12, 6))
                    
                    for engine in engines:
                        engine_data = perf_df[perf_df['Engine'] == engine]
                        if not engine_data.empty:
                            ax_white.plot(
                                engine_data['Month'], 
                                engine_data['White Win Rate'],
                                marker='o',
                                linewidth=2,
                                label=engine
                            )
                    
                    ax_white.set_title('Win Rate as White')
                    ax_white.set_xlabel('Month')
                    ax_white.set_ylabel('Win Rate %')
                    ax_white.legend(title='Engine')
                    ax_white.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig_white)
                
                with black_tab:
                    # Black performance
                    st.subheader("Engine Performance as Black")
                    fig_black, ax_black = plt.subplots(figsize=(12, 6))
                    
                    for engine in engines:
                        engine_data = perf_df[perf_df['Engine'] == engine]
                        if not engine_data.empty:
                            ax_black.plot(
                                engine_data['Month'], 
                                engine_data['Black Win Rate'],
                                marker='o',
                                linewidth=2,
                                label=engine
                            )
                    
                    ax_black.set_title('Win Rate as Black')
                    ax_black.set_xlabel('Month')
                    ax_black.set_ylabel('Win Rate %')
                    ax_black.legend(title='Engine')
                    ax_black.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig_black)
        else:
            st.info("Need data from multiple engines for comparison.")
    else:
        st.info("Need more time-series data with multiple engines for meaningful trend analysis.")
    
    # Raw data and download
    with st.expander("View Raw Data"):
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"chess_analytics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Run the app
if __name__ == "__main__":
    main()
