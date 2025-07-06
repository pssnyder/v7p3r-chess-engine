# metrics/chess_metrics.py
"""
V7P3R Chess Engine Metrics System
A unified, modern metrics collection and analysis system for the V7P3R chess engine.
Features async data collection, Streamlit dashboard, and robust database handling.
"""

import asyncio
import sqlite3
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "chess_metrics.db"

@dataclass
class GameMetric:
    """Data structure for game-level metrics."""
    game_id: str
    timestamp: str
    v7p3r_color: str  # 'white' or 'black'
    opponent: str  # 'stockfish' or other engine name
    result: str  # 'win', 'loss', 'draw'
    total_moves: int
    game_duration: float  # seconds
    opening_name: Optional[str] = None
    final_position_fen: Optional[str] = None
    termination_reason: Optional[str] = None
    
@dataclass
class MoveMetric:
    """Data structure for move-level metrics."""
    game_id: str
    move_number: int
    player: str  # 'v7p3r' or 'opponent'
    move_notation: str
    position_fen: str
    evaluation_score: Optional[float] = None
    search_depth: Optional[int] = None
    nodes_evaluated: Optional[int] = None
    time_taken: Optional[float] = None
    best_move: Optional[str] = None
    pv_line: Optional[str] = None
    quiescence_nodes: Optional[int] = None
    transposition_hits: Optional[int] = None
    move_ordering_efficiency: Optional[float] = None
    
@dataclass  
class EngineConfig:
    """Data structure for engine configuration snapshots."""
    config_id: str
    timestamp: str
    search_depth: int
    time_limit: float
    use_iterative_deepening: bool
    use_transposition_table: bool
    use_quiescence_search: bool
    use_move_ordering: bool
    hash_size_mb: int
    additional_params: Optional[Dict[str, Any]] = None


class v7p3rMetrics:
    """
    Unified metrics collection and analysis system for V7P3R chess engine.
    Supports async collection, robust database handling, and comprehensive analysis.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._lock = threading.Lock()
        self._initialize_database()
        logger.info(f"v7p3rMetrics initialized with database: {self.db_path}")
    
    def _initialize_database(self):
        """Initialize database tables if they don't exist."""
        # Check if database exists and has old schema
        needs_migration = False
        if self.db_path.exists():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    # Check if we have the old schema by looking for old table names
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    existing_tables = [row[0] for row in cursor.fetchall()]
                    
                    # If we have old tables but not new ones, we need to migrate
                    if ('game_results' in existing_tables or 'move_metrics' in existing_tables) and 'games' not in existing_tables:
                        needs_migration = True
                        logger.info("Detected old database schema, backing up and creating new schema...")
                        
                        # Backup the old database
                        backup_path = self.db_path.with_suffix('.db.backup')
                        import shutil
                        shutil.copy2(self.db_path, backup_path)
                        logger.info(f"Old database backed up to: {backup_path}")
                        
                        # Remove old database to start fresh
                        self.db_path.unlink()
            except Exception as e:
                logger.warning(f"Error checking database schema: {e}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Games table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    v7p3r_color TEXT NOT NULL,
                    opponent TEXT NOT NULL,
                    result TEXT NOT NULL,
                    total_moves INTEGER NOT NULL,
                    game_duration REAL NOT NULL,
                    opening_name TEXT,
                    final_position_fen TEXT,
                    termination_reason TEXT
                )
            """)
            
            # Moves table  
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS moves (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    move_number INTEGER NOT NULL,
                    player TEXT NOT NULL,
                    move_notation TEXT NOT NULL,
                    position_fen TEXT NOT NULL,
                    evaluation_score REAL,
                    search_depth INTEGER,
                    nodes_evaluated INTEGER,
                    time_taken REAL,
                    best_move TEXT,
                    pv_line TEXT,
                    quiescence_nodes INTEGER,
                    transposition_hits INTEGER,
                    move_ordering_efficiency REAL,
                    FOREIGN KEY (game_id) REFERENCES games (game_id)
                )
            """)
            
            # Engine configs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS engine_configs (
                    config_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    search_depth INTEGER NOT NULL,
                    time_limit REAL NOT NULL,
                    use_iterative_deepening BOOLEAN NOT NULL,
                    use_transposition_table BOOLEAN NOT NULL,
                    use_quiescence_search BOOLEAN NOT NULL,
                    use_move_ordering BOOLEAN NOT NULL,
                    hash_size_mb INTEGER NOT NULL,
                    additional_params TEXT
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_timestamp ON games(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_moves_game_id ON moves(game_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_moves_player ON moves(player)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_configs_timestamp ON engine_configs(timestamp)")
            
            conn.commit()
            logger.info("Database tables initialized successfully")
            if needs_migration:
                logger.info("New database schema created. Old data backed up but not migrated automatically.")
    
    async def record_game_start(self, game_metric: GameMetric) -> bool:
        """Record the start of a new game."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO games 
                        (game_id, timestamp, v7p3r_color, opponent, result, total_moves, 
                         game_duration, opening_name, final_position_fen, termination_reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        game_metric.game_id, game_metric.timestamp, game_metric.v7p3r_color,
                        game_metric.opponent, game_metric.result, game_metric.total_moves,
                        game_metric.game_duration, game_metric.opening_name,
                        game_metric.final_position_fen, game_metric.termination_reason
                    ))
                    conn.commit()
            logger.info(f"Game started: {game_metric.game_id}")
            return True
        except Exception as e:
            logger.error(f"Error recording game start: {e}")
            return False
    
    async def record_move(self, move_metric: MoveMetric) -> bool:
        """Record a single move with all associated metrics."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO moves 
                        (game_id, move_number, player, move_notation, position_fen,
                         evaluation_score, search_depth, nodes_evaluated, time_taken,
                         best_move, pv_line, quiescence_nodes, transposition_hits,
                         move_ordering_efficiency)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        move_metric.game_id, move_metric.move_number, move_metric.player,
                        move_metric.move_notation, move_metric.position_fen,
                        move_metric.evaluation_score, move_metric.search_depth,
                        move_metric.nodes_evaluated, move_metric.time_taken,
                        move_metric.best_move, move_metric.pv_line,
                        move_metric.quiescence_nodes, move_metric.transposition_hits,
                        move_metric.move_ordering_efficiency
                    ))
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error recording move: {e}")
            return False
    
    async def update_game_result(self, game_id: str, result: str, 
                               total_moves: int, game_duration: float,
                               final_position_fen: Optional[str] = None,
                               termination_reason: Optional[str] = None) -> bool:
        """Update game result when game is completed."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE games 
                        SET result = ?, total_moves = ?, game_duration = ?,
                            final_position_fen = ?, termination_reason = ?
                        WHERE game_id = ?
                    """, (result, total_moves, game_duration, final_position_fen, 
                          termination_reason, game_id))
                    conn.commit()
            logger.info(f"Game result updated: {game_id} -> {result}")
            return True
        except Exception as e:
            logger.error(f"Error updating game result: {e}")
            return False
    
    async def save_engine_config(self, config: EngineConfig) -> bool:
        """Save engine configuration snapshot."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    additional_params_json = json.dumps(config.additional_params) if config.additional_params else None
                    cursor.execute("""
                        INSERT OR REPLACE INTO engine_configs
                        (config_id, timestamp, search_depth, time_limit,
                         use_iterative_deepening, use_transposition_table,
                         use_quiescence_search, use_move_ordering, hash_size_mb,
                         additional_params)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        config.config_id, config.timestamp, config.search_depth,
                        config.time_limit, config.use_iterative_deepening,
                        config.use_transposition_table, config.use_quiescence_search,
                        config.use_move_ordering, config.hash_size_mb, additional_params_json
                    ))
                    conn.commit()
            logger.info(f"Engine config saved: {config.config_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving engine config: {e}")
            return False

    
    # Analysis methods
    async def get_game_summary(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive summary for a specific game."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get game info
                cursor.execute("SELECT * FROM games WHERE game_id = ?", (game_id,))
                game_row = cursor.fetchone()
                if not game_row:
                    return None
                
                # Get column names
                cursor.execute("PRAGMA table_info(games)")
                game_columns = [col[1] for col in cursor.fetchall()]
                game_data = dict(zip(game_columns, game_row))
                
                # Get move statistics
                cursor.execute("""
                    SELECT COUNT(*) as total_moves,
                           AVG(evaluation_score) as avg_evaluation,
                           AVG(search_depth) as avg_depth,
                           AVG(nodes_evaluated) as avg_nodes,
                           AVG(time_taken) as avg_time
                    FROM moves WHERE game_id = ? AND player = 'v7p3r'
                """, (game_id,))
                
                stats = cursor.fetchone()
                if stats:
                    game_data.update({
                        'total_v7p3r_moves': stats[0],
                        'avg_evaluation': stats[1],
                        'avg_search_depth': stats[2],
                        'avg_nodes_evaluated': stats[3],
                        'avg_time_per_move': stats[4]
                    })
                
                return game_data
        except Exception as e:
            logger.error(f"Error getting game summary: {e}")
            return None
    
    async def get_performance_trends(self, limit: int = 50) -> pd.DataFrame:
        """Get recent performance trends for analysis."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT g.game_id, g.timestamp, g.result, g.v7p3r_color,
                           AVG(m.evaluation_score) as avg_evaluation,
                           AVG(m.search_depth) as avg_depth,
                           AVG(m.nodes_evaluated) as avg_nodes,
                           AVG(m.time_taken) as avg_time
                    FROM games g
                    LEFT JOIN moves m ON g.game_id = m.game_id AND m.player = 'v7p3r'
                    WHERE g.timestamp IS NOT NULL
                    GROUP BY g.game_id, g.timestamp, g.result, g.v7p3r_color
                    ORDER BY g.timestamp DESC
                    LIMIT ?
                """
                return pd.read_sql_query(query, conn, params=(limit,))
        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return pd.DataFrame()
    
    async def get_engine_config_history(self) -> pd.DataFrame:
        """Get engine configuration history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM engine_configs 
                    ORDER BY timestamp DESC
                """
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error getting config history: {e}")
            return pd.DataFrame()


# Global metrics instance
_global_metrics = None

def get_metrics_instance(db_path: Optional[Path] = None) -> v7p3rMetrics:
    """Get global metrics instance (singleton pattern)."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = v7p3rMetrics(db_path)
    return _global_metrics


# Streamlit Dashboard
def create_streamlit_dashboard():
    """
    Create a Streamlit dashboard for visualizing chess engine metrics.
    Run this function to launch the dashboard.
    """
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        
        st.set_page_config(
            page_title="V7P3R Chess Engine Analytics",
            page_icon="♔",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for dark theme
        st.markdown("""
        <style>
        .main { background-color: #0E1117; }
        .stApp { background-color: #0E1117; }
        .css-1d391kg { background-color: #262730; }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("♔ V7P3R Chess Engine Analytics Dashboard")
        st.markdown("---")
        
        # Initialize metrics
        metrics = get_metrics_instance()
        
        # Sidebar
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Main dashboard sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Game Analysis", "⚙️ Engine Config", "📈 Performance Trends"])
        
        with tab1:
            st.header("📊 Engine Performance Overview")
            
            # Get recent games
            games_df = asyncio.run(metrics.get_performance_trends(limit=20))
            
            if not games_df.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_games = len(games_df)
                    st.metric("Total Games", total_games)
                
                with col2:
                    wins = len(games_df[games_df['result'] == 'win'])
                    win_rate = (wins / total_games * 100) if total_games > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                with col3:
                    avg_depth = games_df['avg_depth'].mean()
                    st.metric("Avg Search Depth", f"{avg_depth:.1f}" if pd.notnull(avg_depth) else "N/A")
                
                with col4:
                    avg_nodes = games_df['avg_nodes'].mean()
                    st.metric("Avg Nodes/Move", f"{avg_nodes:,.0f}" if pd.notnull(avg_nodes) else "N/A")
                
                # Performance over time
                st.subheader("🏆 Win Rate Trend")
                games_df['win'] = (games_df['result'] == 'win').astype(int)
                games_df['game_number'] = range(len(games_df))
                
                fig = px.line(games_df, x='game_number', y='win', 
                             title="Win Rate Over Recent Games",
                             labels={'game_number': 'Game Number', 'win': 'Win (1) / Loss (0)'})
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("🎲 No game data available. Play some games to see analytics!")
        
        with tab2:
            st.header("🎯 Individual Game Analysis")
            
            games_df = asyncio.run(metrics.get_performance_trends(limit=50))
            if not games_df.empty:
                selected_game = st.selectbox(
                    "Select a game to analyze:",
                    games_df['game_id'].tolist(),
                    format_func=lambda x: f"{x} ({games_df[games_df['game_id']==x]['result'].iloc[0]})"
                )
                
                if selected_game:
                    game_summary = asyncio.run(metrics.get_game_summary(selected_game))
                    if game_summary:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🎮 Game Info")
                            st.write(f"**Result:** {game_summary.get('result', 'N/A')}")
                            st.write(f"**V7P3R Color:** {game_summary.get('v7p3r_color', 'N/A')}")
                            st.write(f"**Opponent:** {game_summary.get('opponent', 'N/A')}")
                            st.write(f"**Duration:** {game_summary.get('game_duration', 0):.1f}s")
                        
                        with col2:
                            st.subheader("📈 Performance Stats")
                            st.write(f"**Avg Evaluation:** {game_summary.get('avg_evaluation', 0):.2f}")
                            st.write(f"**Avg Search Depth:** {game_summary.get('avg_search_depth', 0):.1f}")
                            st.write(f"**Avg Nodes/Move:** {game_summary.get('avg_nodes_evaluated', 0):,.0f}")
                            st.write(f"**Avg Time/Move:** {game_summary.get('avg_time_per_move', 0):.3f}s")
            else:
                st.info("🎲 No games available for analysis.")
        
        with tab3:
            st.header("⚙️ Engine Configuration History")
            
            config_df = asyncio.run(metrics.get_engine_config_history())
            if not config_df.empty:
                st.dataframe(config_df, use_container_width=True)
                
                # Configuration trends
                st.subheader("📊 Configuration Trends")
                fig = px.line(config_df, x='timestamp', y='search_depth',
                             title="Search Depth Over Time")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("⚙️ No configuration history available.")
        
        with tab4:
            st.header("📈 Advanced Performance Trends")
            
            games_df = asyncio.run(metrics.get_performance_trends(limit=100))
            if not games_df.empty:
                # Multiple metrics
                metric_options = ['avg_evaluation', 'avg_depth', 'avg_nodes', 'avg_time']
                selected_metrics = st.multiselect(
                    "Select metrics to display:",
                    metric_options,
                    default=['avg_evaluation', 'avg_depth']
                )
                
                if selected_metrics:
                    fig = go.Figure()
                    
                    for metric in selected_metrics:
                        if metric in games_df.columns:
                            fig.add_trace(go.Scatter(
                                x=games_df.index,
                                y=games_df[metric],
                                mode='lines+markers',
                                name=metric.replace('avg_', '').replace('_', ' ').title()
                            ))
                    
                    fig.update_layout(
                        title="Performance Metrics Over Time",
                        xaxis_title="Game Number",
                        yaxis_title="Metric Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📈 No performance data available.")
        
        # Footer
        st.markdown("---")
        st.markdown("🎯 **V7P3R Chess Engine** | Analytics Dashboard v2.0")
        
    except ImportError:
        logger.error("Streamlit not installed. Install with: pip install streamlit")
        return False
    
    return True


def run_dashboard():
    """Run the Streamlit dashboard."""
    if create_streamlit_dashboard():
        logger.info("Dashboard created successfully. Run with: streamlit run this_file.py")
    else:
        logger.error("Failed to create dashboard. Make sure streamlit is installed.")


# Legacy compatibility functions (for smooth transition)
def add_game_result(game_id: str, timestamp: str, winner: str, game_pgn: str, 
                   white_player: str, black_player: str, game_length: int,
                   white_engine_config: str, black_engine_config: str):
    """Legacy compatibility function."""
    metrics = get_metrics_instance()
    game_metric = GameMetric(
        game_id=game_id,
        timestamp=timestamp,
        v7p3r_color='white' if white_player == 'v7p3r' else 'black',
        opponent=black_player if white_player == 'v7p3r' else white_player,
        result=winner,
        total_moves=game_length,
        game_duration=0.0,  # Will be updated later
        final_position_fen=None
    )
    # Use synchronous recording via run in thread
    try:
        # Try async first
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(metrics.record_game_start(game_metric))
        else:
            asyncio.run(metrics.record_game_start(game_metric))
    except RuntimeError:
        # Fallback: synchronous recording
        try:
            import sqlite3
            with sqlite3.connect(metrics.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO games 
                    (game_id, timestamp, v7p3r_color, opponent, result, total_moves,
                     game_duration, opening_name, final_position_fen, termination_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_metric.game_id, game_metric.timestamp, game_metric.v7p3r_color,
                    game_metric.opponent, game_metric.result, game_metric.total_moves,
                    game_metric.game_duration, game_metric.opening_name,
                    game_metric.final_position_fen, game_metric.termination_reason
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Legacy game recording failed: {e}")


def add_move_metric(game_id: str, move_number: int, player_color: str, 
                   move_uci: str, fen_before: str, evaluation: float,
                   search_algorithm: str, depth: int, nodes_searched: int,
                   time_taken: float, pv_line: str):
    """Legacy compatibility function."""
    metrics = get_metrics_instance()
    move_metric = MoveMetric(
        game_id=game_id,
        move_number=move_number,
        player='v7p3r' if search_algorithm != 'stockfish' else 'opponent',
        move_notation=move_uci,
        position_fen=fen_before,
        evaluation_score=evaluation,
        search_depth=depth,
        nodes_evaluated=nodes_searched,
        time_taken=time_taken,
        pv_line=pv_line
    )
    # Use synchronous recording via run in thread
    try:
        # Try async first
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(metrics.record_move(move_metric))
        else:
            asyncio.run(metrics.record_move(move_metric))
    except RuntimeError:
        # Fallback: synchronous recording
        try:
            import sqlite3
            with sqlite3.connect(metrics.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO moves 
                    (game_id, move_number, player, move_notation, position_fen,
                     evaluation_score, search_depth, nodes_evaluated, time_taken,
                     best_move, pv_line, quiescence_nodes, transposition_hits,
                     move_ordering_efficiency)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    move_metric.game_id, move_metric.move_number, move_metric.player,
                    move_metric.move_notation, move_metric.position_fen,
                    move_metric.evaluation_score, move_metric.search_depth,
                    move_metric.nodes_evaluated, move_metric.time_taken,
                    move_metric.best_move, move_metric.pv_line,
                    move_metric.quiescence_nodes, move_metric.transposition_hits,
                    move_metric.move_ordering_efficiency
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Legacy move recording failed: {e}")


# Main entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        run_dashboard()
    else:
        # Default behavior - just initialize metrics system
        metrics = get_metrics_instance()
        logger.info("V7P3R Metrics system initialized successfully")
        logger.info("To run dashboard: python chess_metrics.py dashboard")