# metrics/v7p3r_metrics.py
"""
V7P3R Chess Engine Metrics System
A unified, modern metrics collection and analysis system for the V7P3R chess engine.
Features async data collection, Streamlit dashboard, and robust database handling.
"""

import os
import site
import sys

# Add site-packages to path
site_packages = site.getsitepackages()
for path in site_packages:
    if path not in sys.path:
        sys.path.append(path)

import asyncio
import sqlite3
import json
import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Any, List
from pathlib import Path
import pandas as pd

# Update dashboard availability check
try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    print(f"Dashboard import error: {str(e)}")
    print(f"Python executable: {sys.executable}")
    print(f"sys.path: {sys.path}")
    DASHBOARD_AVAILABLE = False
    
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Database path
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "chess_metrics.db"

@dataclass
class GameMetric:
    """Data structure for game-level metrics matching database schema."""
    game_id: str  # PRIMARY KEY
    timestamp: str
    v7p3r_color: str
    opponent: str
    result: str
    total_moves: int
    game_duration: float
    opening_name: Optional[str] = None
    final_position_fen: Optional[str] = None
    termination_reason: Optional[str] = None

@dataclass
class MoveMetric:
    """Data structure for move-level metrics matching database schema."""
    game_id: str
    move_number: int
    player: str
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
    """Data structure for engine configuration matching database schema."""
    config_id: str  # PRIMARY KEY
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
    
    def _initialize_database(self):
        """Initialize database tables with exact schema."""
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
            return True
        except Exception as e:
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
            return True
        except Exception as e:
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
            return True
        except Exception as e:
            return False

    
    async def update_game_metadata(self, game_id: str, metadata: Dict[str, Any]) -> bool:
        """Update additional game metadata fields."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Build the update query dynamically based on provided metadata
                    valid_columns = {'opening_name', 'final_position_fen', 'termination_reason'}
                    updates = []
                    values = []
                    
                    for key, value in metadata.items():
                        if key in valid_columns:
                            updates.append(f"{key} = ?")
                            values.append(value)
                    
                    if not updates:
                        return False
                        
                    values.append(game_id)  # For WHERE clause
                    update_query = f"UPDATE games SET {', '.join(updates)} WHERE game_id = ?"
                    
                    cursor.execute(update_query, values)
                    conn.commit()
                    return True
                    
        except Exception as e:
            print(f"Error updating game metadata: {str(e)}")
            return False
    
    # Analysis methods
    async def get_game_summary(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive summary for a specific game with enhanced metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get game info
                cursor.execute("""
                    SELECT g.*,
                           COUNT(DISTINCT m.move_number) as total_moves,
                           AVG(CASE WHEN m.player = 'v7p3r' THEN m.evaluation_score END) as avg_evaluation,
                           AVG(CASE WHEN m.player = 'v7p3r' THEN m.search_depth END) as avg_depth,
                           AVG(CASE WHEN m.player = 'v7p3r' THEN m.nodes_evaluated END) as avg_nodes,
                           AVG(CASE WHEN m.player = 'v7p3r' THEN m.time_taken END) as avg_time,
                           AVG(CASE WHEN m.player = 'v7p3r' THEN m.quiescence_nodes END) as avg_quiescence_nodes,
                           AVG(CASE WHEN m.player = 'v7p3r' THEN m.transposition_hits END) as avg_transposition_hits,
                           AVG(CASE WHEN m.player = 'v7p3r' THEN m.move_ordering_efficiency END) as avg_move_ordering
                    FROM games g
                    LEFT JOIN moves m ON g.game_id = m.game_id
                    WHERE g.game_id = ?
                    GROUP BY g.game_id
                """, (game_id,))
                
                game_row = cursor.fetchone()
                if not game_row:
                    return None
                
                # Get column names including the calculated fields
                cursor.execute("PRAGMA table_info(games)")
                game_columns = [col[1] for col in cursor.fetchall()]
                game_columns.extend([
                    'total_moves', 'avg_evaluation', 'avg_depth', 'avg_nodes',
                    'avg_time', 'avg_quiescence_nodes', 'avg_transposition_hits',
                    'avg_move_ordering'
                ])
                
                game_data = dict(zip(game_columns, game_row))
                
                # Get move sequence
                cursor.execute("""
                    SELECT move_notation, position_fen, evaluation_score,
                           search_depth, nodes_evaluated, time_taken,
                           best_move, pv_line
                    FROM moves
                    WHERE game_id = ?
                    ORDER BY move_number ASC
                """, (game_id,))
                
                moves_data = cursor.fetchall()
                if moves_data:
                    game_data['moves'] = [{
                        'notation': move[0],
                        'position': move[1],
                        'evaluation': move[2],
                        'depth': move[3],
                        'nodes': move[4],
                        'time': move[5],
                        'best_move': move[6],
                        'pv_line': move[7]
                    } for move in moves_data]
                
                return game_data
                
        except Exception as e:
            print(f"Error in get_game_summary: {str(e)}")
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
    """Create a Streamlit dashboard for visualizing chess engine metrics."""
    if not DASHBOARD_AVAILABLE:
        print("Dashboard requires additional dependencies. Install with:")
        print("pip install streamlit plotly")
        return False
        
    try:
        # Streamlit configuration
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
        
        # Sidebar filters
        st.sidebar.header("🎛️ Dashboard Controls")
        num_games = st.sidebar.slider("Number of games to analyze", min_value=5, max_value=100, value=20)
        
        # Main dashboard sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Game Analysis", "⚙️ Engine Config", "📈 Performance Trends"])
        
        with tab1:
            st.header("📊 Engine Performance Overview")
            
            # Get recent games with error handling
            try:
                games_df = asyncio.run(metrics.get_performance_trends(limit=num_games))
                games_df = games_df.fillna(0)  # Replace NaN with 0 for numeric calculations
            except Exception as e:
                st.error(f"Error loading game data: {str(e)}")
                games_df = pd.DataFrame()
            
            if not games_df.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_games = len(games_df)
                    st.metric("Total Games", total_games)
                
                with col2:
                    wins = len(games_df[games_df['result'].str.lower() == 'win'])
                    win_rate = (wins / total_games * 100) if total_games > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                with col3:
                    avg_depth = games_df['avg_depth'].mean()
                    st.metric("Avg Search Depth", f"{avg_depth:.1f}" if pd.notnull(avg_depth) else "N/A")
                
                with col4:
                    avg_nodes = games_df['avg_nodes'].mean()
                    st.metric("Avg Nodes/Move", f"{avg_nodes:,.0f}" if pd.notnull(avg_nodes) else "N/A")
                
                # Performance over time chart
                st.subheader("🏆 Performance Trends")
                
                # Convert game results to numeric values for plotting
                games_df['result_numeric'] = games_df['result'].map({'win': 1, 'draw': 0.5, 'loss': 0})
                
                # Create a rolling average of results
                games_df['rolling_performance'] = games_df['result_numeric'].rolling(window=5, min_periods=1).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=games_df.index,
                    y=games_df['rolling_performance'],
                    mode='lines+markers',
                    name='Performance (5-game rolling avg)',
                    line=dict(color='#00ff00', width=2),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Performance Trend (1.0 = Win, 0.5 = Draw, 0.0 = Loss)",
                    xaxis_title="Game Number",
                    yaxis_title="Performance",
                    yaxis=dict(range=[0, 1]),
                    hovermode='x unified'
                )
                
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
                
                # Detailed game analysis
                if selected_game:
                    game_summary = asyncio.run(metrics.get_game_summary(selected_game))
                    if game_summary:
                        st.subheader("📊 Game Details")
                        
                        # Layout with columns
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            st.markdown("### 🎮 Game Info")
                            st.write(f"**Result:** {game_summary.get('result', 'N/A')}")
                            st.write(f"**V7P3R Color:** {game_summary.get('v7p3r_color', 'N/A')}")
                            st.write(f"**Opponent:** {game_summary.get('opponent', 'N/A')}")
                            st.write(f"**Duration:** {game_summary.get('game_duration', 0.0):.1f}s")
                            st.write(f"**Total Moves:** {game_summary.get('total_moves', 0)}")
                        
                        with col2:
                            st.markdown("### 📈 Performance")
                            eval_score = game_summary.get('avg_evaluation')
                            st.write(f"**Avg Evaluation:** {eval_score:.2f if eval_score is not None else 'N/A'}")
                            depth = game_summary.get('avg_search_depth')
                            st.write(f"**Avg Search Depth:** {depth:.1f if depth is not None else 'N/A'}")
                            nodes = game_summary.get('avg_nodes_evaluated')
                            st.write(f"**Avg Nodes/Move:** {nodes:,.0f if nodes is not None else 'N/A'}")
                            
                        with col3:
                            st.markdown("### ⚡ Engine Stats")
                            time_per_move = game_summary.get('avg_time_per_move')
                            st.write(f"**Avg Time/Move:** {time_per_move:.3f if time_per_move is not None else 'N/A'}s")
                            if game_summary.get('opening_name'):
                                st.write(f"**Opening:** {game_summary['opening_name']}")
                            if game_summary.get('termination_reason'):
                                st.write(f"**Termination:** {game_summary['termination_reason']}")
                        
                        # Add final position display if available
                        if game_summary.get('final_position_fen'):
                            st.subheader("📋 Final Position")
                            st.code(game_summary['final_position_fen'])
                    else:
                        st.warning("⚠️ Could not load detailed game data")
            else:
                st.info("🎲 No games available for analysis.")
        
        with tab3:
            st.header("⚙️ Engine Configuration History")
            
            config_df = asyncio.run(metrics.get_engine_config_history())
            if not config_df.empty:
                st.subheader("📊 Current Configuration")
                
                # Get most recent config
                latest_config = config_df.iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🔍 Search Settings")
                    st.write(f"**Search Depth:** {latest_config['search_depth']}")
                    st.write(f"**Time Limit:** {latest_config['time_limit']}s")
                    st.write(f"**Hash Size:** {latest_config['hash_size_mb']} MB")
                
                with col2:
                    st.markdown("### 🛠️ Features")
                    st.write(f"**Iterative Deepening:** {'✅' if latest_config['use_iterative_deepening'] else '❌'}")
                    st.write(f"**Transposition Table:** {'✅' if latest_config['use_transposition_table'] else '❌'}")
                    st.write(f"**Quiescence Search:** {'✅' if latest_config['use_quiescence_search'] else '❌'}")
                    st.write(f"**Move Ordering:** {'✅' if latest_config['use_move_ordering'] else '❌'}")
                
                st.subheader("📈 Configuration History")
                
                # Create trends for numeric parameters
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=config_df['timestamp'],
                    y=config_df['search_depth'],
                    mode='lines+markers',
                    name='Search Depth'
                ))
                fig.update_layout(title='Search Depth History')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show full configuration history
                st.dataframe(config_df.style.format({
                    'time_limit': '{:.1f}',
                    'hash_size_mb': '{:,.0f}'
                }))
            else:
                st.info("⚙️ No configuration history available.")
        
        with tab4:
            st.header("📈 Advanced Performance Analysis")
            
            games_df = asyncio.run(metrics.get_performance_trends(limit=num_games))
            if not games_df.empty:
                # Add rolling averages for key metrics
                metrics_to_plot = {
                    'avg_evaluation': 'Average Evaluation',
                    'avg_depth': 'Average Search Depth',
                    'avg_nodes': 'Average Nodes Evaluated',
                    'avg_time': 'Average Time per Move'
                }
                
                # Let user select metrics to display
                selected_metrics = st.multiselect(
                    "Select metrics to analyze:",
                    list(metrics_to_plot.keys()),
                    default=['avg_evaluation', 'avg_depth']
                )
                
                if selected_metrics:
                    fig = go.Figure()
                    
                    for metric in selected_metrics:
                        # Create rolling average
                        rolling_data = games_df[metric].rolling(window=5, min_periods=1).mean()
                        
                        # Use a simpler hovertemplate to avoid linting issues
                        fig.add_trace(go.Scatter(
                            x=games_df.index,
                            y=rolling_data,
                            mode='lines+markers',
                            name=metrics_to_plot[metric],
                            customdata=[[metrics_to_plot[metric], val] for val in rolling_data],
                            hovertemplate='%{customdata[0]}: %{customdata[1]:.2f}<br>Game: %{x}<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title="Performance Metrics (5-game rolling average)",
                        xaxis_title="Game Number",
                        yaxis_title="Metric Value",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show summary statistics
                    st.subheader("📊 Summary Statistics")
                    summary_stats = games_df[selected_metrics].describe()
                    st.dataframe(summary_stats.style.format("{:.2f}"))
            else:
                st.info("📈 No performance data available for analysis.")
        
        # Footer
        st.markdown("---")
        st.markdown("🎯 **V7P3R Chess Engine** | Analytics Dashboard v2.1")
        
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        return False
    
    return True

def run_dashboard():
    """Run the Streamlit dashboard."""
    try:
        # Get absolute path to this script
        script_path = os.path.abspath(__file__)
        
        # Use subprocess to properly launch Streamlit
        import subprocess
        subprocess.run(["streamlit", "run", script_path], check=True)
        return True
        
    except Exception as e:
        print(f"Error launching dashboard: {str(e)}")
        print("Try running directly with: streamlit run v7p3r_metrics.py")
        return False

# Main entry point 
if __name__ == "__main__":
    if "streamlit" in sys.modules:
        # Being run by Streamlit
        create_streamlit_dashboard()
    else:
        # Direct script execution
        run_dashboard()