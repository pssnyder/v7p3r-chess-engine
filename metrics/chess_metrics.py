# metrics/chess_metrics.py
""" A locally running dashboard to visualize chess engine tuning metrics.
This dashboard uses Dash and Plotly to display metrics from chess games played by the V7P3R chess engine.
It includes static metrics, A/B testing for dynamic metrics, and a dark mode theme. 
"""

import dash
from dash import dcc, html, Output, Input
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from metrics_store import MetricsStore
from metrics_backup import backup_metrics_db
import yaml
import atexit
from datetime import datetime # Import datetime for parsing timestamps
import sqlite3

ANALYTICS_DB_PATH = "metrics/chess_analytics.db"
RAW_DB_PATH = "metrics/chess_metrics.db"

# Initialize the metrics store globally
metrics_store = MetricsStore(db_path=ANALYTICS_DB_PATH)

# Load engine search algorithms and config from v7p3r_config.yaml
try:
    with open("config/v7p3r_config.yaml", "r") as v7p3r_config_file:
        v7p3r_config_data = yaml.safe_load(v7p3r_config_file)
        search_algorithms = v7p3r_config_data.get("search_algorithms", [])
        v7p3r_engine_config = v7p3r_config_data.get("v7p3r", {})
except Exception as e:
    print(f"Error loading v7p3r_config.yaml for engine config: {e}")
    search_algorithms = []
    v7p3r_engine_config = {}

# Load only game settings and engine names from chess_game_config.yaml
try:
    with open("config/chess_game_config.yaml", "r") as config_file:
        chess_game_config_data = yaml.safe_load(config_file)
        white_engine_name = chess_game_config_data.get("white_engine_config", {}).get("engine", None)
        black_engine_name = chess_game_config_data.get("black_engine_config", {}).get("engine", None)
        # You can also load other game-level settings if needed
except Exception as e:
    print(f"Error loading chess_game_config.yaml for game settings: {e}")
    white_engine_name = None
    black_engine_name = None

# --- DARK MODE COLORS ---
DARK_BG = "#18191A"
DARK_PANEL = "#242526"
DARK_ACCENT = "#3A3B3C"
DARK_TEXT = "#E4E6EB"
DARK_SUBTEXT = "#B0B3B8"
DARK_BORDER = "#3A3B3C"
DARK_HIGHLIGHT = "#4A90E2"
DARK_ERROR = "#FF5252"
DARK_WARNING = "#FFB300"
DARK_SUCCESS = "#00C853"

# --- Metrics Dashboard Initialization: Backup, Cleanup, and Test Data ---
def initialize_metrics_dashboard():
    # 1. Backup the database
    backup_path = backup_metrics_db()
    print(f"Metrics DB backed up to: {backup_path}")
    # 2. Clean up incomplete games (where result is not 1-0, 0-1, or 1/2-1/2)
    connection = metrics_store._get_connection()
    with connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM game_results WHERE winner NOT IN ('1-0', '0-1', '1/2-1/2') OR winner IS NULL")
        connection.commit()
    # 3. Insert test records for white and black under a metrics-test engine name
    test_game_id = f"metrics_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics_store.add_game_result(
        game_id=test_game_id,
        timestamp=datetime.now().isoformat(),
        winner='1-0',
        game_pgn="[Event 'Test']\n[Site 'Local']\n[Date '2025.06.12']\n[Round '-']\n[White 'metrics-test']\n[Black 'metrics-test']\n[Result '1-0']\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0",
        white_player='metrics-test',
        black_player='metrics-test',
        game_length=10,
        white_engine_config={'engine': 'metrics-test', 'exclude_from_metrics': False},
        black_engine_config={'engine': 'metrics-test', 'exclude_from_metrics': False}
    )
    # Add a few test moves
    metrics_store.add_move_metric(game_id=test_game_id, move_number=1, player_color='white', move_uci='e2e4', fen_before='rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1', evaluation=0.1, search_algorithm='metrics-test', depth=1, nodes_searched=10, time_taken=0.01, pv_line='e2e4 e7e5')
    metrics_store.add_move_metric(game_id=test_game_id, move_number=1, player_color='black', move_uci='e7e5', fen_before='rnbqkbnr/pppppppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2', evaluation=0.0, search_algorithm='metrics-test', depth=1, nodes_searched=10, time_taken=0.01, pv_line='e7e5 Nf3')
    print("Test metrics data initialized.")

# Call initialization at module load
initialize_metrics_dashboard()

# ETL: Incrementally copy new data from chess_metrics.db to chess_analytics.db, keeping old analytics data

def etl_to_analytics_db():
    raw_conn = sqlite3.connect(RAW_DB_PATH)
    analytics_conn = sqlite3.connect(ANALYTICS_DB_PATH)
    raw_cur = raw_conn.cursor()
    analytics_cur = analytics_conn.cursor()

    # Ensure analytics tables exist with correct schema (copy structure if empty)
    analytics_cur.execute('''CREATE TABLE IF NOT EXISTS game_results AS SELECT * FROM game_results WHERE 0''')
    analytics_cur.execute('''CREATE TABLE IF NOT EXISTS move_metrics AS SELECT * FROM move_metrics WHERE 0''')

    # --- Incremental for game_results ---
    # Get all game_ids already in analytics
    analytics_cur.execute('SELECT game_id FROM game_results')
    existing_game_ids = set(row[0] for row in analytics_cur.fetchall())
    # Get only new game_results from raw
    raw_cur.execute('SELECT * FROM game_results')
    raw_games = raw_cur.fetchall()
    raw_games_cols = [desc[0] for desc in raw_cur.description]
    new_games = [row for row in raw_games if row[raw_games_cols.index('game_id')] not in existing_game_ids]
    if new_games:
        placeholders = ','.join(['?'] * len(raw_games_cols))
        analytics_cur.executemany(f'INSERT INTO game_results ({','.join(raw_games_cols)}) VALUES ({placeholders})', new_games)

    # --- Incremental for move_metrics ---
    # Get all (game_id, move_number, player_color) already in analytics
    analytics_cur.execute('SELECT game_id, move_number, player_color FROM move_metrics')
    existing_moves = set((row[0], row[1], row[2]) for row in analytics_cur.fetchall())
    # Get only new move_metrics from raw
    raw_cur.execute('SELECT * FROM move_metrics')
    raw_moves = raw_cur.fetchall()
    raw_moves_cols = [desc[0] for desc in raw_cur.description]
    new_moves = [row for row in raw_moves if (row[raw_moves_cols.index('game_id')], row[raw_moves_cols.index('move_number')], row[raw_moves_cols.index('player_color')]) not in existing_moves]
    if new_moves:
        placeholders = ','.join(['?'] * len(raw_moves_cols))
        analytics_cur.executemany(f'INSERT INTO move_metrics ({','.join(raw_moves_cols)}) VALUES ({placeholders})', new_moves)

    analytics_conn.commit()
    raw_conn.close()
    analytics_conn.close()

# Run ETL at startup
etl_to_analytics_db()

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    # Inject global CSS to remove body margin and set background
    dcc.Markdown(
        """
        <style>
            body { margin: 0 !important; background: #18191A !important; }
        </style>
        """,
        dangerously_allow_html=True
    ),
    html.Div([
        html.H1("V7P3R Chess Engine Tuning Dashboard", style={"textAlign": "center", "marginBottom": "20px", "color": DARK_TEXT}),
    ], style={"backgroundColor": DARK_BG}),

    # Main interval for refreshing data
    dcc.Interval(id="interval", interval=8000, n_intervals=0),

    # Static metrics section
    html.Div([
        html.Div([
            html.H2("Static Metrics", style={"textAlign": "center", "color": DARK_TEXT}),
            html.Div(id="static-metrics", style={"padding": "10px", "backgroundColor": DARK_PANEL, "borderRadius": "5px", "color": DARK_TEXT}),
        ], style={"flex": "1", "marginRight": "20px"}),
        html.Div([
            dcc.Graph(id="static-trend-graph", style={"height": "300px", "backgroundColor": DARK_PANEL}),
        ], style={"flex": "2"}),
    ], style={"display": "flex", "flexDirection": "row", "alignItems": "flex-start", "marginBottom": "30px"}),

    # A/B Testing & Dynamic Metric Filters
    html.Div([
        html.H2("A/B Testing & Metric Trends", style={"textAlign": "center", "marginBottom": "15px", "color": DARK_TEXT}),
        html.Div([
                html.Label("Metric to Plot (for v7p3r Engine):", style={"color": DARK_TEXT}),
                dcc.Dropdown(
                    id="dynamic-metric-selector",
                    options=[], # Populated dynamically
                    placeholder="Select a metric (e.g., eval, nodes)",
                    style={"backgroundColor": DARK_PANEL, "color": DARK_TEXT, "borderColor": DARK_BORDER}
                )
            ], style={'width': '90%', 'display': 'inline-block', 'marginLeft': 'auto', 'marginRight': 'auto', 'verticalAlign': 'top'}), # Adjusted width and centering
        ], style={'marginBottom': '15px', 'textAlign': 'center'}), # Centering the dropdown div
        
        dcc.Graph(id="ab-test-trend-graph", style={"height": "400px", "backgroundColor": DARK_PANEL}),
        html.Div(id="move-metrics-details", style={"padding": "10px", "backgroundColor": DARK_PANEL, "borderRadius": "5px", "marginTop": "20px", "color": DARK_TEXT}),

    ], style={"marginBottom": "30px", "border": f"1px solid {DARK_BORDER}", "padding": "15px", "borderRadius": "8px", "backgroundColor": DARK_ACCENT}),


    # Footer

@app.callback(
    Output("dynamic-metric-selector", "options"),
    Input("interval", "n_intervals"), # Changed from log-analysis-interval
)
def update_dynamic_metric_options(_):
    # Only show metrics that are actually useful for engine performance
    # These are the numeric columns in move_metrics: evaluation, depth, nodes_searched, time_taken
    # You may want to add more if you add more numeric columns in the future
    options = [
        {"label": "Evaluation Score", "value": "evaluation"},
        {"label": "Search Depth", "value": "depth"},
        {"label": "Nodes Searched", "value": "nodes_searched"},
        {"label": "Time Taken (s)", "value": "time_taken"},
    ]
    return options

@app.callback(
    [Output("ab-test-trend-graph", "figure"),
     Output("move-metrics-details", "children")],
    [Input("interval", "n_intervals"), # Triggered by the main interval
     Input("dynamic-metric-selector", "value")]
)
def update_ab_testing_section(_, selected_metric):
    fig_ab_test = go.Figure()
    move_metrics_details_components = []

    if not selected_metric:
        fig_ab_test.update_layout(
            title="Select a Metric to Plot for V7P3R Engine",
            paper_bgcolor=DARK_PANEL,
            plot_bgcolor=DARK_PANEL,
            font=dict(color=DARK_TEXT),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        fig_ab_test.add_annotation(text="Please select a metric from the dropdown.", xref="paper", yref="paper", showarrow=False, font=dict(size=16, color=DARK_TEXT))
        move_metrics_details_components.append(html.P("Please select a metric from the dropdown to visualize V7P3R's performance trend.", style={"color": DARK_TEXT}))
        return fig_ab_test, move_metrics_details_components    

    v7p3r_search_algorithms = search_algorithms
    
    all_v7p3r_moves_raw = metrics_store.get_filtered_move_metrics(
        white_engine_names=['v7p3r'],  # Use engine name from config
        black_engine_names=['v7p3r'],
        metric_name=selected_metric
    )
    if not all_v7p3r_moves_raw:
        fig_ab_test.update_layout(
            title=f"No '{selected_metric}' Data for v7p3r Engine",
            paper_bgcolor=DARK_PANEL, plot_bgcolor=DARK_PANEL, font=dict(color=DARK_TEXT),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        fig_ab_test.add_annotation(text=f"No move data found for v7p3r playing '{selected_metric}'.", xref="paper", yref="paper", showarrow=False, font=dict(size=16, color=DARK_TEXT))
        move_metrics_details_components.append(html.P(f"No move metrics data found for v7p3r for the metric: {selected_metric}.", style={"color": DARK_TEXT}))
        return fig_ab_test, move_metrics_details_components

    df_all_v7p3r_moves = pd.DataFrame(all_v7p3r_moves_raw)
    df_games_raw = metrics_store.get_all_game_results_df()

    if df_games_raw is None or df_games_raw.empty:
        # Handle case where there are no game results to filter by
        fig_ab_test.update_layout(title="No Game Results Available for Context", paper_bgcolor=DARK_PANEL, plot_bgcolor=DARK_PANEL, font=dict(color=DARK_TEXT))
        move_metrics_details_components.append(html.P("No game results found to provide context for V7P3R's moves.", style={"color": DARK_TEXT}))
        return fig_ab_test, move_metrics_details_components

    # Determine which V7P3R moves to analyze based on game context
    v7p3r_perspectives = []
    for _, game_row in df_games_raw.iterrows():
        game_id = game_row['game_id']
        white_is_v7p3r = game_row.get('white_search_algorithm') == 'v7p3r'
        black_is_v7p3r = game_row.get('black_search_algorithm') == 'v7p3r'
        exclude_white = game_row.get('exclude_white_from_metrics', False)
        exclude_black = game_row.get('exclude_black_from_metrics', False)
        winner = game_row.get('winner')

        if white_is_v7p3r and black_is_v7p3r: # V7P3R vs V7P3R
            if winner == '1-0' and not exclude_white: # White V7P3R won
                v7p3r_perspectives.append({'game_id': game_id, 'v7p3r_color_to_analyze': 'white'})
            elif winner == '0-1' and not exclude_black: # Black V7P3R won
                v7p3r_perspectives.append({'game_id': game_id, 'v7p3r_color_to_analyze': 'black'})
            # Moves from drawn V7P3R vs V7P3R or losing V7P3R are excluded for this trend
        elif white_is_v7p3r and not exclude_white: # V7P3R (White) vs Non-V7P3R
            v7p3r_perspectives.append({'game_id': game_id, 'v7p3r_color_to_analyze': 'white'})
        elif black_is_v7p3r and not exclude_black: # V7P3R (Black) vs Non-V7P3R
            v7p3r_perspectives.append({'game_id': game_id, 'v7p3r_color_to_analyze': 'black'})
            
    if not v7p3r_perspectives:
        fig_ab_test.update_layout(title=f"No Valid Game Context for V7P3R's '{selected_metric}'", paper_bgcolor=DARK_PANEL, plot_bgcolor=DARK_PANEL, font=dict(color=DARK_TEXT))
        move_metrics_details_components.append(html.P("No games found where V7P3R's performance can be analyzed based on current criteria.", style={"color": DARK_TEXT}))
        return fig_ab_test, move_metrics_details_components

    df_v7p3r_perspectives = pd.DataFrame(v7p3r_perspectives)

    # Merge V7P3R moves with the determined perspectives
    df_merged_moves = pd.merge(df_all_v7p3r_moves, df_v7p3r_perspectives, on='game_id', how='inner')

    # Filter moves to only those matching the 'v7p3r_color_to_analyze'
    # Ensure 'player_color' in df_merged_moves is comparable (e.g., 'white' or 'w')
    # Assuming 'player_color' in move_metrics is 'white' or 'black'
    df_final_moves = df_merged_moves[df_merged_moves['player_color'].str.lower() == df_merged_moves['v7p3r_color_to_analyze'].str.lower()]

    if df_final_moves.empty:
        fig_ab_test.update_layout(
            title=f"No '{selected_metric}' Data for V7P3R (Post-Filtering)",
            paper_bgcolor=DARK_PANEL, plot_bgcolor=DARK_PANEL, font=dict(color=DARK_TEXT),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        fig_ab_test.add_annotation(text=f"No '{selected_metric}' moves from V7P3R met the analysis criteria (e.g., winning side in V7P3R vs V7P3R).", xref="paper", yref="paper", showarrow=False, font=dict(size=16, color=DARK_TEXT))
        move_metrics_details_components.append(html.P(f"No moves for '{selected_metric}' by V7P3R met the specific analysis criteria.", style={"color": DARK_TEXT}))
        return fig_ab_test, move_metrics_details_components

    df_final_moves = df_final_moves.copy() # Avoid SettingWithCopyWarning
    df_final_moves['created_at_dt'] = pd.to_datetime(df_final_moves['created_at'])
    df_final_moves = df_final_moves.sort_values('created_at_dt')

    fig_ab_test.add_trace(go.Scatter(
        x=df_final_moves['created_at_dt'],
        y=df_final_moves[selected_metric],
        mode='markers',
        name=f'V7P3R {selected_metric.replace("_", " ").title()}',
        marker=dict(size=6, opacity=0.7, color=DARK_HIGHLIGHT)
    ))
    
    if len(df_final_moves) > 1:
        df_final_moves['time_numeric'] = (df_final_moves['created_at_dt'] - df_final_moves['created_at_dt'].min()).dt.total_seconds()
        # Ensure selected_metric column is numeric for polyfit
        numeric_metric_values = pd.to_numeric(df_final_moves[selected_metric], errors='coerce').dropna()
        numeric_time_values = df_final_moves.loc[numeric_metric_values.index, 'time_numeric']

        # Convert to numpy arrays for polyfit
        x = np.array(numeric_time_values)
        y = np.array(numeric_metric_values)
        if len(x) > 1 and len(y) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            fig_ab_test.add_trace(go.Scatter(
                x=df_final_moves.loc[numeric_metric_values.index, 'created_at_dt'],
                y=p(x),
                mode='lines',
                name='Trendline',
                line=dict(color=DARK_SUCCESS, dash='dash')
            ))

    fig_ab_test.update_layout(
        title=f"V7P3R Engine: {selected_metric.replace('_', ' ').title()} Trend",
        xaxis_title="Time of Move",
        yaxis_title=selected_metric.replace('_', ' ').title(),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor=DARK_PANEL,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=DARK_TEXT),
        xaxis=dict(gridcolor=DARK_ACCENT),
        yaxis=dict(gridcolor=DARK_ACCENT)
    )

    move_metrics_details_components.append(html.H4(f"V7P3R Summary for {selected_metric.replace('_', ' ').title()}", style={"color": DARK_TEXT}))
    
    metric_series = pd.to_numeric(df_final_moves[selected_metric], errors='coerce').dropna()
    if not metric_series.empty:
        avg_val = metric_series.mean()
        min_val = metric_series.min()
        max_val = metric_series.max()
        count_val = metric_series.count()
        std_val = metric_series.std()
        num_games = len(df_final_moves['game_id'].unique())

        move_metrics_details_components.append(html.Ul([
            html.Li(f"Number of Games Analyzed: {num_games}", style={"color": DARK_SUBTEXT}),
            html.Li(f"Total Data Points (Moves): {count_val}", style={"color": DARK_SUBTEXT}),
            html.Li(f"Average: {avg_val:.3f}", style={"color": DARK_SUBTEXT}),
            html.Li(f"Min: {min_val:.3f}", style={"color": DARK_SUBTEXT}),
            html.Li(f"Max: {max_val:.3f}", style={"color": DARK_SUBTEXT}),
            html.Li(f"Standard Deviation: {std_val:.3f}", style={"color": DARK_SUBTEXT}),
        ], style={"listStyleType": "none", "paddingLeft": "0"}))
    else:
        move_metrics_details_components.append(html.P("No numeric data available for summary statistics.", style={"color": DARK_SUBTEXT}))


    return fig_ab_test, move_metrics_details_components


# Update the static metrics to use stored data
@app.callback(
    [Output("static-metrics", "children"),
     Output("static-trend-graph", "figure")],
    Input("interval", "n_intervals"),
)
def update_static_metrics(_):
    df_games_raw = metrics_store.get_all_game_results_df()
    
    fig_static_trend = go.Figure()
    
    if df_games_raw is None or df_games_raw.empty:
        metrics_display = html.Ul([
            html.Li("Total V7P3R Games: 0", style={"color": DARK_SUBTEXT}),
            html.Li("V7P3R Wins: 0", style={"color": DARK_SUBTEXT}),
            html.Li("V7P3R Losses: 0", style={"color": DARK_SUBTEXT}),
            html.Li("V7P3R Draws: 0", style={"color": DARK_SUBTEXT}),
        ], style={"listStyleType": "none", "paddingLeft": "0"})
        
        fig_static_trend.update_layout(
            title="No Game Result Data Available for V7P3R",
            height=300, margin=dict(t=40, b=20, l=30, r=20),
            paper_bgcolor=DARK_PANEL, plot_bgcolor=DARK_PANEL, font=dict(color=DARK_TEXT),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        fig_static_trend.add_annotation(text="No game data for V7P3R to display.", xref="paper", yref="paper", showarrow=False, font=dict(size=16, color=DARK_TEXT))
        return metrics_display, fig_static_trend

    # Filter games relevant to V7P3R and respect exclusion flags
    v7p3r_games_data = []
    for _, row in df_games_raw.iterrows():
        white_is_v7p3r = row.get('white_search_algorithm') == 'v7p3r'
        black_is_v7p3r = row.get('black_search_algorithm') == 'v7p3r'
        exclude_white = row.get('exclude_white_from_metrics', False)
        exclude_black = row.get('exclude_black_from_metrics', False)
        
        is_v7p3r_game_and_not_excluded = False
        if white_is_v7p3r and not exclude_white:
            is_v7p3r_game_and_not_excluded = True
        if black_is_v7p3r and not exclude_black: # Can be true even if white is also v7p3r and not excluded
            is_v7p3r_game_and_not_excluded = True
            
        if is_v7p3r_game_and_not_excluded:
            v7p3r_games_data.append(row)
    
    if not v7p3r_games_data:
        # Same display as no raw data, but specific to V7P3R after filtering
        metrics_display = html.Ul([
            html.Li("Total V7P3R Games: 0 (after exclusion)", style={"color": DARK_SUBTEXT}),
            html.Li("V7P3R Wins: 0", style={"color": DARK_SUBTEXT}),
            html.Li("V7P3R Losses: 0", style={"color": DARK_SUBTEXT}),
            html.Li("V7P3R Draws: 0", style={"color": DARK_SUBTEXT}),
        ], style={"listStyleType": "none", "paddingLeft": "0"})
        fig_static_trend.update_layout(title="No V7P3R Games Meet Criteria", height=300, paper_bgcolor=DARK_PANEL, plot_bgcolor=DARK_PANEL, font=dict(color=DARK_TEXT))
        fig_static_trend.add_annotation(text="No V7P3R games to display after applying exclusion filters.", xref="paper", yref="paper", showarrow=False, font=dict(size=16, color=DARK_TEXT))
        return metrics_display, fig_static_trend

    df_v7p3r_games = pd.DataFrame(v7p3r_games_data)
    
    total_v7p3r_games = len(df_v7p3r_games)
    v7p3r_wins = 0
    v7p3r_losses = 0
    v7p3r_draws = 0

    for _, row in df_v7p3r_games.iterrows():
        winner = row.get('winner')
        white_is_v7p3r_and_included = row.get('white_engine') == 'v7p3r' and not row.get('exclude_white_from_metrics', False)
        black_is_v7p3r_and_included = row.get('black_engine') == 'v7p3r' and not row.get('exclude_black_from_metrics', False)

        if winner == '1-0':
            if white_is_v7p3r_and_included: v7p3r_wins +=1
            if black_is_v7p3r_and_included: v7p3r_losses +=1
        elif winner == '0-1':
            if black_is_v7p3r_and_included: v7p3r_wins +=1
            if white_is_v7p3r_and_included: v7p3r_losses +=1
        elif winner == '1/2-1/2':
            # A draw counts if V7P3R played and was not excluded
            if white_is_v7p3r_and_included or black_is_v7p3r_and_included:
                 v7p3r_draws += 1
    
    metrics_display = html.Ul([
        html.Li(f"Total V7P3R Games (Analyzed): {total_v7p3r_games}", style={"color": DARK_SUBTEXT}),
        html.Li(f"V7P3R Wins: {v7p3r_wins}", style={"color": DARK_SUBTEXT}),
        html.Li(f"V7P3R Losses: {v7p3r_losses}", style={"color": DARK_SUBTEXT}),
        html.Li(f"V7P3R Draws: {v7p3r_draws}", style={"color": DARK_SUBTEXT}),
    ], style={"listStyleType": "none", "paddingLeft": "0"})
    
    # Trend graph for V7P3R game results
    if not df_v7p3r_games.empty:
        df_v7p3r_games = df_v7p3r_games.copy() # Avoid SettingWithCopyWarning
        try:
            df_v7p3r_games['timestamp_dt'] = pd.to_datetime(df_v7p3r_games['timestamp'])
        except Exception: 
             df_v7p3r_games['timestamp_dt'] = pd.to_datetime(df_v7p3r_games['timestamp'], format="%Y%m%d_%H%M%S", errors='coerce')
        
        df_v7p3r_games = df_v7p3r_games.sort_values('timestamp_dt').reset_index(drop=True)
        
        df_v7p3r_games['cum_v7p3r_wins'] = 0
        df_v7p3r_games['cum_v7p3r_losses'] = 0
        df_v7p3r_games['cum_v7p3r_draws'] = 0

        current_wins = 0
        current_losses = 0
        current_draws = 0
        
        for index, row in df_v7p3r_games.iterrows():
            winner = row.get('winner')
            white_is_v7p3r_and_included = row.get('white_engine') == 'v7p3r' and not row.get('exclude_white_from_metrics', False)
            black_is_v7p3r_and_included = row.get('black_engine') == 'v7p3r' and not row.get('exclude_black_from_metrics', False)

            if winner == '1-0':
                if white_is_v7p3r_and_included: current_wins += 1
                if black_is_v7p3r_and_included: current_losses += 1
            elif winner == '0-1':
                if black_is_v7p3r_and_included: current_wins += 1
                if white_is_v7p3r_and_included: current_losses += 1
            elif winner == '1/2-1/2':
                if white_is_v7p3r_and_included or black_is_v7p3r_and_included:
                    current_draws += 1
            df_v7p3r_games.at[index, 'cum_v7p3r_wins'] = current_wins
            df_v7p3r_games.at[index, 'cum_v7p3r_losses'] = current_losses
            df_v7p3r_games.at[index, 'cum_v7p3r_draws'] = current_draws

        fig_static_trend.add_trace(go.Scatter(
            x=df_v7p3r_games['timestamp_dt'],
            y=df_v7p3r_games['cum_v7p3r_wins'],
            mode='lines+markers',
            name='Cumulative Wins',
            line=dict(color=DARK_SUCCESS)
        ))
        fig_static_trend.add_trace(go.Scatter(
            x=df_v7p3r_games['timestamp_dt'],
            y=df_v7p3r_games['cum_v7p3r_losses'],
            mode='lines+markers',
            name='Cumulative Losses',
            line=dict(color=DARK_ERROR)
        ))
        fig_static_trend.add_trace(go.Scatter(
            x=df_v7p3r_games['timestamp_dt'],
            y=df_v7p3r_games['cum_v7p3r_draws'],
            mode='lines+markers',
            name='Cumulative Draws',
            line=dict(color=DARK_WARNING)
        ))
        fig_static_trend.update_layout(
            title="V7P3R Game Results Over Time",
            xaxis_title="Game Timestamp",
            yaxis_title="Cumulative Count",
            height=300,
            paper_bgcolor=DARK_PANEL,
            plot_bgcolor=DARK_PANEL,
            font=dict(color=DARK_TEXT),
            xaxis=dict(gridcolor=DARK_ACCENT),
            yaxis=dict(gridcolor=DARK_ACCENT)
        )
    else: # Should be covered by earlier checks, but as a fallback
        fig_static_trend.update_layout(
            title="No V7P3R Game Result Data Available",
            height=300, margin=dict(t=40, b=20, l=30, r=20),
            paper_bgcolor=DARK_PANEL, plot_bgcolor=DARK_PANEL, font=dict(color=DARK_TEXT),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        fig_static_trend.add_annotation(text="No V7P3R game data to display.", xref="paper", yref="paper", showarrow=False, font=dict(size=16, color=DARK_TEXT))
            
    return metrics_display, fig_static_trend

# Clean up when the app is closed
def cleanup():
    print("Dashboard shutting down. Closing MetricsStore connection.")
    metrics_store.close()

atexit.register(cleanup)

if __name__ == "__main__":
    import socket
    # Get local IP address for LAN access
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "localhost"
    port = 8050
    print(f"\\nDash is running on: http://{local_ip}:{port} (accessible from your LAN)\\n")
    print(f"Or on this machine: http://localhost:{port}\\n")
    # Use host="0.0.0.0" to allow access from other machines on your LAN
    app.run(debug=True, host="0.0.0.0", port=port)