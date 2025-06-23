# chess_metrics.py
# Simplified V7P3R Chess Engine Metrics Dashboard

import dash
from dash import dcc, html, Output, Input
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from metrics_store import MetricsStore
import threading
import yaml
from datetime import datetime

# Initialize the metrics store globally
metrics_store = MetricsStore()

# Dark mode colors
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

# Initialize cloud-based metrics collection (simplified)
def start_centralized_metrics_collection():
    """Start metrics collection from centralized cloud storage."""
    try:
        from engine_utilities.cloud_store import CloudStore
        cloud_store = CloudStore()
        print("Cloud storage initialized for metrics collection")
    except Exception as e:
        print(f"Cloud metrics collection failed, using local data: {e}")

# Start collection in background
collection_thread = threading.Thread(target=start_centralized_metrics_collection, daemon=True)
collection_thread.start()

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("V7P3R Chess Engine Metrics Dashboard", 
            style={"textAlign": "center", "marginBottom": "20px", "color": DARK_TEXT}),
    
    dcc.Interval(id="interval", interval=8000, n_intervals=0),
    
    # Metric selector
    html.Div([
        html.Label("Select Metric to Analyze:", style={"color": DARK_TEXT, "marginBottom": "10px"}),
        dcc.Dropdown(
            id="metric-selector",
            options=[
                {"label": "Evaluation Score", "value": "evaluation"},
                {"label": "Search Depth", "value": "depth"},
                {"label": "Nodes Searched", "value": "nodes_searched"},
                {"label": "Time Taken (s)", "value": "time_taken"},
            ],
            value="evaluation",
            style={"backgroundColor": DARK_PANEL, "color": DARK_TEXT, "borderColor": DARK_BORDER}
        )
    ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": DARK_ACCENT, "borderRadius": "8px"}),
    
    # Main metrics graph
    dcc.Graph(id="metrics-graph", style={"height": "500px", "backgroundColor": DARK_PANEL}),
    
    # Summary statistics
    html.Div(id="summary-stats", style={"padding": "15px", "backgroundColor": DARK_ACCENT, 
                                       "borderRadius": "8px", "marginTop": "20px", "color": DARK_TEXT}),
    
], style={"fontFamily": "Arial, sans-serif", "padding": "20px", "backgroundColor": DARK_BG, "minHeight": "100vh"})

@app.callback(
    [Output("metrics-graph", "figure"),
     Output("summary-stats", "children")],
    [Input("interval", "n_intervals"),
     Input("metric-selector", "value")]
)
def update_metrics(_, selected_metric):
    """Update the V7P3R engine metrics visualization."""
    fig = go.Figure()
    summary_components = []

    if not selected_metric:
        selected_metric = "evaluation"

    try:
        # Get V7P3R move metrics from database
        connection = metrics_store._get_connection()
        with connection:
            cursor = connection.cursor()
            
            # V7P3R engine types
            v7p3r_engine_types = ['deepsearch', 'lookahead', 'minimax', 'negamax', 'negascout', 
                                  'transposition_only', 'simple_search', 'quiescence_only', 
                                  'simple_eval', 'v7p3r']
            
            # Get V7P3R moves where exclude_from_metrics is False or NULL
            placeholders = ','.join(['?' for _ in v7p3r_engine_types])
            query = f"""
                SELECT game_id, move_number, player_color, move_uci, evaluation, 
                       engine_type, depth, nodes_searched, time_taken, pv_line, created_at
                FROM move_metrics 
                WHERE engine_type IN ({placeholders})
                AND (exclude_from_metrics = 0 OR exclude_from_metrics IS NULL)
                AND {selected_metric} IS NOT NULL
                ORDER BY created_at
            """
            cursor.execute(query, v7p3r_engine_types)
            moves_data = cursor.fetchall()
            
            if not moves_data:
                fig.update_layout(
                    title=f"No V7P3R Data Found for {selected_metric.replace('_', ' ').title()}",
                    paper_bgcolor=DARK_PANEL, plot_bgcolor=DARK_PANEL, font=dict(color=DARK_TEXT),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                fig.add_annotation(text="No V7P3R move data found. Try running some games first.", 
                                 xref="paper", yref="paper", showarrow=False, 
                                 font=dict(size=16, color=DARK_TEXT))
                summary_components.append(html.H3("No Data Available", style={"color": DARK_ERROR}))
                summary_components.append(html.P("Run some chess games with V7P3R engine to see metrics.", 
                                                style={"color": DARK_TEXT}))
                return fig, summary_components
            
            # Convert to DataFrame
            columns = ['game_id', 'move_number', 'player_color', 'move_uci', 'evaluation', 
                      'engine_type', 'depth', 'nodes_searched', 'time_taken', 'pv_line', 'created_at']
            df = pd.DataFrame(moves_data, columns=columns)
            
            # Convert created_at to datetime for plotting
            df['created_at_dt'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('created_at_dt')
            
            # Create main scatter plot
            fig.add_trace(go.Scatter(
                x=df['created_at_dt'],
                y=df[selected_metric],
                mode='markers',
                name=f'V7P3R {selected_metric.replace("_", " ").title()}',
                marker=dict(size=8, opacity=0.7, color=DARK_HIGHLIGHT),
                text=df['engine_type'],
                customdata=df[['game_id', 'move_number', 'player_color']],
                hovertemplate=(
                    f'<b>{selected_metric.title()}</b>: %{{y}}<br>'
                    'Engine: %{text}<br>'
                    'Game: %{customdata[0]}<br>'
                    'Move: %{customdata[1]} (%{customdata[2]})<br>'
                    'Time: %{x}<br>'
                    '<extra></extra>'
                )
            ))
            
            # Add trendline if we have enough data
            if len(df) > 1:
                numeric_values = pd.to_numeric(df[selected_metric], errors='coerce').dropna()
                if len(numeric_values) > 1:
                    df['time_numeric'] = (df['created_at_dt'] - df['created_at_dt'].min()).dt.total_seconds()
                    valid_indices = numeric_values.index
                    x_vals = np.array(df.loc[valid_indices, 'time_numeric'])
                    y_vals = np.array(numeric_values)
                    
                    if len(x_vals) > 1:
                        z = np.polyfit(x_vals, y_vals, 1)
                        p = np.poly1d(z)
                        
                        fig.add_trace(go.Scatter(
                            x=df.loc[valid_indices, 'created_at_dt'],
                            y=p(x_vals),
                            mode='lines',
                            name='Trend',
                            line=dict(color=DARK_SUCCESS, dash='dash', width=2)
                        ))
            
            # Update layout
            fig.update_layout(
                title=f"V7P3R Engine: {selected_metric.replace('_', ' ').title()} Over Time",
                xaxis_title="Time",
                yaxis_title=selected_metric.replace('_', ' ').title(),
                hovermode="closest",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor=DARK_PANEL,
                plot_bgcolor=DARK_PANEL,
                font=dict(color=DARK_TEXT),
                xaxis=dict(gridcolor=DARK_ACCENT),
                yaxis=dict(gridcolor=DARK_ACCENT)
            )
            
            # Generate summary statistics
            summary_components.append(html.H3(f"V7P3R {selected_metric.replace('_', ' ').title()} Summary", 
                                            style={"color": DARK_TEXT, "marginBottom": "15px"}))
            
            metric_values = pd.to_numeric(df[selected_metric], errors='coerce').dropna()
            if not metric_values.empty:
                avg_val = metric_values.mean()
                min_val = metric_values.min()
                max_val = metric_values.max()
                std_val = metric_values.std()
                
                summary_components.extend([
                    html.Div([
                        html.Div([
                            html.H4("Statistics", style={"color": DARK_HIGHLIGHT, "marginBottom": "10px"}),
                            html.P(f"Total moves: {len(df)}", style={"color": DARK_TEXT}),
                            html.P(f"Average: {avg_val:.3f}", style={"color": DARK_TEXT}),
                            html.P(f"Range: {min_val:.3f} to {max_val:.3f}", style={"color": DARK_TEXT}),
                            html.P(f"Std deviation: {std_val:.3f}", style={"color": DARK_TEXT}),
                        ], style={"flex": "1", "marginRight": "20px"}),
                        
                        html.Div([
                            html.H4("Engine Breakdown", style={"color": DARK_HIGHLIGHT, "marginBottom": "10px"}),
                            *[html.P(f"{engine}: {count} moves", style={"color": DARK_SUBTEXT}) 
                              for engine, count in df['engine_type'].value_counts().items()]
                        ], style={"flex": "1"})
                    ], style={"display": "flex", "flexDirection": "row"})
                ])
                
                # Color breakdown
                color_counts = df['player_color'].value_counts()
                summary_components.append(html.Div([
                    html.H4("Move Color Distribution", style={"color": DARK_HIGHLIGHT, "marginTop": "15px"}),
                    *[html.P(f"{'White' if color == 'w' else 'Black'}: {count} moves", 
                            style={"color": DARK_SUBTEXT}) 
                      for color, count in color_counts.items()]
                ]))
            
    except Exception as e:
        fig.update_layout(
            title="Error Loading Data",
            paper_bgcolor=DARK_PANEL, plot_bgcolor=DARK_PANEL, font=dict(color=DARK_TEXT)
        )
        fig.add_annotation(text=f"Database error: {str(e)}", xref="paper", yref="paper", 
                          showarrow=False, font=dict(size=16, color=DARK_ERROR))
        summary_components.append(html.H3("Database Error", style={"color": DARK_ERROR}))
        summary_components.append(html.P(f"Error: {str(e)}", style={"color": DARK_TEXT}))

    return fig, summary_components

if __name__ == "__main__":
    print("Starting V7P3R Chess Engine Metrics Dashboard...")
    print("Open http://127.0.0.1:8050/ in your browser")
    app.run(debug=True, host='127.0.0.1', port=8050)
