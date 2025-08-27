import plotly.graph_objects as go
import json

# Load the component data
data = {"components": [{"name": "React + Tailwind Frontend", "layer": "presentation", "x": 0, "y": 0, "width": 8, "height": 2, "color": "#3B82F6"}, {"name": "FastAPI Gateway", "layer": "api", "x": 0, "y": 3, "width": 4, "height": 1.5, "color": "#10B981"}, {"name": "WebSocket", "layer": "api", "x": 4.5, "y": 3, "width": 3.5, "height": 1.5, "color": "#10B981"}, {"name": "JWT Auth", "layer": "security", "x": 0, "y": 5, "width": 2, "height": 1, "color": "#F59E0B"}, {"name": "Task Manager", "layer": "core", "x": 2.5, "y": 5, "width": 2, "height": 1, "color": "#8B5CF6"}, {"name": "RL Scheduler", "layer": "core", "x": 5, "y": 5, "width": 3, "height": 1, "color": "#8B5CF6"}, {"name": "CPU Engine", "layer": "compute", "x": 0, "y": 7, "width": 2.5, "height": 1.5, "color": "#EF4444"}, {"name": "GPU Engine", "layer": "compute", "x": 3, "y": 7, "width": 2.5, "height": 1.5, "color": "#EF4444"}, {"name": "FPGA/Sim", "layer": "compute", "x": 6, "y": 7, "width": 2, "height": 1.5, "color": "#EF4444"}, {"name": "PostgreSQL", "layer": "storage", "x": 0, "y": 9.5, "width": 4, "height": 1, "color": "#6B7280"}, {"name": "Metrics Store", "layer": "storage", "x": 4.5, "y": 9.5, "width": 3.5, "height": 1, "color": "#6B7280"}]}

# Create figure
fig = go.Figure()

# Add rectangles and text for each component
shapes = []
annotations = []

for comp in data['components']:
    # Add rectangle shape
    shapes.append(
        dict(
            type="rect",
            x0=comp['x'], 
            y0=-comp['y'],  # Invert y to have presentation layer at top
            x1=comp['x'] + comp['width'], 
            y1=-comp['y'] - comp['height'],
            fillcolor=comp['color'],
            line=dict(color="white", width=2),
            opacity=0.8
        )
    )
    
    # Add text annotation
    # Truncate component name to fit 15 char limit
    display_name = comp['name']
    if len(display_name) > 15:
        if "React + Tailwind" in display_name:
            display_name = "React Frontend"
        elif "FastAPI Gateway" in display_name:
            display_name = "FastAPI API"
        elif "RL Scheduler" in display_name:
            display_name = "RL Scheduler"
        elif "Task Manager" in display_name:
            display_name = "Task Manager"
        elif "CPU Engine" in display_name:
            display_name = "CPU Engine"
        elif "GPU Engine" in display_name:
            display_name = "GPU Engine"
        elif "FPGA/Sim" in display_name:
            display_name = "FPGA/Sim"
        elif "Metrics Store" in display_name:
            display_name = "Metrics Store"
    
    annotations.append(
        dict(
            x=comp['x'] + comp['width']/2,
            y=-comp['y'] - comp['height']/2,
            text=display_name,
            showarrow=False,
            font=dict(color="white", size=12, family="Arial"),
            xanchor="center",
            yanchor="middle"
        )
    )

# Add arrows to show data flow
arrows = [
    # Frontend to API Gateway
    dict(
        type="line",
        x0=4, y0=-2,
        x1=2, y1=-3,
        line=dict(color="#1FB8CD", width=2)
    ),
    # API Gateway to Core Services
    dict(
        type="line", 
        x0=2, y0=-4.5,
        x1=3.5, y1=-5,
        line=dict(color="#1FB8CD", width=2)
    ),
    # Core Services to Compute Engines
    dict(
        type="line",
        x0=3.5, y0=-6,
        x1=4, y1=-7,
        line=dict(color="#1FB8CD", width=2)
    ),
    # Core Services to Storage
    dict(
        type="line",
        x0=3.5, y0=-6,
        x1=2, y1=-9.5,
        line=dict(color="#1FB8CD", width=2)
    )
]

# Add arrow heads
for arrow in arrows:
    shapes.append(arrow)

# Update layout
fig.update_layout(
    title="SmartCompute Optimizer Platform",
    shapes=shapes,
    annotations=annotations,
    xaxis=dict(
        range=[-0.5, 8.5],
        showgrid=False,
        showticklabels=False,
        zeroline=False
    ),
    yaxis=dict(
        range=[-11, 1],
        showgrid=False, 
        showticklabels=False,
        zeroline=False
    ),
    plot_bgcolor="white",
    showlegend=False
)

# Add layer labels on the left
layer_labels = [
    {"text": "Frontend", "y": -1, "color": "#3B82F6"},
    {"text": "API Layer", "y": -3.75, "color": "#10B981"}, 
    {"text": "Security/Core", "y": -5.5, "color": "#8B5CF6"},
    {"text": "Compute", "y": -7.75, "color": "#EF4444"},
    {"text": "Storage", "y": -10, "color": "#6B7280"}
]

for label in layer_labels:
    fig.add_annotation(
        x=-0.8,
        y=label['y'],
        text=label['text'],
        showarrow=False,
        font=dict(color=label['color'], size=10, family="Arial"),
        xanchor="center",
        yanchor="middle",
        textangle=90
    )

# Save the chart
fig.write_image("architecture_diagram.png")