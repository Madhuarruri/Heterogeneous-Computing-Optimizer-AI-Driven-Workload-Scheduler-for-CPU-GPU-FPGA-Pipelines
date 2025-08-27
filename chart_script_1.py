import plotly.graph_objects as go
import plotly.express as px

# Data
data = {
    "workloads": ["Matrix 1024x1024", "Matrix 4096x4096", "2D FFT 2048x2048", "Image Conv 4K", "ResNet50 Inference", "Video Transcode"],
    "cpu_speedup": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "gpu_speedup": [8.2, 12.5, 6.8, 15.3, 25.4, 8.9],
    "fpga_speedup": [3.1, 4.8, 12.2, 22.1, 18.7, 4.2]
}

# Abbreviate workload names to fit 15 character limit
workloads_short = ["Matrix 1024", "Matrix 4096", "2D FFT 2048", "Image Conv 4K", "ResNet50 Inf", "Video Trans"]

# Brand colors
colors = ['#1FB8CD', '#DB4545', '#2E8B57']

fig = go.Figure()

# Add CPU bars
fig.add_trace(go.Bar(
    name='CPU',
    x=workloads_short,
    y=data["cpu_speedup"],
    marker_color=colors[0],
    text=[f'{x:.1f}x' for x in data["cpu_speedup"]],
    textposition='outside'
))

# Add GPU bars
fig.add_trace(go.Bar(
    name='GPU',
    x=workloads_short,
    y=data["gpu_speedup"],
    marker_color=colors[1],
    text=[f'{x:.1f}x' for x in data["gpu_speedup"]],
    textposition='outside'
))

# Add FPGA bars
fig.add_trace(go.Bar(
    name='FPGA',
    x=workloads_short,
    y=data["fpga_speedup"],
    marker_color=colors[2],
    text=[f'{x:.1f}x' for x in data["fpga_speedup"]],
    textposition='outside'
))

# Update layout
fig.update_layout(
    title='Compute Device Performance Comparison',
    xaxis_title='Workload Type',
    yaxis_title='Speedup vs CPU',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update traces with cliponaxis=False
fig.update_traces(cliponaxis=False)

# Update y-axis to start from 0
fig.update_yaxes(range=[0, max(max(data["gpu_speedup"]), max(data["fpga_speedup"])) * 1.1])

# Save the chart
fig.write_image("performance_benchmark.png")