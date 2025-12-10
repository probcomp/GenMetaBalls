#!/usr/bin/env python3
"""Load benchmark results and recreate the performance plot without log scale."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

# Load the benchmark results
PROJECT_ROOT = Path(__file__).resolve().parent.parent
results_file = PROJECT_ROOT / "scripts" / "data" / "benchmark_cache" / "full_benchmark_results.json"

print(f"Loading benchmark results from {results_file}...")
with open(results_file, "r") as f:
    results = json.load(f)

print("Creating performance plot (linear scale)...")

# Create the plot
plt.figure(figsize=(12, 8))

# Extract data for plotting
fmb_sizes = results["config"]["fmb_sizes"]

# Define colors and line styles for each method
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
line_styles = ['-', '--', '-.', ':', '-', '--']

for i, (method_name, method_data) in enumerate(results["data"].items()):
    # Extract times for this method
    times = [point["avg_time_us"] for point in method_data]
    
    # Replace kernel names and chunk labels
    display_label = method_name
    display_label = display_label.replace("CUDA Kernel 0", "CUDA (First Version)")
    display_label = display_label.replace("CUDA Kernel 1", "CUDA (Second Version)")
    display_label = display_label.replace("chunk=", "metaball chunk size=")
    
    plt.plot(
        fmb_sizes, 
        times, 
        color=colors[i % len(colors)],
        linestyle=line_styles[i % len(line_styles)],
        marker='o',
        linewidth=4,
        markersize=6,
        label=display_label
    )

plt.xlabel('Number of Metaballs', fontsize=18)
plt.ylabel('Runtime (microseconds)', fontsize=18)
plt.title('Performance Comparison: FMB Rendering Methods', fontsize=20, fontweight='bold')
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
# Make tick labels bigger
plt.tick_params(axis='both', which='major', labelsize=16)
# NOTE: No log scale - using linear scale

# Add configuration info as text
config_text = f"Image: {results['config']['width']}x{results['config']['height']}, " \
              f"Grid: {results['config']['grid_size']}, " \
              f"Block: {results['config']['block_size']}"
plt.figtext(0.02, 0.02, config_text, fontsize=12, style='italic')

plt.tight_layout()

# Save plot with different name
plot_file = PROJECT_ROOT / "scripts" / "data" / "benchmark_cache" / "performance_comparison_linear.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Performance plot saved to {plot_file}")
