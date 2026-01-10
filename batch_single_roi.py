#!/usr/bin/env python3
"""
Batch ROI Analysis - Analyzes ALL h5 files in folder with same ROI

TO CHANGE FPS: Edit line 555
FPS = 25  # Your video frame rate

The script will:
1. Find all .h5 files in the folder
2. Draw/select ROI for first file
3. Automatically use same ROI for all other files
4. Save results with full filenames
"""

from deeplabcut_roi_analysis import DeepLabCutROIAnalyzer
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import glob


def draw_single_roi(analyzer):
    """
    Draw a single ROI for focused analysis
    """
    print("\n" + "="*60)
    print("SINGLE ROI DRAWING")
    print("="*60)
    print("Draw ONE region of interest for ALL files")
    print("\nControls:")
    print("  • Click = Add vertex (minimum 3)")
    print("  • C = Close polygon and analyze")
    print("  • U = Undo last vertex")
    print("  • R = Restart")
    print("="*60)
    
    # Get trajectory data
    bodypart = analyzer.bodyparts[0]
    x_data = analyzer.df[analyzer.scorer][bodypart]['x'].values
    y_data = analyzer.df[analyzer.scorer][bodypart]['y'].values
    
    valid = ~(np.isnan(x_data) | np.isnan(y_data))
    x_data = x_data[valid]
    y_data = y_data[valid]
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot full trajectory
    ax.plot(x_data[::5], y_data[::5], 'gray', alpha=0.3, linewidth=0.5, label='Full trajectory')
    ax.set_xlim(analyzer.arena_bounds['x_min'], analyzer.arena_bounds['x_max'])
    ax.set_ylim(analyzer.arena_bounds['y_min'], analyzer.arena_bounds['y_max'])
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('Click to draw your region of interest')
    
    # State variables
    vertices = []
    plots = []
    roi_defined = False
    roi_name = ""
    
    def onclick(event):
        nonlocal vertices, plots
        if event.inaxes != ax or roi_defined:
            return
        
        # Add vertex
        vertex = (event.xdata, event.ydata)
        vertices.append(vertex)
        
        # Draw vertex
        dot = ax.plot(event.xdata, event.ydata, 'ro', markersize=10)[0]
        plots.append(dot)
        
        # Draw connecting line
        if len(vertices) > 1:
            prev = vertices[-2]
            line = ax.plot([prev[0], vertex[0]], [prev[1], vertex[1]], 'r-', linewidth=2)[0]
            plots.append(line)
        
        # Update title
        n = len(vertices)
        if n < 3:
            ax.set_title(f'{n} vertices - need at least 3 to close')
        else:
            ax.set_title(f'{n} vertices - Press C to close and analyze')
        plt.draw()
    
    def onkey(event):
        nonlocal vertices, plots, roi_defined, roi_name
        
        if event.key.lower() == 'c' and len(vertices) >= 3 and not roi_defined:
            # Close polygon
            closing_line = ax.plot([vertices[-1][0], vertices[0][0]], 
                                  [vertices[-1][1], vertices[0][1]], 
                                  'r-', linewidth=2)[0]
            plots.append(closing_line)
            
            # Get ROI name
            roi_name = input("\nEnter name for this ROI (e.g., 'target_zone'): ").strip()
            if not roi_name:
                roi_name = "target_roi"
            
            # Create filled polygon
            polygon = Polygon(np.array(vertices), closed=True, linewidth=3,
                            edgecolor='red', facecolor='red', alpha=0.2)
            ax.add_patch(polygon)
            
            # Add label
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            center_x = np.mean(xs)
            center_y = np.mean(ys)
            ax.text(center_x, center_y, roi_name,
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color='darkred', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            # Store ROI
            analyzer.rois[roi_name] = analyzer.position(
                topleft=(min(xs), min(ys)),
                bottomright=(max(xs), max(ys))
            )
            
            roi_defined = True
            ax.set_title('ROI defined - Close window to continue with analysis')
            ax.legend()
            plt.draw()
            
        elif event.key.lower() == 'u' and not roi_defined:
            if vertices:
                vertices.pop()
                # Remove last plots
                for _ in range(min(2, len(plots))):
                    if plots:
                        plots[-1].remove()
                        plots.pop()
                ax.set_title('Click to draw your region of interest')
                plt.draw()
                
        elif event.key.lower() == 'r' and not roi_defined:
            vertices.clear()
            for p in plots:
                p.remove()
            plots.clear()
            ax.set_title('Click to draw your region of interest')
            plt.draw()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    
    plt.show()
    
    return roi_name if roi_defined else None


def calculate_preference_metrics(analyzer, roi_name, fps=25):
    """
    Calculate comprehensive behavioral preference metrics for single ROI
    """
    print("\n" + "="*60)
    print("CALCULATING PREFERENCE METRICS")
    print("="*60)
    
    # Tracking data on Nose
    bodypart = "Nose"
    x = analyzer.df[analyzer.scorer][bodypart]['x'].values
    y = analyzer.df[analyzer.scorer][bodypart]['y'].values
    
    # Calculate velocity
    velocity = np.zeros(len(x))
    velocity[1:] = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    
    # Determine if each frame is in ROI
    from time_in_each_roi import get_roi_at_each_frame
    tracking_data = np.column_stack((x, y))
    roi_at_frame = get_roi_at_each_frame(tracking_data, analyzer.rois, check_inroi=False)
    
    # Binary array: 1 if in target ROI, 0 if not
    in_roi = np.array([1 if r == roi_name else 0 for r in roi_at_frame])
    
    # Total experiment duration
    total_frames = len(x)
    total_time_s = total_frames / fps
    
    # 1. TIME-BASED METRICS
    frames_in_roi = np.sum(in_roi)
    frames_outside = total_frames - frames_in_roi
    time_in_roi_s = frames_in_roi / fps
    time_outside_s = frames_outside / fps
    percent_time_in_roi = (frames_in_roi / total_frames) * 100
    
    # 2. ENTRY METRICS
    # Count transitions into ROI
    entries = np.sum((in_roi[1:] == 1) & (in_roi[:-1] == 0))
    entry_frequency = entries / (total_time_s / 60)  # per minute
    
    # 3. BOUT ANALYSIS
    # Find all bouts (continuous periods in ROI)
    bouts = []
    in_bout = False
    bout_start = 0
    
    for i in range(len(in_roi)):
        if in_roi[i] == 1 and not in_bout:
            bout_start = i
            in_bout = True
        elif in_roi[i] == 0 and in_bout:
            bout_duration = (i - bout_start) / fps
            bouts.append(bout_duration)
            in_bout = False
    
    # Handle case where last frame is in ROI
    if in_bout:
        bout_duration = (len(in_roi) - bout_start) / fps
        bouts.append(bout_duration)
    
    if bouts:
        mean_bout_duration = np.mean(bouts)
        median_bout_duration = np.median(bouts)
        max_bout_duration = np.max(bouts)
        bout_variability = np.std(bouts)
    else:
        mean_bout_duration = median_bout_duration = max_bout_duration = bout_variability = 0
    
    # 4. LATENCY METRICS
    # First entry latency
    first_entry_frame = np.argmax(in_roi) if np.any(in_roi) else total_frames
    first_entry_latency_s = first_entry_frame / fps
    
    # 5. VELOCITY METRICS
    velocity_in_roi = velocity[in_roi == 1]
    velocity_outside = velocity[in_roi == 0]
    
    if len(velocity_in_roi) > 0:
        mean_velocity_in_roi = np.mean(velocity_in_roi)
        activity_in_roi = np.sum(velocity_in_roi > 2) / len(velocity_in_roi) * 100  # % active
    else:
        mean_velocity_in_roi = 0
        activity_in_roi = 0
    
    if len(velocity_outside) > 0:
        mean_velocity_outside = np.mean(velocity_outside)
        activity_outside = np.sum(velocity_outside > 2) / len(velocity_outside) * 100
    else:
        mean_velocity_outside = 0
        activity_outside = 0
    
    # 6. PREFERENCE INDICES
    
    # A. Classic Preference Index (Schoenfeld et al., 1980)
    # PI = (Time in ROI - Time outside) / Total time × 100
    preference_index_classic = ((time_in_roi_s - time_outside_s) / total_time_s) * 100
    
    # B. Discrimination Index (Ennaceur & Delacour, 1988)
    # DI = (Time in ROI - Time outside) / (Time in ROI + Time outside)
    discrimination_index = (time_in_roi_s - time_outside_s) / (time_in_roi_s + time_outside_s)
    
    # C. Exploration Ratio (Dix & Aggleton, 1999)
    # ER = Time in ROI / (Time in ROI + Time outside)
    exploration_ratio = time_in_roi_s / (time_in_roi_s + time_outside_s)
    
    # D. Entry Preference Score
    # EPS = (Entries to ROI / Total locomotor activity) × 100
    total_distance = np.sum(velocity) / fps  # total distance in pixels
    if total_distance > 0:
        entry_preference_score = (entries / total_distance) * 1000  # entries per 1000 pixels
    else:
        entry_preference_score = 0
    
    # 7. STATISTICAL TESTS
    
    # A. Binomial test - is time in ROI different from chance?
    # Get ROI area proportion
    roi_coords = analyzer.rois[roi_name]
    roi_width = roi_coords.bottomright[0] - roi_coords.topleft[0]
    roi_height = roi_coords.bottomright[1] - roi_coords.topleft[1]
    roi_area = roi_width * roi_height
    
    arena_width = analyzer.arena_bounds['x_max'] - analyzer.arena_bounds['x_min']
    arena_height = analyzer.arena_bounds['y_max'] - analyzer.arena_bounds['y_min']
    arena_area = arena_width * arena_height
    
    expected_proportion = roi_area / arena_area
    
    # Binomial test
    from scipy.stats import binomtest
    binomial_result = binomtest(frames_in_roi, total_frames, expected_proportion, alternative='two-sided')
    binomial_p = binomial_result.pvalue
    
    # B. One-sample t-test for bout durations against expected
    if len(bouts) > 1:
        expected_bout_duration = (expected_proportion * total_time_s) / (entries if entries > 0 else 1)
        t_stat, t_pvalue = stats.ttest_1samp(bouts, expected_bout_duration)
    else:
        t_stat, t_pvalue = 0, 1
    
    # C. Mann-Whitney U test for velocity differences
    if len(velocity_in_roi) > 0 and len(velocity_outside) > 0:
        u_stat, u_pvalue = stats.mannwhitneyu(velocity_in_roi, velocity_outside, alternative='two-sided')
    else:
        u_stat, u_pvalue = 0, 1
    
    # 8. EFFECT SIZE CALCULATIONS
    
    # Cohen's d for time preference
    if expected_proportion > 0:
        observed_proportion = frames_in_roi / total_frames
        cohens_d = (observed_proportion - expected_proportion) / np.sqrt(expected_proportion * (1 - expected_proportion))
    else:
        cohens_d = 0
    
    # Create results dictionary
    metrics = {
        # Basic metrics
        'roi_name': roi_name,
        'total_time_s': total_time_s,
        'roi_area_proportion': expected_proportion,
        
        # Time metrics
        'time_in_roi_s': time_in_roi_s,
        'time_outside_s': time_outside_s,
        'percent_time_in_roi': percent_time_in_roi,
        'expected_percent_time': expected_proportion * 100,
        
        # Entry metrics
        'number_of_entries': entries,
        'entry_frequency_per_min': entry_frequency,
        'first_entry_latency_s': first_entry_latency_s,
        
        # Bout metrics
        'number_of_bouts': len(bouts),
        'mean_bout_duration_s': mean_bout_duration,
        'median_bout_duration_s': median_bout_duration,
        'max_bout_duration_s': max_bout_duration,
        'bout_duration_variability': bout_variability,
        
        # Activity metrics
        'mean_velocity_in_roi': mean_velocity_in_roi,
        'mean_velocity_outside': mean_velocity_outside,
        'percent_active_in_roi': activity_in_roi,
        'percent_active_outside': activity_outside,
        
        # Preference indices
        'preference_index_classic': preference_index_classic,
        'discrimination_index': discrimination_index,
        'exploration_ratio': exploration_ratio,
        'entry_preference_score': entry_preference_score,
        
        # Statistical tests
        'binomial_test_p': binomial_p,
        'bout_duration_t_test_p': t_pvalue,
        'velocity_mann_whitney_p': u_pvalue,
        'cohens_d_effect_size': cohens_d
    }
    
    return metrics, in_roi, velocity


def create_visualizations(analyzer, metrics, in_roi, velocity, roi_name, fps=25):
    """
    Create comprehensive visualizations for single ROI analysis
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Time course plot
    ax1 = plt.subplot(3, 3, 1)
    time_s = np.arange(len(in_roi)) / fps
    ax1.fill_between(time_s, 0, in_roi, alpha=0.5, color='red', label=roi_name)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('In ROI')
    ax1.set_title('ROI Occupancy Over Time')
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Outside', 'Inside'])
    
    # 2. Cumulative time plot
    ax2 = plt.subplot(3, 3, 2)
    cumulative_time = np.cumsum(in_roi) / fps
    expected_cumulative = time_s * metrics['expected_percent_time'] / 100
    ax2.plot(time_s, cumulative_time, 'r-', linewidth=2, label='Observed')
    ax2.plot(time_s, expected_cumulative, 'k--', linewidth=1, label='Expected by chance')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Cumulative time in ROI (s)')
    ax2.set_title('Cumulative Time in ROI')
    ax2.legend()
    ax2.fill_between(time_s, cumulative_time, expected_cumulative, 
                     where=(cumulative_time > expected_cumulative), 
                     color='green', alpha=0.3, label='Above chance')
    ax2.fill_between(time_s, cumulative_time, expected_cumulative, 
                     where=(cumulative_time <= expected_cumulative), 
                     color='red', alpha=0.3, label='Below chance')
    
    # 3. Bout duration histogram
    ax3 = plt.subplot(3, 3, 3)
    bout_durations = []
    in_bout = False
    bout_start = 0
    for i in range(len(in_roi)):
        if in_roi[i] == 1 and not in_bout:
            bout_start = i
            in_bout = True
        elif in_roi[i] == 0 and in_bout:
            bout_durations.append((i - bout_start) / fps)
            in_bout = False
    if in_bout:
        bout_durations.append((len(in_roi) - bout_start) / fps)
    
    if bout_durations:
        ax3.hist(bout_durations, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(bout_durations), color='red', linestyle='--', label=f'Mean: {np.mean(bout_durations):.2f}s')
        ax3.axvline(np.median(bout_durations), color='green', linestyle='--', label=f'Median: {np.median(bout_durations):.2f}s')
    ax3.set_xlabel('Bout duration (s)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Visit Durations')
    ax3.legend()
    
    # 4. Entry frequency over time (sliding window)
    ax4 = plt.subplot(3, 3, 4)
    window_size = int(60 * fps)  # 60-second window
    entries_per_window = []
    window_times = []
    
    for i in range(0, len(in_roi) - window_size, window_size // 2):
        window = in_roi[i:i + window_size]
        entries_in_window = np.sum((window[1:] == 1) & (window[:-1] == 0))
        entries_per_window.append(entries_in_window)
        window_times.append((i + window_size / 2) / fps)
    
    if entries_per_window:
        ax4.plot(window_times, entries_per_window, 'b-', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Entries per minute')
        ax4.set_title('Entry Frequency Over Time (60s windows)')
    
    # 5. Velocity comparison
    ax5 = plt.subplot(3, 3, 5)
    velocity_in = velocity[in_roi == 1]
    velocity_out = velocity[in_roi == 0]
    
    bp_data = [velocity_in, velocity_out]
    bp_labels = ['In ROI', 'Outside ROI']
    bp = ax5.boxplot(bp_data, labels=bp_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('red')
    bp['boxes'][1].set_facecolor('gray')
    ax5.set_ylabel('Velocity (pixels/frame)')
    ax5.set_title('Movement Speed Comparison')
    
    # Add significance
    if metrics['velocity_mann_whitney_p'] < 0.05:
        ax5.text(1.5, max(velocity.max(), 1) * 0.9, 
                f"p = {metrics['velocity_mann_whitney_p']:.4f} *", 
                ha='center', fontsize=10, color='red')
    
    # 6. Preference indices bar plot
    ax6 = plt.subplot(3, 3, 6)
    indices = ['Classic PI', 'Discrimination', 'Exploration Ratio']
    values = [metrics['preference_index_classic'], 
             metrics['discrimination_index'] * 100,  # Convert to percentage
             metrics['exploration_ratio'] * 100]
    
    colors = ['red' if v > 0 else 'blue' for v in values[:2]] + ['green']
    bars = ax6.bar(indices, values, color=colors, alpha=0.7, edgecolor='black')
    ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_ylabel('Index Value (%)')
    ax6.set_title('Preference Indices')
    ax6.set_ylim(-100, 100)
    
    # Add values on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 2 * np.sign(height),
                f'{value:.1f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # 7. Statistical significance summary
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')
    
    sig_text = "STATISTICAL TESTS\n" + "="*30 + "\n\n"
    
    # Binomial test
    sig_marker = "***" if metrics['binomial_test_p'] < 0.001 else "**" if metrics['binomial_test_p'] < 0.01 else "*" if metrics['binomial_test_p'] < 0.05 else "ns"
    sig_text += f"Time in ROI vs chance:\np = {metrics['binomial_test_p']:.4f} {sig_marker}\n\n"
    
    # Bout duration test definition
    sig_marker = "***" if metrics['bout_duration_t_test_p'] < 0.001 else "**" if metrics['bout_duration_t_test_p'] < 0.01 else "*" if metrics['bout_duration_t_test_p'] < 0.05 else "ns"
    sig_text += f"Bout duration vs expected:\np = {metrics['bout_duration_t_test_p']:.4f} {sig_marker}\n\n"
    
    # Velocity test definition
    sig_marker = "***" if metrics['velocity_mann_whitney_p'] < 0.001 else "**" if metrics['velocity_mann_whitney_p'] < 0.01 else "*" if metrics['velocity_mann_whitney_p'] < 0.05 else "ns"
    sig_text += f"Velocity in vs out:\np = {metrics['velocity_mann_whitney_p']:.4f} {sig_marker}\n\n"
    
    # Effect size
    effect_interpretation = "Large" if abs(metrics['cohens_d_effect_size']) > 0.8 else "Medium" if abs(metrics['cohens_d_effect_size']) > 0.5 else "Small" if abs(metrics['cohens_d_effect_size']) > 0.2 else "Negligible"
    sig_text += f"Effect size (Cohen's d):\n{metrics['cohens_d_effect_size']:.3f} ({effect_interpretation})"
    
    ax7.text(0.1, 0.9, sig_text, transform=ax7.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')
    
    # 8. Key metrics summary
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    summary_text = "KEY METRICS\n" + "="*30 + "\n\n"
    summary_text += f"Time in ROI: {metrics['percent_time_in_roi']:.1f}%\n"
    summary_text += f"Expected: {metrics['expected_percent_time']:.1f}%\n"
    summary_text += f"Difference: {metrics['percent_time_in_roi'] - metrics['expected_percent_time']:.1f}%\n\n"
    summary_text += f"Entries: {metrics['number_of_entries']:.0f}\n"
    summary_text += f"Entry rate: {metrics['entry_frequency_per_min']:.2f}/min\n"
    summary_text += f"First entry: {metrics['first_entry_latency_s']:.1f}s\n\n"
    summary_text += f"Mean visit: {metrics['mean_bout_duration_s']:.2f}s\n"
    summary_text += f"Max visit: {metrics['max_bout_duration_s']:.2f}s"
    
    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')
    
    # 9. Trajectory with ROI that was highlighted I have used "Nose" 
    ax9 = plt.subplot(3, 3, 9)
    x = analyzer.df[analyzer.scorer]["Nose"]['x'].values
    y = analyzer.df[analyzer.scorer]["Nose"]['y'].values
    
    # Plot trajectory colored by ROI occupancy
    for i in range(1, len(x)):
        if not np.isnan(x[i-1]) and not np.isnan(x[i]):
            color = 'red' if in_roi[i] == 1 else 'gray'
            alpha = 0.5 if in_roi[i] == 1 else 0.1
            ax9.plot([x[i-1], x[i]], [y[i-1], y[i]], color=color, alpha=alpha, linewidth=0.5)
    
    # Draw ROI 
    roi_coords = analyzer.rois[roi_name]
    roi_vertices = [
        (roi_coords.topleft[0], roi_coords.topleft[1]),
        (roi_coords.bottomright[0], roi_coords.topleft[1]),
        (roi_coords.bottomright[0], roi_coords.bottomright[1]),
        (roi_coords.topleft[0], roi_coords.bottomright[1])
    ]
    roi_poly = Polygon(np.array(roi_vertices), closed=True, linewidth=2,
                      edgecolor='red', facecolor='red', alpha=0.2)
    ax9.add_patch(roi_poly)
    
    ax9.set_xlim(analyzer.arena_bounds['x_min'], analyzer.arena_bounds['x_max'])
    ax9.set_ylim(analyzer.arena_bounds['y_min'], analyzer.arena_bounds['y_max'])
    ax9.set_xlabel('X (pixels)')
    ax9.set_ylabel('Y (pixels)')
    ax9.set_title('Trajectory (red = in ROI)')
    ax9.set_aspect('equal')
    ax9.invert_yaxis()
    
    plt.suptitle(f'Single ROI Preference Analysis: {roi_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def analyze_single_roi():
    # CHANGE ONLY THIS LINE FOR FPS
    FPS = 25  # Video frame rate (can be checked in ffmpeg using: ffmpeg -i "VIDEO.mp4" 2>&1 | grep "fps")
    
    # Find all h5 files in the current directory
    h5_files = sorted(glob.glob('*.h5'))
    
    if not h5_files:
        print("No .h5 files found in the current directory!")
        return None, None
    
    print("="*60)
    print("BATCH SINGLE ROI PREFERENCE ANALYSIS")
    print("="*60)
    print(f"Found {len(h5_files)} h5 file(s):")
    for i, file in enumerate(h5_files, 1):
        print(f"  {i}. {file}")
    print("="*60)
    
    # Variable to store the ROI from first file
    master_roi_name = None
    master_roi_file = None
    
    # Store results from all files
    all_results = []
    
    # Process each h5 file
    for file_index, h5_file in enumerate(h5_files, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING FILE {file_index}/{len(h5_files)}: {h5_file}")
        print(f"{'='*60}")
        
        base_filename = h5_file.replace('.h5', '')
        
        print(f"FPS: {FPS}")
        print(f"Analyzing ENTIRE recording")
        
        # Load data
        analyzer = DeepLabCutROIAnalyzer(h5_file)
        
        roi_name = None
        
        if file_index == 1:
            # First file - draw or select ROI
            roi_files = glob.glob('*_single_roi_definition.json')
            
            if roi_files:
                print(f"\nFound {len(roi_files)} ROI definition(s):")
                for i, file in enumerate(roi_files, 1):
                    print(f"  {i}. {file}")
                
                print(f"\nOptions:")
                print(f"  1-{len(roi_files)}: Use selected ROI")
                print(f"  n: Draw new ROI")
                
                choice = input("Your choice: ").strip().lower()
                
                if choice == 'n':
                    print("\nDrawing new ROI...")
                    analyzer.rois = {}
                    roi_name = draw_single_roi(analyzer)
                    if roi_name:
                        # Save this ROI
                        temp_roi_file = f'{base_filename}_single_roi_definition.json'
                        analyzer.save_rois(temp_roi_file)
                        master_roi_file = temp_roi_file
                else:
                    try:
                        roi_index = int(choice) - 1
                        if 0 <= roi_index < len(roi_files):
                            master_roi_file = roi_files[roi_index]
                            print(f"\nLoading ROI from: {master_roi_file}")
                            analyzer.load_rois(master_roi_file)
                            if analyzer.rois:
                                roi_name = list(analyzer.rois.keys())[0]
                                print(f"Loaded ROI: '{roi_name}'")
                    except:
                        print("Invalid selection. Drawing new ROI...")
                        roi_name = draw_single_roi(analyzer)
                        if roi_name:
                            temp_roi_file = f'{base_filename}_single_roi_definition.json'
                            analyzer.save_rois(temp_roi_file)
                            master_roi_file = temp_roi_file
            else:
                print("\nNo ROI definitions found. Drawing new ROI...")
                roi_name = draw_single_roi(analyzer)
                if roi_name:
                    temp_roi_file = f'{base_filename}_single_roi_definition.json'
                    analyzer.save_rois(temp_roi_file)
                    master_roi_file = temp_roi_file
            
            master_roi_name = roi_name
            
        else:
            # Subsequent files - use the ROI from first file
            if master_roi_file and master_roi_name:
                print(f"\n✓ Automatically using ROI from first file")
                analyzer.load_rois(master_roi_file)
                roi_name = master_roi_name
                print(f"  ROI: '{roi_name}'")
        
        if not roi_name:
            print("No ROI defined. Skipping this file.")
            continue
        
        # Show recording duration
        total_frames = len(analyzer.df)
        total_duration_s = total_frames / FPS
        total_duration_min = total_duration_s / 60
        
        print(f"\nRecording Info:")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_duration_s:.1f} seconds ({total_duration_min:.1f} minutes)")
        
        try:
            # Calculate metrics
            metrics, in_roi, velocity = calculate_preference_metrics(analyzer, roi_name, fps=FPS)
            
            # Add filename to metrics
            metrics['filename'] = h5_file
            
            # Print results
            print("\n" + "="*60)
            print("RESULTS")
            print("="*60)
            print(f"Time in ROI: {metrics['percent_time_in_roi']:.1f}% (expected: {metrics['expected_percent_time']:.1f}%)")
            print(f"Preference Index: {metrics['preference_index_classic']:+.1f}%")
            print(f"Entries: {metrics['number_of_entries']:.0f}")
            
            # Statistical significance
            if metrics['binomial_test_p'] < 0.001:
                sig = "***"
            elif metrics['binomial_test_p'] < 0.01:
                sig = "**"
            elif metrics['binomial_test_p'] < 0.05:
                sig = "*"
            else:
                sig = "ns"
            print(f"p-value: {metrics['binomial_test_p']:.4f} {sig}")
            
            # Interpretation
            if metrics['binomial_test_p'] < 0.05:
                if metrics['preference_index_classic'] > 0:
                    print(f"✓ SIGNIFICANT PREFERENCE")
                else:
                    print(f"✓ SIGNIFICANT AVOIDANCE")
            else:
                print(f"○ NO SIGNIFICANT PREFERENCE")
            
            # Save individual results
            results_df = pd.DataFrame([metrics])
            results_filename = f'{base_filename}_single_roi_analysis.csv'
            results_df.to_csv(results_filename, index=False)
            print(f"\nResults saved: {results_filename}")
            
            # Create and save visualization
            fig = create_visualizations(analyzer, metrics, in_roi, velocity, roi_name, fps=FPS)
            fig_filename = f'{base_filename}_single_roi_visualization.png'
            plt.savefig(fig_filename, dpi=150, bbox_inches='tight')
            plt.close()  # Close to avoid showing all plots at once
            print(f"Figure saved: {fig_filename}")
            
            # Store results
            all_results.append(metrics)
            
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            continue
    
    # Create summary table
    if all_results:
        print("\n" + "="*60)
        print("BATCH SUMMARY")
        print("="*60)
        
        summary_df = pd.DataFrame(all_results)
        summary_cols = ['filename', 'percent_time_in_roi', 'preference_index_classic', 
                       'number_of_entries', 'binomial_test_p', 'cohens_d_effect_size']
        summary_df = summary_df[summary_cols]
        
        # Save summary
        summary_filename = 'batch_roi_analysis_summary.csv'
        summary_df.to_csv(summary_filename, index=False)
        print(f"\nSummary saved: {summary_filename}")
        
        # Print summary
        print("\nRESULTS OVERVIEW:")
        print("-"*60)
        for _, row in summary_df.iterrows():
            sig = "***" if row['binomial_test_p'] < 0.001 else "**" if row['binomial_test_p'] < 0.01 else "*" if row['binomial_test_p'] < 0.05 else "ns"
            print(f"{row['filename']:30s}: {row['percent_time_in_roi']:5.1f}% (PI={row['preference_index_classic']:+5.1f}%, p={row['binomial_test_p']:.4f} {sig})")
    
    print("\n" + "="*60)
    print("BATCH ANALYSIS COMPLETE!")
    print("="*60)
    
    return None, None


if __name__ == "__main__":
    analyzer, metrics = analyze_single_roi()
