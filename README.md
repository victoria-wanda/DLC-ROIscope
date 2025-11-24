# DeepLabCut ROI Analysis 
Interactive tool for exploring regions of interest (ROIs) in DeepLabCut pose estimation data. This repo helps you analyze time spent in different regions of interest (ROIs) from DeepLabCut tracking data.

## Author: Wiktoria Zaniewska

## Files 
1. **`inspect_dlc_h5.py`** - Utility to inspect your h5 files before analysis. Fill (filepath). Optionally allows you to save the .txt file for further analysis.
2. **`time_in_each_roi.py`** - ROI analysis function available on DeepLabCut official repo (modified and integrated into the main class). 
3. **`deeplabcut_roi_analysis.py`** - Main analysis class with all functionality
4. **`example_roi_usage.py`** - Complete example showing how to use the toolkit
5. **`single_ROI_statistics.py`** - Allows to draw or use preexisting ROI. Output explained in **`SINGLE_ROI_GUIDE.md`**

# Start

## 1. First, inspect your DeepLabCut h5 file to understand the data:

```bash
python inspect_dlc_h5.py "FILENAME".h5
```

## This will show you:
- Available bodyparts
- Number of frames
- Arena boundaries - beware of the experimental cases where the Agent is static/not exploring the arena
- Data quality metrics

## 2. Basic usage for ROI analysis:

```python
from deeplabcut_roi_analysis import DeepLabCutROIAnalyzer

# Load your data
analyzer = DeepLabCutROIAnalyzer("FILENAME.h5")

# Visualize trajectory and heatmap
analyzer.plot_trajectory_with_heatmap(bodypart="nose")

# Define ROIs interactively (opens a clickable window)
analyzer.define_roi_interactive()

# Or define a grid of ROIs
analyzer.define_roi_grid(n_rows=3, n_cols=3)

# Visualize your ROIs
analyzer.visualize_rois()

# Analyze time spent in each ROI
results = analyzer.analyze_roi_occupancy(fps=30)  # Set your video FPS
print(results)

# Save results
results.to_csv("roi_analysis_results.csv")
```

## Methods for Defining ROIs

### Method 1: Interactive (Recommended for custom shapes)
```python
analyzer.define_roi_interactive()
```
- Click to define corners of rectangular ROIs
- Press 'q' to quit
- Press 'c' to clear all ROIs

### Method 2: Automatic Grid
```python
analyzer.define_roi_grid(n_rows=3, n_cols=3, roi_prefix="zone")
```

### Method 3: Manual with Coordinates
```python
analyzer.define_roi_manual("center", x1, y1, x2, y2)
```

### Method 4: Common Behavioral Setups

#### Open Field Test (Center vs Periphery):
```python
# Get arena bounds
arena_width = analyzer.arena_bounds['width']
arena_height = analyzer.arena_bounds['height']
x_min = analyzer.arena_bounds['x_min']
y_min = analyzer.arena_bounds['y_min']

# Define center zone (1/3 of arena)
center_size = 0.33
center_x1 = x_min + arena_width * (0.5 - center_size/2)
center_y1 = y_min + arena_height * (0.5 - center_size/2)
center_x2 = x_min + arena_width * (0.5 + center_size/2)
center_y2 = y_min + arena_height * (0.5 + center_size/2)

analyzer.define_roi_manual("center", center_x1, center_y1, center_x2, center_y2)
```

## Saving and Reusing ROI Definitions

Save your ROI definitions:
```python
analyzer.save_rois("my_rois.json")
```

Load them for other files:
```python
analyzer = DeepLabCutROIAnalyzer("another_file.h5")
analyzer.load_rois("my_rois.json")
```

## Batch Processing Multiple Files

```python
import pandas as pd
from deeplabcut_roi_analysis import DeepLabCutROIAnalyzer

roi_file = "my_rois.json"
h5_files = ["file1.h5", "file2.h5", "file3.h5"]

all_results = []
for h5_file in h5_files:
    analyzer = DeepLabCutROIAnalyzer(h5_file)
    analyzer.load_rois(roi_file)
    results = analyzer.analyze_roi_occupancy(bodypart="nose", fps=30)
    results['filename'] = h5_file
    all_results.append(results)

# Combine all results
combined = pd.concat(all_results)
combined.to_csv("batch_analysis.csv")
```

## Output metrics

The analysis provides these metrics for each ROI:
- **transitions_per_roi**: Number of times animal entered the ROI
- **cumulative_time_in_roi**: Total frames spent in ROI
- **cumulative_time_in_roi_sec**: Total time in seconds
- **avg_time_in_roi**: Average frames per visit
- **avg_time_in_roi_sec**: Average seconds per visit
- **avg_vel_in_roi**: Average velocity (pixels/frame) while in ROI

## Tips!

1. **Start with visualization** to understand your arena and animal movement patterns
2. **Use high-confidence tracking points** (likelihood > 0.9) for more accurate results
3. **Save ROI definitions** for consistent analysis across multiple files
4. **Check data quality** using the inspect script before analysis
5. **Consider your experimental design** when defining ROIs (for instance - center vs periphery for anxiety tests)

## Requirements

- pandas
- numpy
- matplotlib
- scipy
- h5py

Install with:
```bash
pip install pandas numpy matplotlib scipy h5py
```

## Troubleshooting

**Problem**: "No ROIs defined"
**Solution**: Make sure to define ROIs before running analysis

**Problem**: Low confidence tracking points affecting results
**Solution**: The analyzer automatically filters points with likelihood < 0.9

**Problem**: ROIs don't match arena
**Solution**: Use `visualize_rois()` to check ROI placement and adjust coordinates

## Example Workflow

See `example_roi_usage.py` for a complete, commented workflow.
