# DeepLabCut ROI analysis 
Interactive tool for exploring regions of interest (ROIs) in DeepLabCut pose estimation data. This repo helps you analyze time spent in different regions of interest (ROIs) from DeepLabCut tracking data.🐀🐁

## Author: Wiktoria Zaniewska

## Files 
1. **`inspect_dlc_h5.py`** - Utility to inspect your h5 files before analysis. Fill (filepath). Optionally allows you to save the .txt file for further analysis.
2. **`time_in_each_roi.py`** - ROI analysis function available on DeepLabCut official repo (modified and integrated into the main class). 
3. **`deeplabcut_roi_analysis.py`** - Main analysis class with all functionality
4. **`single_ROI_statistics.py`** - Allows to draw or use preexisting ROI. Output explained in **`SINGLE_ROI_GUIDE.md`**

# 1. First, inspect your DeepLabCut h5 file to understand the data (inspect_dlc_h5.py)

```bash
python inspect_dlc_h5.py "FILE_NAME".h5
```
## This will show you:
- Available bodyparts
- Number of frames
- Arena boundaries - beware of the experimental cases where the Agent is static/not exploring the arena
- Data quality metrics

# 2. DeepLabCut ROI analysis (deeplabcut_roi_analysis.py)

## Methods for defining ROIs

### Method 1: Interactive (mostly for custom shapes)
```python
analyzer.define_roi_interactive()
```
- Click to define corners of rectangular ROIs
- Press Q to quit
- Press C to clear all ROIs

### Method 2: Automatic grid
```python
analyzer.define_roi_grid(n_rows=3, n_cols=3, roi_prefix="zone")
```

### Method 3: Manual with coordinates
```python
analyzer.define_roi_manual("center", x1, y1, x2, y2)
```

### Method 4: Common behavioral setups

#### Example --> Open Field Test (Center vs Periphery):
```python
# Get arena bounds
arena_width = analyzer.arena_bounds['width']
arena_height = analyzer.arena_bounds['height']
x_min = analyzer.arena_bounds['x_min']
y_min = analyzer.arena_bounds['y_min']

# Define center zone (in this case 1/3 of arena)
center_size = 0.33
center_x1 = x_min + arena_width * (0.5 - center_size/2)
center_y1 = y_min + arena_height * (0.5 - center_size/2)
center_x2 = x_min + arena_width * (0.5 + center_size/2)
center_y2 = y_min + arena_height * (0.5 + center_size/2)

analyzer.define_roi_manual("center", center_x1, center_y1, center_x2, center_y2)
```
## Output metrics

The analysis provides these metrics for each ROI:
- **transitions_per_roi**: Number of times animal entered the ROI
- **cumulative_time_in_roi**: Total frames spent in ROI
- **cumulative_time_in_roi_sec**: Total time in seconds
- **avg_time_in_roi**: Average frames per visit
- **avg_time_in_roi_sec**: Average seconds per visit
- **avg_vel_in_roi**: Average velocity (pixels/frame) while in ROI

# 3. Single ROI Preference Analysis (single_ROI_statistics.py)

## Main features:
- Interactive ROI drawing  (automatically saved for reuse)
- Processes entire recording
- Multiple preference metrics (classic index, discrimination index, exploration ratio)
- Statistical validation (tests against chance based on ROI size)
- Output: visual data in 9 frames and 
- TO USE for the next files in one experiment edit lines:
h5_file = "your_file.h5"
FPS = ex. 25



## Tips!

1. **Start with visualization** to understand your arena and animal movement patterns
2. **Use high-confidence tracking points** (likelihood > 0.9) for more accurate results
3. **Save ROI definitions** for consistent analysis across multiple files
4. **Check data quality** using the inspect script before analysis
5. **Consider your experimental design** when defining ROIs (for instance - center vs periphery for anxiety tests)

## Requirements

```bash
pip install pandas numpy matplotlib scipy h5py
```

⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
