import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from collections import namedtuple
import os
from pathlib import Path

# Import the functions from your uploaded script
from time_in_each_roi import get_timeinrois_stats, get_roi_at_each_frame

class DeepLabCutROIAnalyzer:
    def __init__(self, h5_file_path):
        """
        Initialize the analyzer with a DeepLabCut h5 file
        
        Args:
            h5_file_path: Path to the DeepLabCut .h5 tracking file
        """
        self.h5_file_path = h5_file_path
        self.df = None
        self.bodyparts = None
        self.scorer = None
        self.arena_bounds = None
        self.rois = {}
        self.position = namedtuple('position', ['topleft', 'bottomright'])
        
        # Load the data
        self.load_deeplabcut_data()
        
    def load_deeplabcut_data(self):
        """Load DeepLabCut h5 file and extract tracking data"""
        print(f"Loading data from: {self.h5_file_path}")
        
        # Read the h5 file
        self.df = pd.read_hdf(self.h5_file_path)
        
        # Get the scorer name (usually 'DLC_resnet50_...' or similar)
        self.scorer = self.df.columns.get_level_values(0)[0]
        
        # Get list of tracked bodyparts
        self.bodyparts = list(self.df[self.scorer].columns.get_level_values(0).unique())
        
        print(f"Scorer: {self.scorer}")
        print(f"Bodyparts found: {self.bodyparts}")
        print(f"Number of frames: {len(self.df)}")
        
        # Calculate arena bounds from the data
        self.calculate_arena_bounds()
        
    def calculate_arena_bounds(self, padding=50):
        """
        Calculate the arena boundaries from tracking data
        
        Args:
            padding: Extra pixels to add around the detected boundaries
        """
        all_x = []
        all_y = []
        
        for bp in self.bodyparts:
            x_data = self.df[self.scorer][bp]['x'].values
            y_data = self.df[self.scorer][bp]['y'].values
            
            # Filter out low likelihood points if likelihood data exists
            if 'likelihood' in self.df[self.scorer][bp].columns:
                likelihood = self.df[self.scorer][bp]['likelihood'].values
                # Only use high confidence points (likelihood > 0.9)
                high_conf = likelihood > 0.9
                x_data = x_data[high_conf]
                y_data = y_data[high_conf]
            
            all_x.extend(x_data[~np.isnan(x_data)])
            all_y.extend(y_data[~np.isnan(y_data)])
        
        if all_x and all_y:
            self.arena_bounds = {
                'x_min': np.min(all_x) - padding,
                'x_max': np.max(all_x) + padding,
                'y_min': np.min(all_y) - padding,
                'y_max': np.max(all_y) + padding,
                'width': np.max(all_x) - np.min(all_x) + 2*padding,
                'height': np.max(all_y) - np.min(all_y) + 2*padding
            }
            
            print(f"\nArena boundaries detected:")
            print(f"X range: {self.arena_bounds['x_min']:.1f} - {self.arena_bounds['x_max']:.1f}")
            print(f"Y range: {self.arena_bounds['y_min']:.1f} - {self.arena_bounds['y_max']:.1f}")
            print(f"Arena size: {self.arena_bounds['width']:.1f} x {self.arena_bounds['height']:.1f} pixels")
        else:
            print("Warning: Could not calculate arena bounds from data")
            
    def plot_trajectory_with_heatmap(self, bodypart=None, sample_every=1):
        """
        Plot the trajectory of a bodypart with a heatmap showing time spent
        
        Args:
            bodypart: Which bodypart to plot (if None, uses first available)
            sample_every: Sample every nth frame for cleaner visualization
        """
        if bodypart is None:
            bodypart = self.bodyparts[0]
            
        # Get data for this bodypart
        x = self.df[self.scorer][bodypart]['x'].values[::sample_every]
        y = self.df[self.scorer][bodypart]['y'].values[::sample_every]
        
        # Filter out NaN values
        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Trajectory
        ax1 = axes[0]
        ax1.plot(x, y, 'b-', alpha=0.3, linewidth=0.5)
        ax1.scatter(x[0], y[0], color='green', s=100, label='Start', zorder=5)
        ax1.scatter(x[-1], y[-1], color='red', s=100, label='End', zorder=5)
        ax1.set_xlim(self.arena_bounds['x_min'], self.arena_bounds['x_max'])
        ax1.set_ylim(self.arena_bounds['y_min'], self.arena_bounds['y_max'])
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.set_title(f'Trajectory - {bodypart}')
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.invert_yaxis()  # Invert y-axis to match image coordinates
        
        # Plot 2: Heatmap
        ax2 = axes[1]
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax2.imshow(heatmap.T, origin='lower', extent=extent, 
                        cmap='hot', interpolation='gaussian')
        ax2.set_xlim(self.arena_bounds['x_min'], self.arena_bounds['x_max'])
        ax2.set_ylim(self.arena_bounds['y_min'], self.arena_bounds['y_max'])
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.set_title(f'Occupancy Heatmap - {bodypart}')
        ax2.set_aspect('equal')
        plt.colorbar(im, ax=ax2, label='Time spent (frames)')
        
        plt.tight_layout()
        plt.show()
        
        return fig
        
    def define_roi_interactive(self):
        """
        Interactive tool to define ROIs by clicking on the arena
        """
        print("\n=== Interactive ROI Definition ===")
        print("Instructions:")
        print("1. Click to define top-left corner of ROI")
        print("2. Click to define bottom-right corner of ROI")
        print("3. Enter ROI name when prompted")
        print("4. Press 'q' to finish defining ROIs")
        print("5. Press 'c' to clear all ROIs")
        
        # Use the first bodypart for visualization
        bodypart = self.bodyparts[0]
        x = self.df[self.scorer][bodypart]['x'].values
        y = self.df[self.scorer][bodypart]['y'].values
        
        # Filter out NaN values
        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot trajectory as background
        ax.plot(x[::10], y[::10], 'gray', alpha=0.2, linewidth=0.5)
        ax.set_xlim(self.arena_bounds['x_min'], self.arena_bounds['x_max'])
        ax.set_ylim(self.arena_bounds['y_min'], self.arena_bounds['y_max'])
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title('Click to define ROIs (press q to quit, c to clear)')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        clicks = []
        roi_patches = []
        
        def onclick(event):
            if event.inaxes != ax:
                return
                
            clicks.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro', markersize=8)
            
            if len(clicks) == 2:
                # Draw rectangle
                x1, y1 = clicks[0]
                x2, y2 = clicks[1]
                
                # Create rectangle (handling coordinate ordering)
                rect_x = min(x1, x2)
                rect_y = min(y1, y2)
                rect_width = abs(x2 - x1)
                rect_height = abs(y2 - y1)
                
                rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                               linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                roi_patches.append(rect)
                
                # Ask for ROI name
                roi_name = input("Enter name for this ROI: ")
                
                # Store ROI
                self.rois[roi_name] = self.position(
                    topleft=(min(x1, x2), min(y1, y2)),
                    bottomright=(max(x1, x2), max(y1, y2))
                )
                
                # Add text label
                ax.text(rect_x + rect_width/2, rect_y + rect_height/2, roi_name,
                       ha='center', va='center', fontsize=12, color='red',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                print(f"ROI '{roi_name}' defined: {self.rois[roi_name]}")
                
                # Clear clicks for next ROI
                clicks.clear()
                
            plt.draw()
            
        def onkey(event):
            if event.key == 'q':
                plt.close()
            elif event.key == 'c':
                # Clear all ROIs
                self.rois.clear()
                clicks.clear()
                for patch in roi_patches:
                    patch.remove()
                roi_patches.clear()
                # Redraw
                plt.draw()
                print("All ROIs cleared")
                
        cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
        cid_key = fig.canvas.mpl_connect('key_press_event', onkey)
        
        plt.show()
        
        print(f"\nTotal ROIs defined: {len(self.rois)}")
        for name, roi in self.rois.items():
            print(f"  {name}: {roi}")
            
    def define_roi_grid(self, n_rows, n_cols, roi_prefix="zone"):
        """
        Define a grid of ROIs automatically
        
        Args:
            n_rows: Number of rows in the grid
            n_cols: Number of columns in the grid
            roi_prefix: Prefix for ROI names (e.g., "zone_1_1", "zone_1_2", etc.)
        """
        x_min = self.arena_bounds['x_min']
        x_max = self.arena_bounds['x_max']
        y_min = self.arena_bounds['y_min']
        y_max = self.arena_bounds['y_max']
        
        x_step = (x_max - x_min) / n_cols
        y_step = (y_max - y_min) / n_rows
        
        for row in range(n_rows):
            for col in range(n_cols):
                roi_name = f"{roi_prefix}_{row+1}_{col+1}"
                
                topleft = (x_min + col * x_step, y_min + row * y_step)
                bottomright = (x_min + (col + 1) * x_step, y_min + (row + 1) * y_step)
                
                self.rois[roi_name] = self.position(topleft=topleft, bottomright=bottomright)
                
        print(f"Created {n_rows}x{n_cols} grid with {len(self.rois)} ROIs")
        
    def define_roi_manual(self, roi_name, x1, y1, x2, y2):
        """
        Manually define a single ROI with coordinates
        
        Args:
            roi_name: Name for the ROI
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
        """
        self.rois[roi_name] = self.position(
            topleft=(min(x1, x2), min(y1, y2)),
            bottomright=(max(x1, x2), max(y1, y2))
        )
        print(f"ROI '{roi_name}' defined: {self.rois[roi_name]}")
        
    def visualize_rois(self, bodypart=None):
        """
        Visualize all defined ROIs with trajectory overlay
        
        Args:
            bodypart: Which bodypart to show (if None, uses first available)
        """
        if not self.rois:
            print("No ROIs defined yet!")
            return
            
        if bodypart is None:
            bodypart = self.bodyparts[0]
            
        # Get data for this bodypart
        x = self.df[self.scorer][bodypart]['x'].values
        y = self.df[self.scorer][bodypart]['y'].values
        
        # Filter out NaN values
        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot trajectory
        ax.plot(x[::10], y[::10], 'b-', alpha=0.3, linewidth=0.5, label='Trajectory')
        
        # Plot ROIs
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.rois)))
        
        for (roi_name, roi), color in zip(self.rois.items(), colors):
            rect = Rectangle(
                roi.topleft,
                roi.bottomright[0] - roi.topleft[0],
                roi.bottomright[1] - roi.topleft[1],
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=0.2
            )
            ax.add_patch(rect)
            
            # Add label
            center_x = (roi.topleft[0] + roi.bottomright[0]) / 2
            center_y = (roi.topleft[1] + roi.bottomright[1]) / 2
            ax.text(center_x, center_y, roi_name,
                   ha='center', va='center', fontsize=10, fontweight='bold')
            
        ax.set_xlim(self.arena_bounds['x_min'], self.arena_bounds['x_max'])
        ax.set_ylim(self.arena_bounds['y_min'], self.arena_bounds['y_max'])
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(f'Defined ROIs with {bodypart} trajectory')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig
        
    def analyze_roi_occupancy(self, bodypart=None, fps=30):
        """
        Analyze time spent in each ROI using the imported functions
        
        Args:
            bodypart: Which bodypart to analyze (if None, uses first available)
            fps: Frames per second of the video
            
        Returns:
            Dictionary with ROI statistics
        """
        if not self.rois:
            print("No ROIs defined! Please define ROIs first.")
            return None
            
        if bodypart is None:
            bodypart = self.bodyparts[0]
            print(f"Using bodypart: {bodypart}")
            
        # Get tracking data
        x = self.df[self.scorer][bodypart]['x'].values
        y = self.df[self.scorer][bodypart]['y'].values
        
        # Calculate velocity (pixels/frame)
        velocity = np.zeros(len(x))
        velocity[1:] = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        
        # Prepare data in the format expected by the ROI analysis functions
        tracking_data = np.column_stack((x, y, velocity))
        
        # Remove NaN values
        valid_idx = ~(np.isnan(x) | np.isnan(y))
        tracking_data = tracking_data[valid_idx]
        
        # Run the analysis
        results = get_timeinrois_stats(tracking_data, self.rois, fps=fps, returndf=True)
        
        return results
        
    def save_rois(self, filename):
        """Save ROI definitions to a file"""
        import json
        
        # Convert namedtuples to dictionaries for JSON serialization
        rois_dict = {}
        for name, roi in self.rois.items():
            rois_dict[name] = {
                'topleft': list(roi.topleft),
                'bottomright': list(roi.bottomright)
            }
            
        with open(filename, 'w') as f:
            json.dump(rois_dict, f, indent=2)
            
        print(f"ROIs saved to {filename}")
        
    def load_rois(self, filename):
        """Load ROI definitions from a file"""
        import json
        
        with open(filename, 'r') as f:
            rois_dict = json.load(f)
            
        self.rois = {}
        for name, coords in rois_dict.items():
            self.rois[name] = self.position(
                topleft=tuple(coords['topleft']),
                bottomright=tuple(coords['bottomright'])
            )
            
        print(f"Loaded {len(self.rois)} ROIs from {filename}")


# Example usage
if __name__ == "__main__":
    # Example workflow
    print("DeepLabCut ROI Analysis Example")
    print("="*50)
    
    # Initialize with your h5 file
    # analyzer = DeepLabCutROIAnalyzer("path/to/your/file.h5")
    
    # Visualize trajectory and heatmap
    # analyzer.plot_trajectory_with_heatmap()
    
    # Define ROIs using different methods:
    
    # Method 1: Interactive definition
    # analyzer.define_roi_interactive()
    
    # Method 2: Grid-based definition
    # analyzer.define_roi_grid(n_rows=3, n_cols=3, roi_prefix="zone")
    
    # Method 3: Manual definition
    # analyzer.define_roi_manual("center", 300, 300, 500, 500)
    # analyzer.define_roi_manual("corner", 0, 0, 200, 200)
    
    # Visualize ROIs
    # analyzer.visualize_rois()
    
    # Analyze occupancy
    # results = analyzer.analyze_roi_occupancy(fps=30)
    # print("\nROI Occupancy Results:")
    # print(results)
    
    # Save ROIs for later use
    # analyzer.save_rois("my_rois.json")
    
    # Load previously saved ROIs
    # analyzer.load_rois("my_rois.json")
