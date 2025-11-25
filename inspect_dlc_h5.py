#!/usr/bin/env python3
"""
First, understand the data structure.
Author: Wiktoria Zaniewska
"""

import pandas as pd
import numpy as np
import sys
from io import StringIO

def inspect_deeplabcut_h5(filepath):
    """
    Inspect a DeepLabCut h5 file and print useful information
    
    Args:
        filepath: Path to the .h5 file
    """
    print("\n" + "="*60)
    print(f"INSPECTING: {filepath}")
    print("="*60)
    
    # Load the data
    df = pd.read_hdf(filepath)
    
    # Get basic info
    print(f"BASIC INFORMATION:")
    print(f"   Total frames: {len(df)}")
    print(f"   Data shape: {df.shape}")
    
    # Get scorer info
    scorer = df.columns.get_level_values(0)[0]
    print(f"SCORER: {scorer}")
    
    # Get bodyparts
    bodyparts = list(df[scorer].columns.get_level_values(0).unique())
    print(f"BODYPARTS TRACKED ({len(bodyparts)}):")
    for i, bp in enumerate(bodyparts, 1):
        print(f"   {i}. {bp}")
    
    # Get coordinates for each bodypart
    print(f"COORDINATES PER BODYPART:")
    first_bp = bodyparts[0]
    coords = list(df[scorer][first_bp].columns)
    print(f"   Coordinates: {coords}")
    
    # Check for likelihood values
    has_likelihood = 'likelihood' in coords
    if has_likelihood:
        print(f"Likelihood scores: Available")
        
        # Show likelihood statistics
        print(f" LIKELIHOOD STATISTICS:")
        for bp in bodyparts[:3]:  # Show first 3 bodyparts
            likelihood = df[scorer][bp]['likelihood'].values
            print(f"   {bp}:")
            print(f"      Mean: {np.mean(likelihood):.3f}")
            print(f"      Min: {np.min(likelihood):.3f}")
            print(f"      Max: {np.max(likelihood):.3f}")
            print(f"      % > 0.9: {(likelihood > 0.9).sum() / len(likelihood) * 100:.1f}%")
    else:
        print(f"Likelihood scores: Not available")
    
    # Calculate arena bounds (excluding low confidence points)
    print(f"ARENA BOUNDS (from high confidence points):")
    all_x = []
    all_y = []
    
    for bp in bodyparts:
        x_data = df[scorer][bp]['x'].values
        y_data = df[scorer][bp]['y'].values
        
        if has_likelihood:
            likelihood = df[scorer][bp]['likelihood'].values
            high_conf = likelihood > 0.9
            x_data = x_data[high_conf]
            y_data = y_data[high_conf]
        
        valid = ~(np.isnan(x_data) | np.isnan(y_data))
        all_x.extend(x_data[valid])
        all_y.extend(y_data[valid])
    
    if all_x and all_y:
        print(f"   X range: {np.min(all_x):.1f} - {np.max(all_x):.1f} pixels")
        print(f"   Y range: {np.min(all_y):.1f} - {np.max(all_y):.1f} pixels")
        print(f"   Arena size: {np.max(all_x) - np.min(all_x):.1f} x {np.max(all_y) - np.min(all_y):.1f} pixels")
    
    # Check for NaN values
    print(f"DATA QUALITY CHECK:")
    for bp in bodyparts[:3]:  # Check first 3 bodyparts
        x_data = df[scorer][bp]['x'].values
        y_data = df[scorer][bp]['y'].values
        
        nan_x = np.isnan(x_data).sum()
        nan_y = np.isnan(y_data).sum()
        
        print(f"   {bp}:")
        print(f"      NaN in X: {nan_x} ({nan_x/len(x_data)*100:.1f}%)")
        print(f"      NaN in Y: {nan_y} ({nan_y/len(y_data)*100:.1f}%)")
    
    # Sample data preview
    print(f"FIRST 5 FRAMES OF DATA (for '{bodyparts[0]}'):")
    print(df[scorer][bodyparts[0]].head())
    
    print("\n" + "="*60)
    print("INSPECTION COMPLETE")
    print("="*60)
    
    return df, scorer, bodyparts


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_dlc_h5.py <path_to_h5_file>")
        print("\nExample:")
        print("  python inspect_dlc_h5.py my_tracking_data.h5")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    try:
        df, scorer, bodyparts = inspect_deeplabcut_h5(filepath)
        
        # Optional: Save a summary
        save_summary = input("\nSave inspection summary to text file? (y/n): ")
        if save_summary.lower() == 'y':
            summary_file = filepath.replace('.h5', '_inspection_summary.txt')
            
            # Redirect print to file
            old_stdout = sys.stdout
            sys.stdout = summary = StringIO()
            
            # Re-run inspection to capture output
            inspect_deeplabcut_h5(filepath)
            
            # Write to file
            with open(summary_file, 'w') as f:
                f.write(summary.getvalue())
            
            sys.stdout = old_stdout
            print(f"Summary saved to: {summary_file}")
            
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
