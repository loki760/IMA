import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# --- (UPDATED) Parsing Function for your specific sample.txt format ---
def parse_reference_txt(file_path):
    """
    Parses a .txt reference file with d-spacing and 2-Theta columns.
    - Assigns a default intensity of 100 to all peaks.
    - Creates a placeholder for hkl as it's not in the file.
    """
    data = []
    print(f"Attempting to parse reference file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Skip empty lines
                if not line.strip():
                    continue
                
                parts = line.strip().split()
                # Skip lines that don't have at least 2 columns
                if len(parts) < 2:
                    continue
                
                try:
                    # Read 2-Theta from the second column (index 1)
                    two_theta = float(parts[1])

                    # --- Assign default values for missing data ---
                    intensity = 100.0  # Assign a default intensity of 100
                    hkl_label = '---'  # Placeholder for hkl since it's missing

                    data.append({
                        '2Theta': two_theta,
                        'Intensity': intensity,
                        'hkl': hkl_label
                    })
                except (ValueError):
                    # This will skip the text header line automatically
                    print(f"Skipping header/malformed line #{line_num} in {file_path}: {line.strip()}")
                    continue
        if not data:
             print(f"Warning: No valid data was found in '{file_path}'.")
        return pd.DataFrame(data)

    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_path}'")
        return pd.DataFrame()


# --- Data Parsing Function for Raw Data Files (.txt) ---
def parse_raw_xrd_data(file_path):
    """Parses a raw XRD data file, skipping the header."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        start_line = 0
        for i, line in enumerate(lines):
            try:
                float(line.strip().split()[0]), float(line.strip().split()[1])
                start_line = i
                break
            except (ValueError, IndexError):
                continue
        
        df = pd.read_csv(file_path, sep=r'\s+', skiprows=start_line, header=None, names=['2Theta', 'Intensity'])
        return df
    except FileNotFoundError:
        print(f"Error: Could not find file '{file_path}'")
        return pd.DataFrame()


# --- Main Script ---

# 1. Load All Data
exp_data_full = parse_raw_xrd_data('TiO2 file 1.TXT')
exp_data_zoom = parse_raw_xrd_data('TiO2 file 2.txt')
anatase_ref_df = parse_reference_txt('sample.txt')

# Exit if reference data could not be loaded
if anatase_ref_df.empty:
    print("\nHalting script because reference data is empty. Cannot generate plots.")
else:
    # --- GRAPH 1: Three-Panel Comparison Plot (Main Plot) ---
    print("\nGenerating Graph 1: Three-Panel Comparison Plot...")

    fig1, axs1 = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                            gridspec_kw={'height_ratios': [3, 1, 0.5]})
    fig1.suptitle('XRD Pattern Analysis of TiO₂ Sample', fontsize=18, y=0.95) # Simplified title

    # (a) Top Plot: Experimental Data
    axs1[0].plot(exp_data_full['2Theta'], exp_data_full['Intensity'], color='black', label='Experimental Data')
    axs1[0].set_ylabel('Intensity (counts)', fontsize=12)
    axs1[0].grid(linestyle=':', alpha=0.7)
    axs1[0].legend(loc='upper right')
    # NOTE: (h,k,l) labels for individual peaks are NOT added as this info is not in sample.txt

    # (b) Middle Plot: Reference Stick Pattern
    markerline, stemlines, baseline = axs1[1].stem(
        anatase_ref_df['2Theta'], 
        anatase_ref_df['Intensity'],
        linefmt='r-', 
        markerfmt=' ', 
        basefmt=" ",   
        label='Reference Peaks (from sample.txt)' # Updated label
    )
    plt.setp(stemlines, 'linewidth', 1.5) 
    axs1[1].set_ylabel('Reference Intensity\n(Arbitrary Units)', fontsize=12) # Updated ylabel
    axs1[1].grid(linestyle=':', alpha=0.7)
    axs1[1].legend(loc='upper right')
    axs1[1].set_ylim(bottom=0) # Ensure y-axis starts at 0

    # (c) Bottom Plot: Peak Position markers
    axs1[2].vlines(anatase_ref_df['2Theta'], ymin=-1, ymax=1, color='gray')
    axs1[2].set_xlabel('2θ (degrees)', fontsize=14)
    axs1[2].set_yticks([])
    axs1[2].set_ylim(-1, 1)

    plt.xlim(20, 85) # Adjusted x-limit to see all your data points
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('graph1_final_comparison.png')
    print("Saved 'graph1_final_comparison.png'")

    # --- GRAPH 2: FWHM Calculation from Real Data (Separate Plot) ---
    print("\nGenerating Graph 2: FWHM Calculation...")

    fig2 = plt.figure(figsize=(10, 7)) # Create a new figure for this plot

    plt.plot(exp_data_zoom['2Theta'], exp_data_zoom['Intensity'], 'o-', label='Experimental Data', color='blue')

    intensity = exp_data_zoom['Intensity']
    theta_2 = exp_data_zoom['2Theta']
    half_max = intensity.max() / 2
    peak_pos = theta_2[intensity.idxmax()]

    try:
        left_side = intensity[:intensity.idxmax()]
        left_indices = np.where(left_side > half_max)[0]
        
        right_side = intensity[intensity.idxmax():]
        right_indices = np.where(right_side < half_max)[0] + intensity.idxmax()

        if left_indices.size > 0 and right_indices.size > 0:
            left_idx_start = left_indices[0] 
            right_idx_start = right_indices[0] 

            x1 = np.interp(half_max, intensity.iloc[max(0, left_idx_start-1):left_idx_start+1], theta_2.iloc[max(0, left_idx_start-1):left_idx_start+1])
            x2 = np.interp(half_max, intensity.iloc[right_idx_start-1:min(len(intensity), right_idx_start+1)][::-1], theta_2.iloc[right_idx_start-1:min(len(theta_2), right_idx_start+1)][::-1])
            fwhm_val = x2 - x1

            plt.hlines(half_max, x1, x2, color='red', linestyle='--', label=f'Half Maximum: {half_max:.0f}')
            plt.axvline(peak_pos, color='gray', linestyle=':', label=f'Peak Max at {peak_pos:.3f}°')
            plt.annotate('', xy=(x1, half_max), xytext=(x2, half_max),
                         arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            plt.text((x1+x2)/2, half_max * 1.1, f'FWHM (B$_{{1/2}}$) = {fwhm_val:.3f}°',
                     horizontalalignment='center', color='red', fontsize=12)
        else:
            raise IndexError("Could not find points crossing half-maximum (might be outside current view or too noisy).")

    except (IndexError, ValueError) as e:
        print(f"Could not accurately determine FWHM for this peak: {e}. Adjust 'TiO2 file 2.txt' range or `find_peaks` parameters if needed.")

    plt.title('FWHM of TiO₂ Peak', fontsize=16) 
    plt.xlabel('2θ (degrees)', fontsize=12)
    plt.ylabel('Intensity (counts)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig('graph2_final_fwhm.png')
    print("Saved 'graph2_final_fwhm.png'")

    # --- GRAPH 3: Reference Data Only (Separate Plot, optional now but kept for consistency) ---
    print("\nGenerating Graph 3: Reference Pattern from sample.txt (separate plot)...")

    fig3 = plt.figure(figsize=(12, 6)) # Create a new figure for this plot

    markerline, stemlines, baseline = plt.stem(
        anatase_ref_df['2Theta'], 
        anatase_ref_df['Intensity'],
        linefmt='b-', 
        markerfmt=' ', 
        basefmt=" ",   
        label='Reference Peaks (from sample.txt)'
    )
    plt.setp(stemlines, 'linewidth', 1.5) 

    plt.title('Reference XRD Pattern (from sample.txt)', fontsize=16)
    plt.xlabel('2θ (degrees)', fontsize=12)
    plt.ylabel('Reference Intensity (Arbitrary Units)', fontsize=12)
    plt.xlim(20, 85) # Consistent x-axis range
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('graph3_reference_pattern.png')
    print("Saved 'graph3_reference_pattern.png'")

    print("\nAll graphs generated successfully!")