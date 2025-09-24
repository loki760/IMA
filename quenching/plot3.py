import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import re
import os

def get_peak_intensity(filepath):
    """
    Parses a fluorescence data file and returns the maximum intensity.
    """
    intensities = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        data_start_index = -1
        for i, line in enumerate(lines):
            if line.strip().upper() == 'XYDATA':
                data_start_index = i + 1
                break
        
        if data_start_index == -1: return None

        for line in lines[data_start_index:]:
            try:
                parts = line.split()
                if len(parts) >= 2:
                    intensities.append(float(parts[1]))
            except (ValueError, IndexError):
                continue
        
        return max(intensities) if intensities else None

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

# --- 1. Experimental Setup ---
stock_ki_conc = 1.0  # Molarity (M)
total_volume = 10.0  # Total volume in mL

files_and_volumes = {
    'fluorescein _0mL KI.txt': 0.0,
    'fluorescein _1mL KI.txt': 1.0,
    'fluorescein _2mL KI.txt': 2.0,
    'fluorescein _3mL KI.txt': 3.0,
    'fluorescein _4mL KI.txt': 4.0,
}

# --- 2. Process Data ---
I_o = get_peak_intensity('fluorescein _0mL KI.txt')

if I_o is None:
    print("Could not determine the initial intensity (I_o). Aborting.")
else:
    quencher_conc = []
    intensity_ratio = []

    for filename, ki_volume in files_and_volumes.items():
        final_conc = (stock_ki_conc * ki_volume) / total_volume
        quencher_conc.append(final_conc)
        
        I = get_peak_intensity(filename)
        if I is not None:
            intensity_ratio.append(I_o / I)

    quencher_conc = np.array(quencher_conc)
    intensity_ratio = np.array(intensity_ratio)

    # --- 3. Perform Linear Regression with Intercept Forced to 1 ---
    # We are fitting the model y = mx + 1, which is equivalent to y - 1 = mx
    y_transformed = intensity_ratio - 1
    x_transformed = quencher_conc.reshape(-1, 1) # Reshape for lstsq

    # Use least squares to find the slope 'm' for the transformed data
    slope, _, _, _ = np.linalg.lstsq(x_transformed, y_transformed, rcond=None)
    
    K_sv = slope[0]      # The calculated slope is our K_sv
    intercept = 1.0    # We are fixing the intercept to 1.0
    
    # Manually calculate R-squared for the fixed-intercept fit
    y_pred = K_sv * quencher_conc + 1
    ss_res = np.sum((intensity_ratio - y_pred)**2)
    ss_tot = np.sum((intensity_ratio - np.mean(intensity_ratio))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # --- 4. Generate the Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(quencher_conc, intensity_ratio, marker='s', color='black', label='Experimental Data ($I_o/I$)')
    
    fit_line_x = np.linspace(0, quencher_conc.max(), 10)
    fit_line_y = intercept + K_sv * fit_line_x # Use the new K_sv and fixed intercept
    ax.plot(fit_line_x, fit_line_y, color='red', linewidth=2, label='Linear Fit (Intercept = 1)')

    # --- 5. Customize the Plot ---
    ax.set_title('Stern-Volmer Plot (Intercept Forced to 1)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Final [KI] (M)', fontsize=12)
    ax.set_ylabel('$I_o / I$', fontsize=12)
    ax.legend()

    # --- 6. Add Results Box ---
    results_text = (
        f"Forced-Intercept Fit Results:\n"
        f"---------------------------------\n"
        f"Equation:  $y = 1.0 + {K_sv:.2f}x$\n"
        f"Intercept:   1.0 (Fixed)\n"
        f"Slope ($K_{{sv}}$): {K_sv:.2f} $M^{{-1}}$\n"
        f"R-Squared:   {r_squared:.4f}"
    )
    
    ax.text(0.05, 0.95, results_text, 
            transform=ax.transAxes, 
            fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))

    plt.tight_layout()
    plt.show()