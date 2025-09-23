import matplotlib.pyplot as plt
import numpy as np
import re

def plot_uv_spectrum(filepath):
    """
    Reads UV spectrum data from a file, finds the peak absorbance (λ_max),
    and generates an annotated plot with the x-axis increasing from left to right.

    Args:
        filepath (str): The path to the data file.
    """
    wavelengths = []
    absorbance = []
    title = "UV Spectrum" # Default title

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        data_start_index = -1
        for i, line in enumerate(lines):
            if line.upper().startswith('TITLE'):
                match = re.search(r'TITLE\s+(.*)', line)
                if match:
                    title = match.group(1).strip()
            
            if line.strip() == 'XYDATA':
                data_start_index = i + 1
                break

        if data_start_index == -1:
            print("Error: Could not find the 'XYDATA' section in the file.")
            return

        for line in lines[data_start_index:]:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split()
                if len(parts) >= 2:
                    wavelengths.append(float(parts[0]))
                    absorbance.append(float(parts[1]))
            except (ValueError, IndexError):
                continue

        if not wavelengths or not absorbance:
            print("No valid data points were parsed. Cannot generate plot.")
            return
        
        # Convert lists to NumPy arrays for efficient calculations
        wave_arr = np.array(wavelengths)
        abs_arr = np.array(absorbance)
        
        # Find the index of the maximum absorbance value
        peak_index = np.argmax(abs_arr)
        
        # Get the corresponding wavelength and absorbance at that peak
        lambda_max = wave_arr[peak_index]
        absorbance_max = abs_arr[peak_index]

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(wave_arr, abs_arr, color='purple', linewidth=2, label='Absorbance')

        # Annotate the Peak on the Plot
        ax.annotate(f'$\\lambda_{{max}} = {lambda_max:.0f}$ nm',
                    xy=(lambda_max, absorbance_max),
                    xytext=(lambda_max + 60, absorbance_max * 0.8), # Adjusted text position
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=12,
                    fontweight='bold')
        
        # Add a marker for the peak
        ax.plot(lambda_max, absorbance_max, 'o', color='red', markersize=8)

        # Set plot title and labels
        ax.set_title(f'UV-Vis Spectrum: {title}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Absorbance', fontsize=12)

        # --- THIS LINE WAS REMOVED ---
        # ax.invert_xaxis() 
        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the function with the provided file name
data_file = 'UV of fluorescein group-2F.txt'
plot_uv_spectrum(data_file)