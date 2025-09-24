import matplotlib.pyplot as plt
import re
import os

def parse_spectrum_data(filepath):
    """
    Reads spectrum data from a file, parses it, and extracts a label from the filename.

    Args:
        filepath (str): The path to the data file.

    Returns:
        tuple: A tuple containing (wavelengths, intensities, label) or (None, None, None) on error.
    """
    wavelengths = []
    intensities = []

    # --- Create a label from the filename ---
    # Extracts the part like "0mL KI" from "fluorescein _0mL KI.txt"
    label = "Unknown"
    match = re.search(r'(\d+mL\s*KI)', os.path.basename(filepath))
    if match:
        label = match.group(1).strip()

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Find where the actual data starts
        data_start_index = -1
        for i, line in enumerate(lines):
            if line.strip().upper() == 'XYDATA':
                data_start_index = i + 1
                break

        if data_start_index == -1:
            print(f"Warning: Could not find 'XYDATA' in {filepath}. Skipping file.")
            return None, None, None

        # Process the data lines
        for line in lines[data_start_index:]:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split()
                if len(parts) >= 2:
                    wavelengths.append(float(parts[0]))
                    intensities.append(float(parts[1]))
            except (ValueError, IndexError):
                # Silently skip malformed lines as they are common at EOF
                continue

        return wavelengths, intensities, label

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while processing {filepath}: {e}")
        return None, None, None

def plot_all_spectra(filepaths):
    """
    Plots multiple fluorescence spectra on a single graph.

    Args:
        filepaths (list): A list of paths to the data files.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Use a colormap to get a nice sequence of colors
    colors = plt.cm.viridis_r([i/len(filepaths) for i in range(len(filepaths))])

    for i, filepath in enumerate(filepaths):
        wavelengths, intensities, label = parse_spectrum_data(filepath)

        if wavelengths and intensities:
            ax.plot(wavelengths, intensities, label=label, color=colors[i], linewidth=2)

    # --- Configure the plot ---
    ax.set_title('Fluorescence Quenching of Fluorescein by KI', fontsize=18, fontweight='bold')
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Fluorescence Intensity (a.u.)', fontsize=14)
    ax.legend(title='KI Volume', fontsize=12)

    # Adjust grid and spines for a cleaner look
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # List of the files you want to plot together
    files_to_plot = [
        'fluorescein _0mL KI.txt',
        'fluorescein _1mL KI.txt',
        'fluorescein _2mL KI.txt',
        'fluorescein _3mL KI.txt',
        'fluorescein _4mL KI.txt',
    ]
    plot_all_spectra(files_to_plot)
