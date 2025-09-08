import matplotlib.pyplot as plt
import pandas as pd
import io

# Data from your files is loaded here.
# This is a simplified representation of loading your actual files.
# For a real script, you would read each file from your computer.
file_contents = {
    'pure kmno4 G2B1': """
"Wavelength nm.","Abs."
200.00,4.0000
201.00,4.0000
...
""",
    '6Mn+4cr G2B1': """
"Wavelength nm.","Abs."
200.00,3.7974
201.00,3.2997
...
""",
    '2mn+8cr G2B1': """
"Wavelength nm.","Abs."
200.00,2.0050
201.00,1.8664
...
""",
    '4mn+6cr G2B1': """
"Wavelength nm.","Abs."
200.00,2.9055
201.00,2.6299
...
""",
    'Pure k2cr2o7 G2B1': """
"Wavelength nm.","Abs."
200.00,1.7921
201.00,1.7054
...
""",
    '8mn+2cr G2B1': """
"Wavelength nm.","Abs."
200.00,4.0000
201.00,4.0000
...
"""
}

# --- Plotting Code ---

# 1. Set up the plot
plt.figure(figsize=(12, 7))

# 2. Loop through each file and plot its data
# In a real script, you would loop through your actual file paths
# For this example, we loop through the dictionary keys
file_paths = [
    "fwdimaabsorbancedata(1)/pure kmno4 G2B1.txt",
    "fwdimaabsorbancedata(1)/6Mn+4cr G2B1.txt",
    "fwdimaabsorbancedata(1)/2mn+8cr G2B1.txt",
    "fwdimaabsorbancedata(1)/4mn+6cr G2B1.txt",
    "fwdimaabsorbancedata(1)/Pure k2cr2o7 G2B1.txt",
    "fwdimaabsorbancedata(1)/8mn+2cr G2B1.txt"
]

for file_path in file_paths:
    # Read the data, skipping the first descriptive row
    df = pd.read_csv(file_path, skiprows=1)
    
    # Extract a clean name for the legend
    legend_label = file_path.split('/')[-1].replace('.txt', '').replace(' G2B1', '')

    # Plot Wavelength vs. Absorbance
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=legend_label)

# 3. Add labels, title, legend, and grid (just like your example)
plt.title("Absorbance Spectra", fontsize=16)
plt.xlabel("Wavelength (nm)", fontsize=12)
plt.ylabel("Absorbance", fontsize=12)
plt.legend(title="Sample")
plt.grid(True, linestyle='--', alpha=0.6)

# 4. Display the final graph
plt.show()