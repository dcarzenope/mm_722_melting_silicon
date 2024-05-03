import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def process_file(filename):
    # Extract temperature T from the filename
    T = int(filename.split('.')[1])

    # Read the whole file into a list of lines
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Constants
    V = 5 * 5 * 5  # Volume in cubic Ångströms
    k_B = 8.617333262e-5  # Boltzmann constant in eV/K
    dt = 0.001  # Time step in ps

    # Conversion factors
    ev_to_joules = 1.602176634e-19  # 1 eV to J
    ps_to_seconds = 1e-12  # 1 ps to s
    angstrom_to_meters = 1e-10  # 1 Å to m

    # Initialize results
    results = []

    # Determine number of blocks
    num_blocks = len(lines) // 201  # Assuming each block has 200 data lines plus one header line
    plt.figure(figsize=(15, num_blocks * 2))  # Adjust figure size based on number of blocks

    # Iterate through blocks in the file
    num_lines_per_block = 201
    for block_index in range(0, len(lines), num_lines_per_block):
        if block_index + 1 >= len(lines):
            break
        block_data = lines[block_index + 1: block_index + num_lines_per_block]  # Skip the header
        data = np.array([list(map(float, line.split())) for line in block_data])

        # Extract time and heat flux components
        time = data[:, 1]
        heat_flux = data[:, 3:]

        # Remove extreme high values using IQR
        Q3 = np.percentile(heat_flux, 75, axis=0)
        IQR = Q3 - np.percentile(heat_flux, 25, axis=0)
        upper_bound = Q3 + 1.5 * IQR
        valid_heat_flux = heat_flux[np.all(heat_flux <= upper_bound, axis=1)]

        # Calculate autocorrelation for filtered data
        autocorr_values = np.zeros(len(valid_heat_flux))
        for i in range(valid_heat_flux.shape[1]):
            mean_flux = np.mean(valid_heat_flux[:, i])
            autocorr = np.correlate(valid_heat_flux[:, i] - mean_flux, valid_heat_flux[:, i] - mean_flux, mode='full')[len(valid_heat_flux)-1:]
            autocorr_values += autocorr
        autocorr_values /= valid_heat_flux.shape[1]

        # Plot autocorrelation function
        ax = plt.subplot(num_blocks, 1, (block_index // num_lines_per_block) + 1)  # Dynamic subplot layout
        ax.plot(time[:len(autocorr_values)], autocorr_values)
        ax.set_title(f'Block {block_index // num_lines_per_block + 1}')
        ax.grid(True)

        # Compute thermal conductivity
        k = np.trapz(autocorr_values, dx=dt) / (3 * V * k_B * T**2)
        results.append(k)

    plt.tight_layout()
    plt.show()

    # Calculate conversion factor
    conversion_factor = ev_to_joules / (ps_to_seconds * angstrom_to_meters)
    thermal_conductivities = [abs(k) / conversion_factor for k in results]
    return thermal_conductivities

# Process multiple files and collect results
years = range(1675, 1696)  # From 1975 to 1984
all_results = {}
for year in years:
    print(year)
    filename = f'profile.{year}.heatflux'
    all_results[f'{year}'] = process_file(filename)
    print("\n\n\n")

# Display results in a table format
df = pd.DataFrame(all_results)
print(df)

def loadDataFromFile(file_name):
    with open(file_name, 'r') as file:
        return file.readlines()

def parseData(lines):
    time_vals, temperature_vals = [], []
    for line in lines[1:]:
        elements = line.split()
        time_vals.append(float(elements[1]))
        temperature_vals.append(float(elements[2]))
    return time_vals, temperature_vals

file_data_map = {
    "1675": "rdf.1675.out",
    "1676": "rdf.1676.out",
    "1677": "rdf.1677.out",
    "1678": "rdf.1678.out",
    "1679": "rdf.1679.out",
    "1680": "rdf.1680.out",
    "1681": "rdf.1681.out",
    "1682": "rdf.1682.out",
    "1683": "rdf.1683.out",
    "1684": "rdf.1684.out",
    "1685": "rdf.1685.out",
    "1686": "rdf.1686.out",
    "1687": "rdf.1687.out",
    "1688": "rdf.1688.out",
    "1689": "rdf.1689.out",
    "1690": "rdf.1690.out",
    "1691": "rdf.1691.out",
    "1692": "rdf.1692.out",
    "1693": "rdf.1693.out",
    "1694": "rdf.1694.out",
    "1695": "rdf.1695.out"
}

for temperature, filename in file_data_map.items():
    lines = loadDataFromFile(filename)
    times, temps = parseData(lines)
    plt.figure()
    plt.plot(times, temps, label=temperature, linewidth=0.75)
    plt.title(f"Silicon Temperature: {temperature}K")
    plt.xlabel("r/sigma")
    plt.ylabel("g(r)")
    plt.legend(loc="upper right")
    plt.savefig(f"{temperature}.jpeg", format='jpeg')
    plt.close()