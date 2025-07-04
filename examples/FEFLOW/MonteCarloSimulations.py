# Import only the packages necessary to check if FEFLOW and ifm_contrib is present
import sys
import os

# Check the current operating system
if sys.platform == 'win32':  # For Windows use \\, for Linux use /
    sys.path.append('C:\\Program Files\\DHI\\2025\\FEFLOW 10.0\\python')
    os.environ['FEFLOW10_ROOT'] = 'C:\\Program Files\\DHI\\2025\\FEFLOW 10.0\\'
    os.environ['FEFLOW_KERNEL_VERSION'] = '10'  # Modify for your use e.g. FEFLOW 7.4 would be 74
elif sys.platform == 'linux':
    sys.path.append('/opt/feflow/10.0/python/')
    os.environ['FEFLOW10_ROOT'] = '/opt/feflow/10.0/'
    os.environ['FEFLOW_KERNEL_VERSION'] = '10'  # Modify for your use e.g. FEFLOW 7.4 would be 74
else:
    sys.exit("Unsupported operating system.")
# Try to import the ifm package
try:
    from ifm import Enum
    import ifm_contrib as ifm
except ModuleNotFoundError:
    sys.exit("ifm_contrib could not be imported.")
# Check which version of FEFLOW is being used
if ifm.getKernelVersion() < 8000:
    sys.exit("This script is designed for FEFLOW version 8 and higher.")

# Import the rest of the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Define the location of the FEFLOW FEM file
INPUT_FEM = os.path.join(os.path.join(os.getcwd(), "fem"), "MonteCarlo.fem")  # For Windows use \\, for Linux use /
# Define output directory
OUTPUT_DIR = os.path.join(os.path.join(os.getcwd(), "output"), "MonteCarlo")
# Define FEFLOW variables
SELECTION_NAME = "sandstone"
SPECIES = "Copper"
PARAMETER = Enum.P_PORO
TRANSIENT_DURATION = 420  # days
# Create the output dictionary
DICT_OUT = {}
# Define the Monte Carlo variables
AVERAGE = 40.0
STD_DEV = 5.0
NUM_VALUES = 1000
NUM_SIMULATIONS = 5
# Create the Monte Carlo distribution
RNG = np.random.default_rng()
MC_VALUES = RNG.normal(loc=AVERAGE, scale=STD_DEV, size=NUM_VALUES)
# Output a graph of the created distribution
plt.hist(MC_VALUES, 30)
plt.savefig(os.path.join(OUTPUT_DIR, "MC_histogram.png"))


def load_fem_file(input_path: str):
    # Create the variable to load the FEM file
    doc = None
    loaded = False
    # Create a variable to track the number of reconnection attempts
    reconnection_attempts = 0
    # Attempt to load the FEM file
    while not loaded:
        try:
            # Load the FEFLOW model
            doc = ifm.loadDocument(input_path)
            # Set the boolean to true for success
            loaded = True
        except ConnectionError:
            # Print an error message
            print("Failed to establish a connection to the FEFLOW license server.")
            # Check how many reconnection attempts have been made
            if reconnection_attempts > 3:
                sys.exit(
                    f"Failed to establish a connection to the FEFLOW license server after {reconnection_attempts} attempts.")
            else:
                # Wait and retry
                time.sleep(300)
                reconnection_attempts += 1
        except FileNotFoundError:
            # Exit the program
            sys.exit("Failed to find the specified FEFLOW model.")
    return doc


def assign_parameters(doc, iteration_number: int):
    parameter_value = MC_VALUES[iteration_number]
    selection = doc.getSelectionItems(Enum.SEL_ELEMS, SELECTION_NAME)
    # Set the value for specific parameters of the current species
    for selected_element in selection:
        doc.setParamValue((PARAMETER, SPECIES), selected_element, parameter_value)


def get_results(doc, iteration_number: int):
    # Get the history values of the species
    species_id = doc.findSpecies(SPECIES)
    hist_time, hist_values, hist_labels = doc.getHistoryValues(Enum.HIST_LOCA_M, species_id)
    if iteration_number == 0:
        # Add the time column
        DICT_OUT["Time"] = hist_time
    # Loop through the results
    for hist_index in range(len(hist_labels)):
        # Add the data to the dictionary
        DICT_OUT[f"{iteration_number}_{hist_labels[hist_index]}"] = hist_values[hist_index]


def run_simulation(doc, iteration_number: int):
    # Set the solver to direct (steady-state)
    doc.setEquationSolvingType(Enum.EQSOLV_DIRECT)
    # Perform steady-state simulation
    doc.startSimulator()
    doc.stopSimulator()
    # Configure model for transient simulation
    # Set the solver to iterative (transient)
    doc.setEquationSolvingType(Enum.EQSOLV_ITERAT)
    # Set final simulation time
    doc.setFinalSimulationTime(TRANSIENT_DURATION)
    # Set dac file output
    # doc.setOutput(os.path.join(OUTPUT_DIR, f"MonteCarloTransResults_{iteration_number}.dac"), Enum.F_BINARY, time_steps)
    # Perform transient simulation
    doc.startSimulator()
    get_results(doc, iteration_number)
    plot_results(doc, iteration_number)
    doc.stopSimulator()


def plot_results(doc, iteration_number: int):
    # Visualise and inspect the final transient results
    # Create a new matplotlib figure
    fig, ax = plt.subplots(1, figsize=(10, 6))
    # Set equal x and y axis
    plt.axis("equal")
    # Add the mesh (faces and edges)
    doc.c.plot.faces()
    doc.c.plot.edges(alpha=0.1)
    # Add the observation points (If using run all in jupyter, do this afterwards, hangs for some reason)
    doc.c.plot.fringes(par=(Enum.P_CONC, "Copper"), slice=1, alpha=0.5, vmin=0)

    # Add colorbar, title and axis labels
    cbar = plt.colorbar()
    cbar.set_label('Concentration', rotation=90)
    plt.title(f"MC Iteration {iteration_number+1}")
    plt.xlabel("Coord X [m]")
    plt.ylabel("Coord Y [m]")

    plt.savefig(f"C:\\Users\\Anton\\Code\\ArtesiumConsulting\\output\\MC_ts_{iteration_number+1}.png")
    plt.close(fig)


def main():
    for iteration_number in range(NUM_SIMULATIONS):
        print(f"Iteration number {iteration_number+1} started.")
        # Load the FEM file
        doc = load_fem_file(input_path=INPUT_FEM)
        # Apply the parameterisation
        assign_parameters(doc=doc, iteration_number=iteration_number)
        # Run the model
        run_simulation(doc=doc, iteration_number=iteration_number)
        # Close the FEM file
        doc.closeDocument()
        print(f"Iteration number {iteration_number+1} completed.")
    # Save the results
    df_hist = pd.DataFrame(DICT_OUT)
    df_hist.to_excel(os.path.join(OUTPUT_DIR, f"{SPECIES}_MC_Results.xlsx"))


if __name__ == '__main__':
    main()
