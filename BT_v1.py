import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from scipy.interpolate import griddata
import time
import matplotlib.patches as patches
import h5py

####################################################################################################################
                                                             #1#
####################################################################################################################
# Toggle to enable or disable logging
ENABLE_LOGGING = True

# Constants
Re, Re_km = 6.371e6, 6.371e3  # Earth's radius in meter and km
include_gc = True
time_step=-1

output_dir = '/content/drive/MyDrive/ColabNotebooks/BT_out/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# If logging is disabled, suppress all logging output
if not ENABLE_LOGGING:
    logging.disable(logging.CRITICAL)

logger = logging.getLogger()

# Define debug log file path
debug_log_path = os.path.join(output_dir, 'particle_tracking_debug.txt')

####################################################################################################################
                                                             #2#
####################################################################################################################

def generate_initial_positions(num_electrons, num_protons, R1, R2, phi1, phi2, min_distance=0.001, enable_min_distance=False):
    # Initialize lists to store positions
    initial_positions_electron = []
    initial_positions_proton = []
    particle_types = []  # This list tracks the type of each particle

    # Generate random positions for electrons
    for _ in range(num_electrons):
        while True:
            # Generate random radial distances between R1 and R2
            radius = np.random.uniform(R1, R2)
            angle = np.random.uniform(phi1, phi2)  # Random angle in left half-plane

            # Calculate the X and Y coordinates
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # Check if the new position satisfies the min_distance condition
            if not enable_min_distance or all(np.sqrt((x - px)**2 + (y - py)**2) > min_distance for px, py in initial_positions_electron):
                initial_positions_electron.append((x, y))
                particle_types.append('electron')  # Track particle type
                break  # Exit the loop when a valid position is found

    # Generate random positions for protons
    for _ in range(num_protons):
        while True:
            radius = np.random.uniform(R1, R2)
            angle = np.random.uniform(phi1, phi2)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # Check if the new position satisfies the min_distance condition
            if not enable_min_distance or all(np.sqrt((x - px)**2 + (y - py)**2) > min_distance for px, py in initial_positions_proton):
                initial_positions_proton.append((x, y))
                particle_types.append('proton')  # Track particle type
                break

    #initial_positions_electron = [(-18, 2), (-10, 5)]#, (3, -5), (3, 6), (-8, -1), (-3, -6.6)]#, (-10, -3)]#, (-10, -3.5), (-10, -4), (-15, -4.5), (-15, -5)]
    #initial_positions_proton = [(-18, -2), (-10, -5)]#, (3, 5), (3, -6), (-8, 1), (-3, 6.6)]#, (-15, 4)]#, (-10, -3.5), (-15, -4), (-10, 4.5), (-10, 5)]
    #initial_positions_electron = [(-10, 2), (-10, 5), (3, -5), (3, 6)]
    #initial_positions_proton = [(-10, -2), (-10, -5), (3, 5), (3, -6)]
    return initial_positions_electron, initial_positions_proton

####################################################################################################################
                                                             #3#
####################################################################################################################

# Function to create and add the circle with a black border
def add_circle(ax, fill_black=False):
    # Draw the circle outline
    circle = patches.Circle((0, 0), 1, edgecolor='white', facecolor='none', linewidth=2)
    GEO_circle = patches.Circle((0, 0), 6.6, edgecolor='white', linestyle='--', facecolor='none', linewidth=2)
    ax.add_patch(circle)
    ax.add_patch(GEO_circle)

    if fill_black:
        # Fill the right (negative X) side with black
        circle_black = patches.Wedge((0, 0), 1, 90, 270, edgecolor='none', facecolor='black')
        ax.add_patch(circle_black)

####################################################################################################################
                                                             #4#
####################################################################################################################

def plot_ellipse_boundary(a1=9.0,a2=-20.0,semiminor=12.0, color='black', linestyle='-', mask=False, plot=True):

    # Parametric equation for the ellipse
    theta = np.linspace(0, 2 * np.pi, 300)
    semimajor=0.5*(a1-a2)
    x0=0.5*(a1+a2)
    y0=0.0
    x_ellipse = x0 + semimajor * np.cos(theta)
    y_ellipse = y0 + semiminor * np.sin(theta)

    # Apply mask if provided
    if mask is not False:
        mask_condition = x_ellipse < 0
        x_ellipse = x_ellipse[mask_condition]
        y_ellipse = y_ellipse[mask_condition]
    if plot:
        plt.plot(x_ellipse, y_ellipse, color=color, linestyle=linestyle)
    # Return the ellipse parameters
    return x0, y0, semimajor, semiminor
####################################################################################################################
                                                             #5#
####################################################################################################################
def load_data():
    file_pathPot = '/content/drive/MyDrive/ColabNotebooks/RCM_out/datapot.dat'
    # Load data while keeping empty lines
    dataPot = pd.read_csv(file_pathPot, sep='\\s+', skiprows=1, names=['XMIN', 'YMIN', 'VM', 'BMIN', 'V', 'PV_gamma', 'Vx_exb', 'Vy_exb'], skip_blank_lines=False)
    logger.info("Data loaded successfully from %s", file_pathPot)
    return dataPot

####################################################################################################################
                                                             #6#
####################################################################################################################

def split_data(dataPot):
    # Find indices where empty lines are present (these rows contain NaNs)
    empty_line_indices = dataPot[dataPot.isna().all(axis=1)].index
    logger.info("Empty line indices: %s", empty_line_indices.tolist())

    # Remove any trailing empty lines at the end of the file
    if empty_line_indices[-1] == len(dataPot) - 1:
        empty_line_indices = empty_line_indices[:-1]

    # Add start and end indices for chunk boundaries
    chunk_boundaries = [0] + empty_line_indices.tolist() + [len(dataPot)]

    # Split data into chunks based on boundaries, removing any NaN-only rows
    chunksPot = [
        dataPot.iloc[chunk_boundaries[i] + 1:chunk_boundaries[i + 1]].dropna(how='all').reset_index(drop=True)
        for i in range(len(chunk_boundaries) - 1)
    ]

    # Print the number of chunks detected
    logger.info("Number of chunks detected: %d", len(chunksPot))

    return chunksPot
####################################################################################################################
                                                             #7#
####################################################################################################################
def interpolate_fields(chunksPot, grid_resolution=100):

    # Determine min and max values of XMIN and YMIN across all chunks
    all_X = pd.concat([chunk['XMIN'] for chunk in chunksPot])
    all_Y = pd.concat([chunk['YMIN'] for chunk in chunksPot])
    grid_x_min, grid_x_max = all_X.min(), all_X.max()
    grid_y_min, grid_y_max = all_Y.min(), all_Y.max()

    # Set up the common grid based on min and max values of XMIN and YMIN
    x_values = np.linspace(grid_x_min, grid_x_max, grid_resolution)
    y_values = np.linspace(grid_y_min, grid_y_max, grid_resolution)
    common_grid_x, common_grid_y = np.meshgrid(x_values, y_values)

    # # Diagnostic check for the shapes of common_grid_x and common_grid_y
    # print("Shape of common_grid_x:", common_grid_x.shape)  # Expected: (250, 250)
    # print("Shape of common_grid_y:", common_grid_y.shape)  # Expected: (250, 250)

    precomputed_interpolated_fields = []

    for chunk_index, chunk in enumerate(chunksPot):
        # Extract original grid points and field values
        points = chunk[['XMIN', 'YMIN']].values
        VM_interp = griddata(points, chunk['VM'].values, (common_grid_x, common_grid_y), method='linear', fill_value=np.nan)
        BMIN_interp = griddata(points, chunk['BMIN'].values, (common_grid_x, common_grid_y), method='linear', fill_value=np.nan)
        V_interp = griddata(points, chunk['V'].values, (common_grid_x, common_grid_y), method='linear', fill_value=np.nan)
        PV_gamma_interp = griddata(points, chunk['PV_gamma'].values, (common_grid_x, common_grid_y), method='linear', fill_value=np.nan)
        Vx_exb_interp = griddata(points, chunk['Vx_exb'].values, (common_grid_x, common_grid_y), method='linear', fill_value=np.nan)
        Vy_exb_interp = griddata(points, chunk['Vy_exb'].values, (common_grid_x, common_grid_y), method='linear', fill_value=np.nan)

        precomputed_interpolated_fields.append({
           'VM': VM_interp,
           'BMIN': BMIN_interp,
           'V': V_interp,
           'PV_gamma': PV_gamma_interp,
           'Vx_exb': Vx_exb_interp,
           'Vy_exb': Vy_exb_interp,
       })

    return precomputed_interpolated_fields, x_values, y_values

####################################################################################################################
                                                             #8#
####################################################################################################################

def calculate_vtot(precomputed_interpolated_fields, landa):

        global include_gc
        total_velocities = []

        for fields in precomputed_interpolated_fields:
            BMIN_interp = fields ['BMIN']
            Vx_exb_interp = fields['Vx_exb']
            Vy_exb_interp = fields['Vy_exb']
            VM_interp = fields['VM']

            VM_interp_safe = np.where((VM_interp <= 0) | np.isnan(VM_interp), np.nan, VM_interp)
            V_power = np.where(np.isnan(VM_interp_safe), np.nan, VM_interp_safe ** (-2/3))

            dV_dx = np.gradient(V_power, axis=1)
            dV_dy = np.gradient(V_power, axis=0)

            # BMIN conversion and GC factor calculation
            valid_BMIN_mask = BMIN_interp > 0
            BMIN_interp_converted = np.where(valid_BMIN_mask, BMIN_interp * 1.0e-9, np.nan)

            GC_factor_electron = np.zeros_like(BMIN_interp_converted)
            GC_factor_proton = np.zeros_like(BMIN_interp_converted)

            if include_gc:
                GC_factor_electron[valid_BMIN_mask] = -landa / (-1 * BMIN_interp_converted[valid_BMIN_mask])
                GC_factor_proton[valid_BMIN_mask] = -landa / (1 * BMIN_interp_converted[valid_BMIN_mask])

            # Calculate GC velocities
            Vx_gc_electron = np.where(valid_BMIN_mask, -GC_factor_electron * dV_dy, np.nan) * (60.0 / (Re * Re))
            Vy_gc_electron = np.where(valid_BMIN_mask, GC_factor_electron * dV_dx, np.nan) * (60.0 / (Re * Re))
            Vx_gc_proton = np.where(valid_BMIN_mask, -GC_factor_proton * dV_dy, np.nan) * (60.0 / (Re * Re))
            Vy_gc_proton = np.where(valid_BMIN_mask, GC_factor_proton * dV_dx, np.nan) * (60.0 / (Re * Re))

            # Calculate total velocities
            Vx_tot_electron = Vx_gc_electron + Vx_exb_interp * (60.0 / Re_km)
            Vy_tot_electron = Vy_gc_electron + Vy_exb_interp * (60.0 / Re_km)
            Vx_tot_proton = Vx_gc_proton + Vx_exb_interp * (60.0 / Re_km)
            Vy_tot_proton = Vy_gc_proton + Vy_exb_interp * (60.0 / Re_km)

            # Append results
            total_velocities.append({
            'Vx_tot_e': Vx_tot_electron,
            'Vy_tot_e': Vy_tot_electron,
            'Vx_tot_p': Vx_tot_proton,
            'Vy_tot_p': Vy_tot_proton,
            })

        return total_velocities

####################################################################################################################
                                                             #9#
####################################################################################################################

def rk4_step(position, field_parameters):
    """
    Perform a single Runge-Kutta 4th order (RK4) step to advance the particle's position.

    Args:
        position (tuple): Current (x, y) position of the particle.
        field_parameters (dict): Contains Vx and Vy field values for the particle.
        time_step (float): Time step for integration (negative for backward in time).

    Returns:
        tuple: New (x, y) position after one RK4 step.
    """
    global time_step
    x, y = position

    def velocity_at(pos):
       """Retrieve velocity components at a given position using grid-based interpolation or indexing."""
       return field_parameters["Vx"], field_parameters["Vy"]

    # RK4 integration
    k1_x, k1_y = velocity_at((x, y))
    k1_x *= time_step
    k1_y *= time_step

    k2_x, k2_y = velocity_at((x + 0.5 * k1_x, y + 0.5 * k1_y))
    k2_x *= time_step
    k2_y *= time_step

    k3_x, k3_y = velocity_at((x + 0.5 * k2_x, y + 0.5 * k2_y))
    k3_x *= time_step
    k3_y *= time_step

    k4_x, k4_y = velocity_at((x + k3_x, y + k3_y))
    k4_x *= time_step
    k4_y *= time_step

    # Update position
    new_x = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
    new_y = y + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6

    return new_x, new_y


####################################################################################################################
                                                             #10#
####################################################################################################################

def advance_particle(particle, is_electron, interpolated_fields, landa,
                     x_values, y_values):
    global include_gc
    global time_step
    total_velocities = calculate_vtot(interpolated_fields, landa)

    # Retrieve ellipse parameters using plot_ellipse_boundary
    x0, y0, semimajor, semiminor = plot_ellipse_boundary(a1=9.0,a2=-20.0,semiminor=12.0, plot=False)

    pos_x, pos_y = particle
    path = [(pos_x, pos_y, 0, 0)]  # Track particle positions over time

    # Redefine is_inside_boundary using returned parameters
    def is_inside_boundary(x, y, tolerance=1e-6):
        ellipse_value = ((x - x0) ** 2) / semimajor ** 2 + ((y - y0) ** 2) / semiminor ** 2
        #print(f"Checking boundary for ({x}, {y}): value={ellipse_value}, inside={ellipse_value <= 1 + tolerance}")
        #print(f"Parameters: x0={x0}, semimajor={semimajor}, semiminor={semiminor}")
        return ellipse_value <= 1 + tolerance


    def get_grid_indices(pos, x_values, y_values):
        x, y = pos
        ix = np.abs(x_values - x).argmin()
        iy = np.abs(y_values - y).argmin()
        #print(f"Grid indices for position ({pos_x}, {pos_y}): ix={ix}, iy={iy}")
        return ix, iy

    def stop_at_boundary(old_pos, new_pos):
        """ Interpolates between two points to place the particle at the boundary. """
        x1, y1 = old_pos
        x2, y2 = new_pos

        # Calculate the interpolation factor to the boundary
        factor = 0.0
        while factor < 1.0:
            x_test = x1 + factor * (x2 - x1)
            y_test = y1 + factor * (y2 - y1)
            is_inside = is_inside_boundary(x_test, y_test)
            #print(f"Interpolating: ({x_test}, {y_test}), inside={is_inside}")
            if not is_inside:
                return x1 + (factor + 0.01) * (x2 - x1), y1 + (factor + 0.01) * (y2 - y1)
            factor += 0.001
        return new_pos

    stopped = False  # Flag to track if particle has exited boundary and is stopped

    for chunk, velocities in zip(reversed(interpolated_fields), reversed(total_velocities)):
        # Initialize vx and vy to defaults for safety
        # vx, vy = 0, 0
        if stopped:
            # If particle is stopped, append the current position without advancing
            path.append((pos_x, pos_y, 0, 0))
            continue

        # Check if the particle is outside the boundary
        if not is_inside_boundary(pos_x, pos_y):
            # Adjust the particle position to be slightly outside the boundary
            new_x, new_y = stop_at_boundary((pos_x, pos_y), (pos_x, pos_y))
            stopped = True  # Mark as stopped

            # Manually set vx and vy to zero for this stopped particle
            vx, vy = 0, 0
            path.append((new_x, new_y, vx, vy))
            continue

        ix, iy = get_grid_indices((pos_x, pos_y), x_values, y_values)

        # Assign velocity field based on particle type
        if is_electron:
            vx = velocities['Vx_tot_e'][iy, ix]
            vy = velocities['Vy_tot_e'][iy, ix]

            #print(f"Electron at ({pos_x}, {pos_y}), ix={ix}, iy={iy}")
            #print(f"Retrieved velocities: Vx={vx}, Vy={vy}")

        else:
            vx = velocities['Vx_tot_p'][iy, ix]
            vy = velocities['Vy_tot_p'][iy, ix]

            #print(f"Proton at ({pos_x}, {pos_y}), ix={ix}, iy={iy}")
            #print(f"Retrieved velocities: Vx={vx}, Vy={vy}")

        new_x, new_y = rk4_step((pos_x, pos_y),  {"Vx": vx, "Vy": vy})

        # Check if new position is outside boundary and adjust to boundary if so
        if not is_inside_boundary(new_x, new_y):
            new_x, new_y = stop_at_boundary((pos_x, pos_y), (new_x, new_y))
            stopped = True  # Stop further movement once it reaches the boundary
            # Manually set vx and vy to zero for this stopped particle
            vx, vy = 0, 0

        pos_x, pos_y = new_x, new_y
        path.append((pos_x, pos_y, vx, vy))

    return path

####################################################################################################################
                                                             #11#
####################################################################################################################

def process_particles(particles, is_electron, interpolated_fields, landa_values, x_values, y_values):
    paths = []
    global time_step
    for idx, particle in enumerate(particles):
        landa = landa_values[idx]  # Get the corresponding `landa` value for the particle
        path = advance_particle(particle, is_electron, interpolated_fields, landa, x_values, y_values)
        paths.append(path)
    return paths

####################################################################################################################
                                                             #12#
####################################################################################################################

def plot_circle(ax, X00, Y00, rad00, color='white', linestyle='--', linewidth=1.5):
    """
    Plots a circle representing the region of interest on the given axes.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot the circle on.
        X0, Y0 (float): The x and y coordinates of the circle center.
        rad (float): Radius of the circle.
        color (str): Color of the circle boundary.
        linestyle (str): Style of the circle boundary.
        linewidth (float): Width of the circle boundary line.
    """
    # Circle definition
    theta = np.linspace(0, 2 * np.pi, 300)
    x_circle = X00 + rad00 * np.cos(theta)
    y_circle = Y00 + rad00 * np.sin(theta)

    # Plot the circle on the provided axes
    ax.plot(x_circle, y_circle, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.set_aspect('equal', adjustable='box')  # Ensures the aspect ratio is equal for better visualization

####################################################################################################################
                                                             #13#
####################################################################################################################

def plot_PV_gamma_map_with_particles(PV_gamma_grid, grid_x, grid_y, chunk_index, output_dir, electron_positions,
                                     proton_positions, interpolated_fields, electron_velocities_current, proton_velocities_current, x_values, y_values, subset_idx=None,
                                     selected_idx=None, num_electrons=None, num_protons=None, landa_e=None, landa_p=None):
    """
    Plots the PV_gamma values as a color map and marks the positions of a massless electron and proton at each chunk.
    Includes a color bar for `landa` values of electrons and protons.

    Args:
        PV_gamma_grid (2D array): Grid of PV_gamma values to plot.
        grid_x (2D array): X-coordinates for the common grid.
        grid_y (2D array): Y-coordinates for the common grid.
        chunk_index (int): Index of the current chunk to label the plot.
        output_dir (str): Directory to save the plot files.
        electron_positions (list of tuples): Current positions of electrons at this chunk.
        proton_positions (list of tuples): Current positions of protons at this chunk.
        interpolated_fields (list of dicts): Interpolated field values for all chunks.
        electron_velocities_current (list of dicts): Current velocities for each electron at this chunk.
        proton_velocities_current (list of dicts): Current velocities for each proton at this chunk.
        x_values (array): X-axis values of the common grid.
        y_values (array): Y-axis values of the common grid.
        selected_idx (int, optional): Index of the particle to highlight.
        subset_idx (int, optional): Index of the subset for labeling purposes.
        num_electrons (int, optional): Total number of electrons.
        num_protons (int, optional): Total number of protons.
        landa_e (list, optional): `landa` values for electrons.
        landa_p (list, optional): `landa` values for protons.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define the color scale limits for PV_gamma and plot as a heatmap
    min_value, max_value = 5e-3, 3e-1
    heatmap = ax.pcolormesh(grid_x, grid_y, PV_gamma_grid, shading='auto', cmap='binary',
                            norm=mcolors.LogNorm(vmin=min_value, vmax=max_value))
    cbar = plt.colorbar(heatmap, ax=ax, shrink=0.7, aspect=15)
    cbar.set_label('$PV^{5/3}$ (nPa $(Re/nT)^{{\\frac{{5}}{{3}}}}$)')

    # Calculate time label (T=hh:mm format)
    hours, minutes = divmod(chunk_index, 60)
    time_label = f"T={hours:02}:{minutes:02}"

    # Retrieve ellipse parameters
    x0, y0, semimajor, semiminor = plot_ellipse_boundary(a1=9.0,a2=-20.0,semiminor=12.0, plot=False)

    # Helper function to check boundary
    def is_inside_boundary(x, y):
        ellipse_value = ((x - x0) ** 2) / semimajor ** 2 + ((y - y0) ** 2) / semiminor ** 2
        return ellipse_value <= 1

    # Initialize labels for electron and proton field properties
    properties_labele = ""
    properties_labelp = ""

    # Scatter plot color mapping
    all_landa = np.concatenate([landa_e, landa_p])  # Combine electron and proton `landa` for color bar
    cmap = plt.cm.rainbow
    #norm = plt.Normalize(vmin=np.min(all_landa), vmax=np.max(all_landa))  # Normalize based on all `landa`
    norm=mcolors.LogNorm(vmin=np.min(all_landa), vmax=np.max(all_landa))

    # Add a separate color bar for the scatter plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # Create a ScalarMappable for the scatter plot
    sm.set_array([])  # Dummy array for the color bar
    scatter_cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7, aspect=15, pad=0.05)
    scatter_cbar.set_label('λ (eV $(Re/nT)^{{\\frac{{2}}{{3}}}}$)')  # Label for the scatter color bar

    # Helper function to get the grid index
    def get_grid_indices(pos, x_values, y_values):
        x, y = pos
        ix = np.abs(x_values - x).argmin()
        iy = np.abs(y_values - y).argmin()
        return ix, iy

    # Set plot title to include time label, number of electrons and protons, and `landa` value
    plt.title(f'$PV^{{\\frac{{5}}{{3}}}}$ Map - {time_label} | Electrons: {num_electrons}, Protons: {num_protons}')

    # Loop over all electron positions
    for i, (pos_x, pos_y) in enumerate(electron_positions):
        #color = cmap(norm(landa_e[i]))  # Map `landa` value to a color
        inside = is_inside_boundary(pos_x, pos_y)
        color = cmap(norm(landa_e[i])) if inside else "gray"  # Decolorize if outside boundary
        marker_size = 30 if inside else 10  # Reduce size if outside boundary
        velocities = electron_velocities_current[i]  # Get the velocity for this electron
        ax.scatter(pos_x, pos_y, color=color, marker='<', s=marker_size)
        if subset_idx is not None and selected_idx is not None and i == selected_idx:  # Only label and add legend for selected electron
            ix, iy = get_grid_indices((pos_x, pos_y), x_values, y_values)
            field_params = interpolated_fields[chunk_index]
            landa = landa_range[subset_idx]
            properties_labele = (
                f"Electron {i} | λ={landa:.2e}\n"
                f"X={pos_x:.2f} Re, Y={pos_y:.2f} Re, PV_gamma={field_params['PV_gamma'][iy, ix]:.4f}\n"
                f"BMIN={field_params['BMIN'][iy, ix]:.4f} nT, V={field_params['V'][iy, ix]:.4f} Volt\n"
                f"Vxtot={velocities['Vx_tot_e'][iy, ix]:.4f} Re/min, Vytot={velocities['Vy_tot_e'][iy, ix]:.4f} Re/min"
            )
            ax.scatter(pos_x, pos_y, color=color, marker='<', label=f'Electron {i}')  # Only add label for legend
#        else:
#            ax.scatter(pos_x, pos_y, color=color, marker='<', marker_size=marker_size)

    # Loop over all proton positions
    for i, (pos_x, pos_y) in enumerate(proton_positions):
        #color = cmap(norm(landa_p[i]))  # Map `landa` value to a color
        inside = is_inside_boundary(pos_x, pos_y)
        color = cmap(norm(landa_p[i])) if inside else "gray"  # Decolorize if outside boundary
        marker_size = 30 if inside else 10  # Reduce size if outside boundary
        velocities = proton_velocities_current[i]  # Get the velocity for this proton
        ax.scatter(pos_x, pos_y, color=color, marker='o', s=marker_size)
        if subset_idx is not None and selected_idx is not None and i == selected_idx:  # Only label and add legend for selected proton
            ix, iy = get_grid_indices((pos_x, pos_y), x_values, y_values)
            field_params = interpolated_fields[chunk_index]
            landa = landa_range[subset_idx]
            properties_labelp = (
                f"Proton {i} | λ={landa:.2e}\n"
                f"X={pos_x:.2f} Re, Y={pos_y:.2f} Re, PV_gamma={field_params['PV_gamma'][iy, ix]:.4f}\n"
                f"BMIN={field_params['BMIN'][iy, ix]:.4f} nT, V={field_params['V'][iy, ix]:.4f} Volt\n"
                f"Vxtot={velocities['Vx_tot_p'][iy, ix]:.4f} Re/min, Vytot={velocities['Vy_tot_p'][iy, ix]:.4f} Re/min"
            )
            ax.scatter(pos_x, pos_y, color=color, marker='o', label=f'Proton {i}')  # Only add label for legend
#        else:
#            ax.scatter(pos_x, pos_y, color=color, marker='o')

    # Add the labels to the plot using fig.text
    box_y_pos = 0.03  # Adjust vertical position as needed
    fig.text(0.055, box_y_pos + 0.08, properties_labele, ha='left', fontsize=12, color='black', verticalalignment='top', transform=fig.transFigure)
    fig.text(0.505, box_y_pos + 0.08, properties_labelp, ha='left', fontsize=12, color='black', verticalalignment='top', transform=fig.transFigure)

    # Set plot labels and title
    plt.xlim([-25, 10][::-1])
    plt.ylim([-15, 15][::-1])
    plt.xlabel('X (Re)')
    plt.ylabel('Y (Re)')
    legend = ax.legend(loc='upper right')  # Set size and weight if needed
    plt.setp(legend.get_texts(), color='white')  # Change 'blue' to your desired color
    #legend.get_frame().set_facecolor('green')  # Background color
    legend.get_frame().set_visible(False)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.title(f'PV_gamma Map - Chunk {chunk_index}')

    #mark_segments(ax, num_segments=5)

    # Add the circle with the black half
    add_circle(ax, fill_black=True)

    plot_ellipse_boundary(a1=9.0,a2=-20.0,semiminor=12.0)

    # Save the plot as a PNG file in the specified output directory
    output_file = os.path.join(output_dir, f'map_{chunk_index}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    logger.info(f"PV_gamma map saved for chunk {chunk_index} at {output_file}")

####################################################################################################################
                                                             #14#
####################################################################################################################
if __name__ == "__main__":
    # Load data, interpolate fields, and define common grid
    # Generate initial positions for electrons and protons
    initial_positions_electron, initial_positions_proton = (
    generate_initial_positions(num_electrons=1000,num_protons=1000,R1=6.6,R2=9.0, phi1=0., phi2=2.0*np.pi)
    )

    num_electrons = len(initial_positions_electron)
    num_protons = len(initial_positions_proton)

    # Divide particles into subsets
    landa_min, landa_max, num_subsets = 1e3, 1e5, 5
    landa_range = np.linspace(landa_min, landa_max, num_subsets)
    electron_subsets = np.array_split(initial_positions_electron, num_subsets)
    proton_subsets = np.array_split(initial_positions_proton, num_subsets)

    # Assign landa values to subsets
    landa_values_electron = [landa for landa, subset in zip(landa_range, electron_subsets) for _ in subset]
    landa_values_proton = [landa for landa, subset in zip(landa_range, proton_subsets) for _ in subset]

    print("Starting data load and interpolation...")  # Notify start of data loading
    dataPot = load_data()
    chunksPot = split_data(dataPot)
    precomputed_interpolated_fields, x_values, y_values = interpolate_fields(chunksPot, grid_resolution=100)
    # Step 2: Define multiprocessing parameters
    num_workers = 1  # Get number of CPU cores
    electron_split = np.array_split(initial_positions_electron, num_workers)
    proton_split = np.array_split(initial_positions_proton, num_workers)

    start_time = time.time()

    #electron_results = process_particles(electron_split[0], True, precomputed_interpolated_fields, landa_value, x_values, y_values)
    #proton_results = process_particles(proton_split[0], False, precomputed_interpolated_fields, landa_value, x_values, y_values)

    # Process particles with individual landa values
    electron_results = process_particles(electron_split[0], True, precomputed_interpolated_fields,
                                      landa_values_electron, x_values, y_values)
    proton_results = process_particles(proton_split[0], False, precomputed_interpolated_fields,
                                     landa_values_proton, x_values, y_values)

    # Combine paths from all processes, ensuring each path is a separate list of positions
    electron_paths = [path for path in electron_results]  # Directly use electron_results
    proton_paths = [path for path in proton_results]      # Directly use proton_results

    # Define the output file for saving tracking results
    output_file_path = os.path.join(output_dir, "tracking_results.h5")

    # Open the output file and write the results for each chunk
    with h5py.File(output_file_path, "w") as h5file:

        # Create dictionaries to store velocities for each subset
        electron_velocities = {}
        proton_velocities = {}

        # Calculate velocities for each subset of electrons
        for subset_idx, landa in enumerate(landa_range):
            electron_velocities[subset_idx] = calculate_vtot(precomputed_interpolated_fields, landa)

        # Calculate velocities for each subset of protons
        for subset_idx, landa in enumerate(landa_range):
            proton_velocities[subset_idx] = calculate_vtot(precomputed_interpolated_fields, landa)

        # Plot each chunk's PV_gamma map with electron and proton positions
        for chunk_index, chunk in enumerate(precomputed_interpolated_fields):
            #file.write(f"Chunk#{chunk_index}           x           y            vx           vy\n\n")
            print(f"Now processing chunk {chunk_index} in plot_PV_gamma_map_with_particles")  # Print chunk being processed
            # Create groups for each chunk
            chunk_group = h5file.create_group(f"Chunk_{chunk_index}")
            PV_gamma_grid = chunk['PV_gamma']

            # Extract the current positions and velocities for each particle at the current chunk
            try:
                electron_positions_current = [
                    (electron_paths[p][len(precomputed_interpolated_fields) - 1 - chunk_index][:2]) for p in range(len(electron_paths))
                    ]
                proton_positions_current = [
                    (proton_paths[p][len(precomputed_interpolated_fields) - 1 - chunk_index][:2]) for p in range(len(proton_paths))
                    ]
            except IndexError as e:
                print(f"Error accessing paths at chunk {chunk_index}: {e}")
                continue

            # Assign velocities based on particle subset
            electron_velocities_current = []
            for p in range(len(electron_paths)):
                initial_position = electron_paths[p][0][:2]  # Extract the initial (x, y) position
                subset_idx = next(idx for idx, subset in enumerate(electron_subsets) if initial_position in subset)
                electron_velocities_current.append(electron_velocities[subset_idx][chunk_index])

            proton_velocities_current = []
            for p in range(len(proton_paths)):
                initial_position = proton_paths[p][0][:2]  # Extract the initial (x, y) position
                subset_idx = next(idx for idx, subset in enumerate(proton_subsets) if initial_position in subset)
                proton_velocities_current.append(proton_velocities[subset_idx][chunk_index])

            # Ensure these are lists of tuples for the plotting function
            if isinstance(electron_positions_current, tuple):
                electron_positions_current = [electron_positions_current]
            if isinstance(proton_positions_current, tuple):
                proton_positions_current = [proton_positions_current]

            # Pass only the positions relevant to this chunk to the plotting function
            plot_PV_gamma_map_with_particles(PV_gamma_grid, np.meshgrid(x_values, y_values)[0], np.meshgrid(x_values, y_values)[1],
                chunk_index, output_dir, electron_positions_current, proton_positions_current,
                precomputed_interpolated_fields, electron_velocities_current, proton_velocities_current, x_values, y_values, subset_idx = 0,
                selected_idx=0, num_electrons=num_electrons, num_protons=num_protons, landa_e=landa_values_electron, landa_p=landa_values_proton)

            # Prepare electron data for saving
            electron_data = []
            for p, electron in enumerate(electron_paths):
                pos_x, pos_y = electron[len(precomputed_interpolated_fields) - 1 - chunk_index][:2]
                initial_position = electron[0][:2]  # Get the initial position of the particle
                #subset_idx = next(idx for idx, subset in enumerate(electron_subsets) if initial_position in subset)
                #landa = landa_range[idx]  # Retrieve the corresponding `landa` value
                # Determine the subset index for the particle
                # Find the correct subset index for the particle
                subset_idx = -1
                for idx, subset in enumerate(electron_subsets):
                    if any(np.allclose(initial_position, point, atol=1e-6) for point in subset):
                        subset_idx = idx
                        break
                if subset_idx == -1:
                    raise ValueError(f"Initial position {initial_position} not found in any subset for electrons.")

                landa = landa_range[subset_idx]  # Retrieve the corresponding `landa` value


                # Get grid indices for the particle's position
                ix, iy = np.abs(x_values - pos_x).argmin(), np.abs(y_values - pos_y).argmin()

                # Extract velocities from the current velocity fields
                velocities = electron_velocities_current[p]
                vx, vy = velocities['Vx_tot_e'][iy, ix], velocities['Vy_tot_e'][iy, ix]

                # Append data
                electron_data.append([pos_x, pos_y, vx, vy, landa])

            electron_data = np.array(electron_data)  # Convert to numpy array for HDF5 storage

            # Convert to numpy array with float32 for smaller file size
            electron_data = np.array(electron_data, dtype=np.float32)

            # Save electron data in a dataset under the chunk group
            chunk_group.create_dataset("electrons", data=electron_data, compression="gzip")

            # Prepare proton data for saving
            proton_data = []
            for p, proton in enumerate(proton_paths):
                pos_x, pos_y = proton[len(precomputed_interpolated_fields) - 1 - chunk_index][:2]
                initial_position = proton[0][:2]
                #subset_idx = next(idx for idx, subset in enumerate(proton_subsets) if initial_position in subset)
                # Determine the subset index for the particle
                # Find the correct subset index for the particle
                subset_idx = -1
                for idx, subset in enumerate(proton_subsets):
                    if any(np.allclose(initial_position, point, atol=1e-6) for point in subset):
                        subset_idx = idx
                        break
                if subset_idx == -1:
                    raise ValueError(f"Initial position {initial_position} not found in any subset for protons.")

                landa = landa_range[subset_idx]  # Retrieve the corresponding `landa` value

                # Get grid indices for the particle's position
                ix, iy = np.abs(x_values - pos_x).argmin(), np.abs(y_values - pos_y).argmin()

                # Extract velocities from the current velocity fields
                velocities = proton_velocities_current[p]
                vx, vy = velocities['Vx_tot_p'][iy, ix], velocities['Vy_tot_p'][iy, ix]

                # Append data
                proton_data.append([pos_x, pos_y, vx, vy, landa])

            proton_data = np.array(proton_data)  # Convert to numpy array for HDF5 storage

            # Convert to numpy array with float32 for smaller file size
            proton_data = np.array(proton_data, dtype=np.float32)

            # Save proton data in a dataset under the chunk group
            chunk_group.create_dataset("protons", data=proton_data, compression="gzip")

    print(f"Tracking results saved to {output_file_path}")

# Calculate and print the total runtime
end_time = time.time()
total_time = end_time - start_time
print(f"Total computation time: {total_time:.2f} seconds")
logger.info(f"Total computation time: {total_time:.2f} seconds")
