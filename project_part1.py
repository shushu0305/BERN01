# %%
#### basic reduced value
import numpy as np
import matplotlib.pyplot as plt
import time 
# %%
#Periodic Boundary Conditions
def apply_pbc_distance(dx, dy, L):
    dx = dx - np.round(dx / L) * L
    dy = dy - np.round(dy / L) * L
    return dx, dy
# energy conservation check tool(maybe every 100000 run for one check)
def total_energy(x, y, L):
    U = 0.0
    n_particles = len(x)
    for i in range(n_particles - 1):
        # Distance from particle i to particles i+1, i+2, ...
        dx = x[i] - x[i + 1:]
        dy = y[i] - y[i + 1:]
        dx, dy = apply_pbc_distance(dx, dy, L)
        r2 = dx**2 + dy**2
        # Hard-core overlap
        if np.any(r2 < sigma0**2):
            return np.inf
        # Count soft-core interactions
        U += np.count_nonzero(
            (r2 >= sigma0**2) &
            (r2 < sigma1**2)
        ) * epsilon
    return U
# %%
def particle_energy(i, x, y, L):
    dx = x[i] - x
    dy = y[i] - y
    dx, dy = apply_pbc_distance(dx, dy, L)
    r2 = dx**2 + dy**2
    # Exclude particle itself
    r2[i] = np.inf
    if np.any(r2 < sigma0**2):
        return np.inf
    U = np.count_nonzero(
        r2 < sigma1**2
    ) * epsilon
    return U
# %%
def mc_move(x, y, L, T_star, max_move):
    # 1. Randomly select one particle
    i = np.random.randint(len(x))
    # Save old position
    old_x = x[i]
    old_y = y[i]
    # 2. Energy before the move
    U_old = particle_energy(i, x, y, L)
    # 3. Make a trial displacement
    x[i] += np.random.uniform(-max_move, max_move)
    y[i] += np.random.uniform(-max_move, max_move)
    # Keep the particle inside the box
    x[i] %= L
    y[i] %= L
    # 4. Energy after the trial move
    U_new = particle_energy(i, x, y, L)
    # Hard-core overlap -> reject immediately
    if np.isinf(U_new):
        x[i] = old_x
        y[i] = old_y
        return False, 0.0
    delta_U = U_new - U_old
    # 5. Metropolis criterion
    if delta_U <= 0:
        return True, delta_U
    if np.random.random() < np.exp(-delta_U / T_star):
        return True, delta_U
    # Reject: restore old coordinates
    x[i] = old_x
    y[i] = old_y
    return False, 0.0

# %%
def radial_distribution(snapshots, L, N, n_bins=100):
    # Maximum distance
    r_max = L / 2
    # Radial bins
    bins = np.linspace(
        0,
        r_max,
        n_bins + 1
    )
    # Centre of each bin
    r_centers = 0.5 * (
        bins[:-1] + bins[1:]
    )
    # Area of each circular shell
    shell_area = np.pi * (
        bins[1:]**2
        - bins[:-1]**2
    )
    # Overall average density
    n_b = N / L**2
    # Store n(r) from every snapshot
    snapshot_densities = []
    for x_snap, y_snap in snapshots:
        tagged_densities = []
        for i in range(N):
            # Distance from particle i
            # to all particles
            dx = x_snap - x_snap[i]
            dy = y_snap - y_snap[i]
            dx, dy = apply_pbc_distance(
                dx,
                dy,
                L
            )
            distances = np.sqrt(
                dx**2 + dy**2
            )
            # Remove the tagged particle itself
            distances = np.delete(
                distances,
                i
            )
            # Number of neighbours in each circular shell
            N_shell, _ = np.histogram(
                distances,
                bins=bins
            )
            # Local density around tagged particle i
            n_i_r = (
                N_shell
                / shell_area
            )
            tagged_densities.append(
                n_i_r
            )
        # Average over all tagged particles
        tagged_densities = np.array(
            tagged_densities
        )
        n_snapshot = np.mean(
            tagged_densities,
            axis=0
        )
        snapshot_densities.append(
            n_snapshot
        )
    # Average over 200 snapshots
    snapshot_densities = np.array(
        snapshot_densities
    )
    mean_n_r = np.mean(
        snapshot_densities,
        axis=0
    )
    g_r = mean_n_r / n_b
    return r_centers, g_r
# %%
# Reduced units
sigma0 = 1.0
sigma1 = 2.5 * sigma0
epsilon = 1.0
# Simulation parameters
N = 100
T_star = 0.1
densities = [0.10, 0.15, 0.227, 0.291, 0.38]
# MC parameters
initial_max_move = 0.15 * sigma0
target_low = 0.28
target_high = 0.32
tune_block = 100000
max_tune_blocks = 100
n_equil = 2*10**6
n_prod = 2*10**7
# For energy-run plot
energy_sample_interval = 10000
# Save exactly 200 production snapshots
n_snapshots = 200
snapshot_interval = n_prod // n_snapshots
print("Snapshot interval =", snapshot_interval)
all_results = {}
# %%
starttime = time.time()
for rho_star in densities:
    good_blocks = 0
    max_move = initial_max_move
    print(f"Running rho* = {rho_star}")
    # 1. Box length
    L = np.sqrt(N * sigma0**2 / rho_star)
    print("Box length L =", L)
    # 2. Initial coordinates (cold start)
    n_side = int(np.sqrt(N))
    spacing = L / n_side
    x = []
    y = []
    for i in range(n_side):
        for j in range(n_side):
            x.append((i + 0.5) * spacing)
            y.append((j + 0.5) * spacing)
    x = np.array(x)
    y = np.array(y)
    print("Number of particles =", len(x))
    U_total = total_energy(x, y, L)
    print("Initial total energy =", U_total)
    # 3. Tune max_move to 0.3 acceptance ratio
    tune_steps = [0]
    tune_energy = [U_total]
    total_tune_steps = 0
    print("\nTuning max_move...")
    for block in range(max_tune_blocks):
        accepted_block = 0
        for step in range(tune_block):
            accepted, delta_U = mc_move(
                x, y, L,
                T_star,
                max_move
            )
            if accepted:
                accepted_block += 1
                U_total += delta_U
            total_tune_steps += 1
            # Record energy during tuning
            if total_tune_steps % energy_sample_interval == 0:
                tune_steps.append(
                    total_tune_steps
                )
                tune_energy.append(
                    U_total
                )
        block_acceptance = (
            accepted_block / tune_block
        )
        print(
            f"Block {block+1}: "
            f"acceptance = {block_acceptance:.3f}, "
            f"max_move = {max_move:.4f}"
        )
        # Acceptance much too high, then increase move strongly
        if block_acceptance > 0.50:
            max_move *= 1.20
            good_blocks = 0
        # Slightly too high
        elif block_acceptance > target_high:
            max_move *= 1.05
            good_blocks = 0
        # Acceptance much too low
        elif block_acceptance < 0.15:
            max_move *= 0.80
            good_blocks = 0
        # Slightly too low
        elif block_acceptance < target_low:
            max_move *= 0.95
            good_blocks = 0
        # Target reached
        else:
            good_blocks += 1
            print("Target acceptance range reached.")
            if good_blocks >= 3:
                break
    # 4. Equilibration
    equil_steps = []
    equil_energy = []
    accepted_equil = 0
    for step in range(n_equil):
        accepted, delta_U = mc_move(
            x, y, L, T_star, max_move
        )
        if accepted:
            accepted_equil += 1
            U_total += delta_U
        if (step+1) % energy_sample_interval == 0:
            equil_steps.append(total_tune_steps+step+1)
            equil_energy.append(U_total)
    equil_acceptance = accepted_equil / n_equil
    print("Equilibration acceptance ratio =",
          equil_acceptance)
    # Energy check after equilibration
    U_check_equil = total_energy(x, y, L)
    print("Tracked energy after equilibration =", U_total)
    print("Recalculated energy after equilibration =", U_check_equil)
    # 5. Production
    prod_steps = []
    prod_energy = []
    energy_samples = []
    energy2_samples = []
    snapshots = []
    accepted_prod = 0
    for step in range(n_prod):
        accepted, delta_U = mc_move(
            x, y, L, T_star, max_move
        )
        if accepted:
            accepted_prod += 1
            U_total += delta_U
        # Record energy for energy-run plot
        if (step + 1) % energy_sample_interval == 0:
            # Add n_equil so production continues
            # after equilibration on x-axis
            prod_steps.append(
                total_tune_steps
                + n_equil
                + step
                + 1
)
            prod_energy.append(U_total)
            energy_samples.append(U_total)
            energy2_samples.append(U_total**2)
        # Save 200 production snapshots
        if (step + 1) % snapshot_interval == 0:
            snapshots.append(
                (
                    x.copy(),
                    y.copy()
                )
            )
    prod_acceptance = accepted_prod / n_prod
    print(
        "Production acceptance ratio =",
        prod_acceptance
    )
    print(
        "Number of saved snapshots =",
        len(snapshots)
    )
    # Energy check after production
    U_check_prod = total_energy(x, y, L)
    # Save configuration exactly at the end of production
    x_prod_final = x.copy()
    y_prod_final = y.copy()
    U_prod_final = U_total
    print("Production tracked energy =", U_total)
    print("Production recalculated energy =", U_check_prod)
    # 6. Energy conservation check
    n_check = 10**7
    check_interval = 10000
    check_steps = []
    check_energy = []
    accepted_check = 0
    for step in range(n_check):
        accepted, delta_U = mc_move(
            x, y, L,
            T_star,
            max_move
        )
        if accepted:
            accepted_check += 1
            U_total += delta_U
        # Record energy
        if (step + 1) % check_interval == 0:

            check_steps.append(
                total_tune_steps
                + n_equil
                + n_prod
                + step
                + 1
            )
            check_energy.append(U_total)
    # 6. Energy vs MC moves
    plt.figure(figsize=(9, 5))
    plt.plot(
        tune_steps,
        tune_energy,
        label="Tuning"
    )
    plt.plot(
        equil_steps,
        equil_energy,
        label="Equilibration"
    )
    plt.plot(
        prod_steps,
        prod_energy,
        label="Production"
    )
    plt.plot(
        check_steps,
        check_energy,
        label="Equilibrium check"
    )
    plt.axvline(
        x=total_tune_steps + n_equil,
        linestyle="--",
        label="Start of production"
    )
    plt.axvline(
        x=total_tune_steps + n_equil + n_prod,
        linestyle="--",
        label="Start of check"
    )
    plt.xlabel("Attempted MC moves")
    plt.ylabel("Total energy U")
    plt.title(
        rf"Energy history: "
        rf"$\rho^*={rho_star},\ T^*={T_star}$"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
    # 7. Final configuration plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x_prod_final, y_prod_final)
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        rf"Final configuration: $\rho^*={rho_star},\ T^*={T_star}$"
    )
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
    # 8. Radial distribution function
    r, g_r = radial_distribution(
        snapshots,
        L,
        N
    )
    plt.figure(figsize=(7, 5))
    plt.plot(r, g_r)
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title(
        rf"$\rho^*={rho_star},\ T^*={T_star}$"
    )
    plt.tight_layout()
    plt.show()
endtime = time.time()
print(
    "\nTotal simulation time =",
    endtime - starttime,
    "seconds"
)
