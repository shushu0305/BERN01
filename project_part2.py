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
def tune_max_move(
    x, y, L,
    T_star,
    max_move,
    U_total,
    target=0.30,
    tolerance=0.02,
    block_size=20000,
    max_blocks=30
):
    good_blocks = 0
    for block in range(max_blocks):
        accepted_block = 0
        for step in range(block_size):
            accepted, delta_U = mc_move(
                x, y, L,
                T_star,
                max_move
            )
            if accepted:
                accepted_block += 1
                U_total += delta_U
        block_acceptance = (
            accepted_block / block_size
        )
        print(
            f"T*={T_star:.2f}, "
            f"block={block+1}, "
            f"acceptance={block_acceptance:.3f}, "
            f"max_move={max_move:.4f}"
        )
        # Acceptance too high:
        if block_acceptance > target + tolerance:
            max_move *= 1.05
            good_blocks = 0
        # Acceptance too low:
        elif block_acceptance < target - tolerance:
            max_move *= 0.95
            good_blocks = 0
        # Within 28%-32%
        else:
            good_blocks += 1
            if good_blocks >= 2:
                break
    return max_move, U_total
# %%
# Gradual cooling simulation
# Reduced units
sigma0 = 1.0
sigma1 = 2.5 * sigma0
epsilon = 1.0
# Simulation parameters
N = 100
rho_star = 0.291
# MC parameters
max_move = 0.30 * sigma0
temperatures = np.round(
    np.arange(0.25, 0.14, -0.01),
    2
)
print(temperatures)
# Box length
L = np.sqrt(N * sigma0**2 / rho_star)
# Simulation length
n_equil = 2*10**6
n_prod = 2*10**7
sample_interval = 1000
# %%
# Initial cold-start configuration at T*=0.25
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
U_total = total_energy(x, y, L)
print("Number of particles =", len(x))
print("Initial total energy =", U_total)
# %%
temperature_results = []
final_configurations = {}
energy_histories = {}
# %%
import time
starttime = time.time()
for T_star in temperatures:
    print("T* =", T_star)
    # 0. Tune max_move
    max_move, U_total = tune_max_move(
        x=x,
        y=y,
        L=L,
        T_star=T_star,
        max_move=max_move,
        U_total=U_total
    )
    print(
        "Tuned max_move =",
        max_move
    )
    print(
        "Corresponding Delta =",
        2 * max_move
    )
    # 1. Equilibration
    accepted_equil = 0
    for step in range(n_equil):
        accepted, delta_U = mc_move(
            x, y, L,
            T_star,
            max_move
        )
        if accepted:
            accepted_equil += 1
            U_total += delta_U
    equil_acceptance = accepted_equil / n_equil
    print(
        "Equilibration acceptance ratio =",
        equil_acceptance
    )

    # 2. Production
    energy_samples = []
    accepted_prod = 0
    for step in range(n_prod):
        accepted, delta_U = mc_move(
            x, y, L,
            T_star,
            max_move
        )
        if accepted:
            accepted_prod += 1
            U_total += delta_U
        if step % sample_interval == 0:
            energy_samples.append(U_total)
    prod_acceptance = accepted_prod / n_prod
    energy_samples = np.array(energy_samples)
    # 3. Calculate averages
    mean_U = np.mean(energy_samples)
    mean_U2 = np.mean(
        energy_samples**2
    )
    energy_fluctuation = (
        mean_U2 - mean_U**2
    )
    print("Production acceptance ratio =",
          prod_acceptance)
    print("<U> =", mean_U)
    print("<U^2> =", mean_U2)
    print(
        "<U^2> - <U>^2 =",
        energy_fluctuation
    )
    # 4. Save results
    temperature_results.append(
        [
            T_star,
            mean_U,
            mean_U2,
            energy_fluctuation,
            equil_acceptance,
            prod_acceptance
        ]
    )
    # Save final coordinates at this temperature
    final_configurations[T_star] = (
        x.copy(),
        y.copy()
    )
    # Sampled production energies
    energy_histories[T_star] = (
        energy_samples.copy()
    )
endtime = time.time()
print(
    "\nTotal cooling simulation time =",
    endtime - starttime,
    "seconds"
)
# %%
temperature_results = np.array(
    temperature_results
)
T_values = temperature_results[:, 0]
mean_U_values = temperature_results[:, 1]
mean_U2_values = temperature_results[:, 2]
fluctuation_values = temperature_results[:, 3]
equil_acceptance_values = temperature_results[:, 4]
prod_acceptance_values = temperature_results[:, 5]
# %%
# Sort temperature from low to high
order = np.argsort(T_values)
T_plot = T_values[order]
U_plot = mean_U_values[order]
fluctuation_plot = fluctuation_values[order]
# %%
# Energy fluctuation
plt.figure(figsize=(7, 5))
plt.scatter(
    T_plot,
    fluctuation_plot,
    marker="o"
)
plt.xlabel(r"$T^*$")
plt.ylabel(r"$\langle U^2\rangle - \langle U\rangle^2$")
plt.title(
    r"Energy fluctuation at $\rho^*=0.291$"
)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
# Mean energy
plt.figure(figsize=(7, 5))
plt.scatter(
    T_plot,
    U_plot,
    marker="o"
)
plt.xlabel(r"$T^*$")
plt.ylabel(r"$\langle U\rangle$")
plt.title(
    r"Mean energy at $\rho^*=0.291$"
)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
# Plot final configuration for each temperature
for T_star in temperatures:
    x_final, y_final = final_configurations[T_star]
    plt.figure(figsize=(6, 6))
    plt.scatter(
        x_final,
        y_final,
        s=20
    )
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        rf"Final configuration: $T^*={T_star:.2f}$, $\rho^*=0.291$"
    )
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()
# %%
# Sort temperature from low to high
order = np.argsort(T_values)
T_plot = T_values[order]
fluctuation_plot = fluctuation_values[order]
# Calculate Cv / kB
Cv_plot = fluctuation_plot / T_plot**2
# Plot Cv vs T
plt.figure(figsize=(7, 5))

plt.scatter(
    T_plot,
    Cv_plot
)
plt.xlabel(r"$T^*$")
plt.ylabel(r"$C_V/k_B$")

plt.title(
    r"Heat capacity at $\rho^*=0.291$"
)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
# Save all final configurations after simulation
T_saved = np.array(
    sorted(final_configurations.keys())
)
x_saved = np.stack([
    final_configurations[T][0]
    for T in T_saved
])
y_saved = np.stack([
    final_configurations[T][1]
    for T in T_saved
])
# Recalculate final total energy for each temperature
U_final_saved = np.array([
    total_energy(
        final_configurations[T][0],
        final_configurations[T][1],
        L
    )
    for T in T_saved
])
np.savez(
    "cooling_final_states.npz",
    temperatures=T_saved,
    x=x_saved,
    y=y_saved,
    U_total=U_final_saved,
    rho_star=rho_star,
    L=L,
    N=N
)
print("Saved successfully.")
# %%
for T_star in sorted(energy_histories.keys()):
    energy_values = energy_histories[T_star]
    steps = np.arange(len(energy_values)) * sample_interval
    plt.figure(figsize=(7, 5))
    plt.plot(steps, energy_values)
    plt.xlabel("Attempted MC moves in production")
    plt.ylabel("Total energy U")
    plt.title(rf"Production energy at $T^*={T_star:.2f}$")
    plt.tight_layout()
    plt.show()