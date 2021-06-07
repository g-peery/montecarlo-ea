"""
Methods to run simulations and output data to a file for later analysis.

Author: Gabriel Peery
Date: 6/7/2021
"""
from numba import njit, prange
import numpy as np
import time_logging
from typing import Tuple


@njit
def get_site_energy(
    sites : np.ndarray,
    binding : np.ndarray,
    coord : np.ndarray
) -> float:
    """Gets the energy contribution of the site at some coord."""
    energy = 0
    energy -= (sites[(coord[0] + 1) % len(sites)][coord[1]][coord[2]]
                * binding[0][coord[0]][coord[1]][coord[2]])
    energy -= (sites[(coord[0] - 1) % len(sites)][coord[1]][coord[2]]
                * binding[0][(coord[0] - 1) % len(sites)][coord[1]][coord[2]])
    energy -= (sites[coord[0]][(coord[1] + 1) % len(sites)][coord[2]]
                * binding[1][coord[0]][coord[1]][coord[2]])
    energy -= (sites[coord[0]][(coord[1] - 1) % len(sites)][coord[2]]
                * binding[1][coord[0]][(coord[1] - 1) % len(sites)][coord[2]])
    energy -= (sites[coord[0]][coord[1]][(coord[2] + 1) % len(sites)]
                * binding[2][coord[0]][coord[1]][coord[2]])
    energy -= (sites[coord[0]][coord[1]][(coord[2] - 1) % len(sites)]
                * binding[2][coord[0]][coord[1]][(coord[2] - 1) % len(sites)])
    return energy * sites[coord[0]][coord[1]][coord[2]]


@njit
def get_energy(
    sites : np.ndarray,
    binding : np.ndarray
) -> float:
    """Gets the energy of the state"""
    energy = 0.0
    for x in range(len(sites)):
        for y in range(len(sites)):
            for z in range(len(sites)):
                energy += get_site_energy(sites, binding, np.array([x, y, z]))
    return energy / 2.0


@njit
def get_change(
    sites : np.ndarray,
    binding : np.ndarray,
    coord : np.ndarray
) -> float:
    """Gets the change in energy if the site at coord is flipped."""
    delta_E = 0
    return -2 * get_site_energy(sites, binding, coord)


@njit
def edwards_anderson(size : int, iters : int, temperatures : np.ndarray):
    sites = np.random.choice(np.array([-1, 1]), size=(size, size, size))
    binding = np.random.normal(0.0, 1.0, size=(3, size, size, size))
    site_count = (size**3)
    real_iters = site_count * iters
    rand_coords = np.random.randint(0, size, size=(real_iters, 3))
    uniforms = np.random.rand(real_iters)
    do_swap = False
    beta = 1 / temperatures[0]
    cur_energy = get_energy(sites, binding)
    avg_energy = 0.0
    
    # Monte Carlo loop
    for i in range(real_iters):
        this_coord = rand_coords[i]
        delta_E = get_change(sites, binding, this_coord)
        do_swap = (delta_E <= 0)
        if not do_swap:
            do_swap = (uniforms[i] < np.exp(-beta*delta_E))
        if do_swap:
            sites[this_coord[0]][this_coord[1]][this_coord[2]] *= -1
            cur_energy += delta_E
        # Record state info
        if i % site_count == 0:
            avg_energy += cur_energy
    avg_energy = avg_energy / iters
    return avg_energy


SPIN_DTYPE = np.dtype("i1")
BINDING_DTYPE = np.dtype("f4")
_STATE_DTYPE = np.dtype([
    ("s", SPIN_DTYPE),    # Spin
    ("r", BINDING_DTYPE), # Right binding
    ("d", BINDING_DTYPE), # Down binding
    ("i", BINDING_DTYPE)  # In binding
])

IDX_DTYPE = np.dtype("i1") # Change if using more temperatures


@njit
def _init_rand_states(statesA : np.ndarray, statesB : np.ndarray):
    """Draws binding energies according to a Gaussian distribution with
    variance 1 and center 0 and randomizes initial spins.

    Effects:
    Changes bindings and spins of both states arrays.
    """
    size = len(statesA[0])
    rand_spins = np.random.choice(np.array([-1, 1]), size=(2, size, size, size))
    rand_binding = np.random.normal(0.0, 1.0, size=(2, 3, size, size, size))
    for stateA, stateB in zip(statesA, statesB):
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    stateA[x][y][z]["s"] = rand_spins[0][x][y][z]
                    stateB[x][y][z]["s"] = rand_spins[1][x][y][z]
                    stateA[x][y][z]["r"] = rand_binding[0][0][x][y][z]
                    stateB[x][y][z]["r"] = rand_binding[1][0][x][y][z]
                    stateA[x][y][z]["d"] = rand_binding[0][1][x][y][z]
                    stateB[x][y][z]["d"] = rand_binding[1][1][x][y][z]
                    stateA[x][y][z]["i"] = rand_binding[0][2][x][y][z]
                    stateB[x][y][z]["i"] = rand_binding[1][2][x][y][z]


# Trying experimental parallelization
@njit(parallel=True)
def _calc_energies(
        stA : np.ndarray,
        stB : np.ndarray,
        energiesA : np.ndarray,
        energiesB : np.ndarray
    ):
    """For each state in stA and stB are the same size, calculate the 
    energies and put them in the provided arrays.
    
    Effects:
        Modifies supplied energy arrays.
    """
    size = len(states[0])
    for i in prange(len(stA)):
        energiesA[i] = energiesB[i] = 0
        for x in prange(size):
            for y in prange(size):
                for z in prange(size):
                    energiesA[i] += (stA[i][x][y][z]["s"] * (
                        (stA[i][x][y][z]["r"]*stA[i][(x+1)%size][y][z]["s"])
                        + (stA[i][x][y][z]["d"]*stA[i][x][(y+1)%size][z]["s"])
                        + (stA[i][x][y][z]["i"]*stA[i][x][y][(z+1)%size]["s"])
                    ))
                    energiesB[i] += (stB[i][x][y][z]["s"] * (
                        (stB[i][x][y][z]["r"]*stB[i][(x+1)%size][y][z]["s"])
                        + (stB[i][x][y][z]["d"]*stB[i][x][(y+1)%size][z]["s"])
                        + (stB[i][x][y][z]["i"]*stB[i][x][y][(z+1)%size]["s"])
                    ))


# Trying experimental parallelization
@njit(parallel=True)
def _mc_step(
        stA : np.ndarray,
        stB : np.ndarray,
        betas : np.ndarray,
        betaA_reloc : np.ndarray,
        betaB_reloc : np.ndarray,
        cs : np.ndarray,
        uniforms : np.ndarray,
        do_swaps : np.ndarray,
        changes : np.ndarray,
        energiesA : np.ndarray,
        energiesB : np.ndarray
    ):
    """Given states, information to get their temperatures, and some
    random values to use, performs a Metropolis-Hastings Monte Carlo
    step on all of them and updates energiesA and energiesB accordingly.
    The cs array is of random coordinates. Also requires arrays to keep 
    booleans and changes in energy for parallelization.

    Effects:
        May swap spins in states.
        Updates energiesA and energiesB.
    """
    # Loop over state indices
    for sti in prange(len(stA)):
        # Check if non-positive
        changes[0][sti] = _calc_change(stA[sti], cs[0][sti])
        changes[1][sti] = _calc_change(stA[sti], cs[0][sti])
        do_swaps[0][sti] = (changes[0][sti] <= 0)
        do_swaps[1][sti] = (changes[1][sti] <= 0)

        # If positive, use the uniforms
        if not do_swaps[0][sti]:
            do_swaps[0][sti] = (uniforms[0][sti] < np.exp(
                -betas[betaA_reloc[sti]] * changes[0][sti]
            ))
        if not do_swaps[1][sti]:
            do_swaps[1][sti] = (uniforms[1][sti] < np.exp(
                -betas[betaB_reloc[sti]] * changes[1][sti]
            ))

        # Perform swaps, update energy info
        # TODO - can overlaps be updated here?
        if do_swaps[0][sti]:
            stA[sti][cs[0][sti][0]][cs[0][sti][1]][cs[0][sti][2]]["s"] *= -1
            energiesA[sti] += changes[0][sti]
        if do_swaps[1][sti]:
            stB[sti][cs[1][sti][0]][cs[1][sti][1]][cs[1][sti][2]]["s"] *= -1
            energiesB[sti] += changes[1][sti]


@time_logging.print_time
@njit
def ptmc(
        size : int,
        samples : int,
        sweeps : int,
        betas : np.ndarray,
        global_move_period : int,
        equilibriate_sweeps : int
        ) -> np.ndarray:
    """Runs a Parallel-Tempering Monte Carlo simulation of an
    edwards_anderson model in a 3D cube with edge lengths size. samples
    is the number of times binding energies are recalculated and
    randomized. During each sample, two sets of systems are run
    independently. Each of the sets has a system for each of the betas 
    provided, which are run using Metropolis-Hastings Monte Carlo for 
    sweeps steps, and an attempt is made every global_move_period to 
    exchange adjacent betas of the systems. Also, the sweeps are allowed 
    to reach equilibrium for some number of sweeps.
    """
    #
    # Memory allocation, startup
    #
    statesA = np.zeros(
        (len(betas), size, size, size),
        dtype=_STATE_DTYPE
    )
    statesB = np.zeros(
        (len(betas), size, size, size),
        dtype=_STATE_DTYPE
    )

    # Store the indices into betas of where to get current
    # beta. Stands for Temperature _ Relocation
    betaA_reloc = np.arange(len(betas), dtype=IDX_DTYPE)
    betaB_reloc = np.arange(len(betas), dtype=IDX_DTYPE)

    # Will initialize energies in loop
    energiesA = np.zeros(len(betas), dtype=BINDING_DTYPE)
    energiesB = np.zeros(len(betas), dtype=BINDING_DTYPE)

    # Logistics, counts, useful variables
    do_swaps = np.zeros((2, len(betas)), dtype=bool)
    changes = np.zeros((2, len(betas)), dtype=BINDING_DTYPE) # Ephemera
    site_count = size ** 3
    real_sweeps = equilibriate_sweeps + sweeps
    sub_sweep_count = site_count * real_sweeps
    tot_mc_iters = samples * sub_sweep_count * len(betas) * 2
    mc_shape = (samples, real_sweeps, site_count, 2, len(betas))
    swap_shape = (
        samples,
        np.ceil(real_sweeps / global_move_period),
        len(betas),
        2
    )
    quo = rem = 0 # For Modular Operations
    swap_idx = low_i = high_i = 0 # Beta swapping indices
    exponent = 0.0

    # Generate randoms from start
    print("Generating randoms...")
    rand_coords = np.random.randint(0, size, size=mc_shape)
    mc_uniforms = np.random.rand(mc_shape)
    swap_idxs = np.random.randint(0, len(betas) - 1, size=swap_shape)
    swap_uniforms = np.random.rand(swap_shape)
    print("...finished randoms.")

    #
    # Sampling loop
    #
    print("Beginning Simulation with %i Monte Carlo Steps" % tot_mc_iters)
    for sai in range(samples): # Sample index
        print("Running sample %i..." % sai)

        #
        # Initalize
        #
        _init_rand_states(statesA, statesB)
        _calc_energies(statesA, statesB, energiesA, energiesB)

        #
        # Sweeping loop
        #
        for swi in range(real_sweeps): # Sweep index
            # Metropolis-Hastings Monte-Carlo
            for ssi in range(site_count): # Sub-sweep index
                _mc_step(
                    statesA,
                    statesB,
                    betas,
                    betaA_reloc,
                    betaB_reloc,
                    rand_coords[sai][swi][ssi],
                    mc_uniforms[sai][swi][ssi],
                    do_swaps,
                    changes,
                    energies_A,
                    energies_B
                )

            # Record data after equilibrium phase
            if swi >= real_eq_sweeps:
                # TODO
                pass

            # Periodic Temperature Swap
            quo, rem = np.divmod(swi, global_move_period)
            if rem == 0:
                # Attempt swaps equal to number of betas
                for swap_trial in range(len(betas)):
                    # Choose a pair to swap
                    swap_idx = swap_idxs[sai][quo][swap_trial][0]
                    low_i = np.where(betaA_reloc==swap_idx)[0][0]
                    high_i = np.where(betaA_reloc==(swap_idx+1))[0][0]
                    # Accept or reject based on boltzmann factor
                    exponent = (
                        -energiesA[low_i]*betas[betaA_reloc[high_i]]
                        -energiesA[high_i]*betas[betaA_reloc[low_i]]
                        +energiesA[low_i]*betas[betaA_reloc[low_i]]
                        +energiesA[high_i]*betas[betaA_reloc[high_i]]
                    )
                    if (exponent >= 1) or (swap_uniforms <= np.exp(exponent)):
                        betaA_reloc[low_i], betaA_reloc[high_i] = (
                            betaA_reloc[high_i],
                            betaA_reloc[low_i]
                        )

        print("...finished sample %i." % sai)

    return statesA


def ea_main():
    print("Starting simulation.")
    print(edwards_anderson(5, 1000, np.array([1])))


def ptmc_main():
    print("Starting Parallel Tempering Monte Carlo.")
    ptmc(5, 1, 1, np.ones(10), 100, 10)


if __name__ == "__main__":
    ptmc_main()

