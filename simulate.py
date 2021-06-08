"""
Methods to run simulations and output data to a file for later analysis.

TODO: 
1. Broadly refactor. Specifically in periodic temperature swap and keep 
a memo of inverses to beta_reloc that can be passed to methods.
2. Test if agrees with previous results.
3. Speed up. Specifically, investigate inline numba options. May also
investigate CUDA random number generation if there is time.

Author: Gabriel Peery
Date: 6/7/2021
"""
import argparse
import logging
from numba import boolean, njit, prange
import numpy as np
import pandas as pd
import time_logging
from typing import Tuple


numba_log = logging.getLogger("numba")
numba_log.setLevel(logging.DEBUG)


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


# Trying parallelization
@njit(parallel=True)
def _init_rand_states(states : np.ndarray):
    """Draws binding energies according to a Gaussian distribution with
    variance 1 and center 0 and randomizes initial spins.

    Effects:
        Changes bindings and spins of states, both pairs of sets.
    """
    size = len(states[0][0])
    rand_spins = np.random.choice(np.array([-1, 1]), size=(2, size, size, size))
    rand_binding = np.random.normal(0.0, 1.0, size=(2, 3, size, size, size))
    for stateA, stateB in zip(states[0], states[1]):
        for x in prange(size):
            for y in prange(size):
                for z in prange(size):
                    stateA[x][y][z].s = rand_spins[0][x][y][z]
                    stateB[x][y][z].s = rand_spins[1][x][y][z]
                    stateA[x][y][z].r = rand_binding[0][0][x][y][z]
                    stateB[x][y][z].r = rand_binding[1][0][x][y][z]
                    stateA[x][y][z].d = rand_binding[0][1][x][y][z]
                    stateB[x][y][z].d = rand_binding[1][1][x][y][z]
                    stateA[x][y][z].i = rand_binding[0][2][x][y][z]
                    stateB[x][y][z].i = rand_binding[1][2][x][y][z]


# Trying parallelization
@njit(parallel=True)
def _calc_energies(
        ss : np.ndarray,
        energies : np.ndarray
    ):
    """For each pair of states in ss, calculate the energies and put 
    them in the provided array.
    
    Effects:
        Modifies supplied energy array.
    """
    size = len(ss[0][0])
    for i in prange(len(ss[0])):
        energies[0][i] = energies[1][i] = 0
        for x in prange(size):
            for y in prange(size):
                for z in prange(size):
                    energies[0][i] += (ss[0][i][x][y][z].s * (
                        (ss[0][i][x][y][z].r*ss[0][i][(x+1)%size][y][z].s)
                        + (ss[0][i][x][y][z].d*ss[0][i][x][(y+1)%size][z].s)
                        + (ss[0][i][x][y][z].i*ss[0][i][x][y][(z+1)%size].s)
                    ))
                    energies[1][i] += (ss[1][i][x][y][z].s * (
                        (ss[1][i][x][y][z].r*ss[1][i][(x+1)%size][y][z].s)
                        + (ss[1][i][x][y][z].d*ss[1][i][x][(y+1)%size][z].s)
                        + (ss[1][i][x][y][z].i*ss[1][i][x][y][(z+1)%size].s)
                    ))


@njit
def _calc_change(st : np.ndarray, c : np.ndarray) -> float:
    """Returns the change in energy that would occur if the site in the
    state st at coordinate c were flipped.
    """
    s = len(st)
    return -2 * st[c[0]][c[1]][c[2]].s * (
        st[c[0]][c[1]][c[2]].r*st[(c[0]+1)%s][c[1]][c[2]].s
        + st[c[0]][c[1]][c[2]].d*st[c[0]][(c[1]+1)%s][c[2]].s
        + st[c[0]][c[1]][c[2]].i*st[c[0]][c[1]][(c[2]+1)%s].s
        + st[(c[0]-1)%s][c[1]][c[2]].r*st[(c[0]-1)%s][c[1]][c[2]].s
        + st[c[0]][(c[1]-1)%s][c[2]].d*st[c[0]][(c[1]-1)%s][c[2]].s
        + st[c[0]][c[1]][(c[2]-1)%s].i*st[c[0]][c[1]][(c[2]-1)%s].s
    )


@njit
def _calc_ss_overlap(
        stA : np.ndarray,
        stB : np.ndarray,
        x : int,
        y : int,
        z : int
    ) -> float:
    """Calculates the contribution of a single site to the link overlap.
    Returns that contribution.
    """
    s = len(stA)
    return (
        stA[x][y][z].s * stB[x][y][z].s * (
        (stA[(x+1)%s][y][z].s*stB[(x+1)%s][y][z].s)
        + (stA[x][(y+1)%s][z].s*stB[x][(y+1)%s][z].s)
        + (stA[x][y][(z+1)%s].s*stB[x][y][(z+1)%s].s)
        )
    )


# Trying parallelization
@njit(parallel=True)
def _calc_overlaps(
        ss : np.ndarray,
        spin_overlap : np.ndarray,
        link_overlap : np.ndarray,
        beta_reloc : np.ndarray,
        idx_trace : np.ndarray
    ):
    """For pairs of states in ss, calculate the overlaps index by index.
    Requires an array to store index transformations for reasons of
    parallelization.
    
    Effects:
        Modifies supplied overlap arrays.
    """
    size = len(ss[0][0])
    for i in prange(len(ss[0])):
        spin_overlap[i] = link_overlap[i] = 0.0
        idx_trace[i][0] = np.where(beta_reloc[0] == i)[0][0]
        idx_trace[i][1] = np.where(beta_reloc[1] == i)[0][0]
        for x in prange(size):
            for y in prange(size):
                for z in prange(size):
                    spin_overlap[i] += (
                        ss[0][idx_trace[i][0]][x][y][z].s 
                        * ss[1][idx_trace[i][1]][x][y][z].s
                    )
                    link_overlap[i] += _calc_ss_overlap(
                        ss[0][idx_trace[i][0]], ss[1][idx_trace[i][1]], x, y, z
                    )


# Trying parallelization
@njit(parallel=True)
def _mc_step(
        ss : np.ndarray,
        betas : np.ndarray,
        beta_reloc : np.ndarray,
        cs : np.ndarray,
        uniforms : np.ndarray,
        do_swaps : np.ndarray,
        changes : np.ndarray,
        energies : np.ndarray,
        spin_overlap : np.ndarray,
        link_overlap : np.ndarray,
        site_overlap : np.ndarray
    ):
    """Given states ss, information to get their temperatures, and some
    random values to use, performs a Metropolis-Hastings Monte Carlo
    step on all of them and updates quantities accordingly. The cs array 
    is of random coordinates. Also requires arrays to keep booleans, 
    changes in energy, and site_overlap for parallelization.

    Effects:
        May swap spins in states.
        Updates energies, spin_overlap, link_overlap.
        Modifies parallelization utility arrays
    """
    # Loop over beta indices
    for bi in prange(len(ss[0])):
        # We need to convert the beta index to state indices in A and B
        # We'll deal only with those coordinates for overlap calculation
        siA = np.where(beta_reloc[0] == bi)[0][0]
        siB = np.where(beta_reloc[0] == bi)[0][0]

        # Record old overlap in neighborhood of both coords
        coords_same = (cs[0][siA] == cs[1][siB]).all()
        if coords_same:
            site_overlap[bi] = _calc_ss_overlap(
                ss[0][siA],ss[1][siB],cs[0][siA][0],cs[0][siA][1],cs[0][siA][2]
            )
        else:
            site_overlap[bi] = (
                _calc_ss_overlap(
                    ss[0][siA],
                    ss[1][siB],
                    cs[0][siA][0],
                    cs[0][siA][1],
                    cs[0][siA][2]
                )
                + _calc_ss_overlap(
                    ss[0][siA],
                    ss[1][siB],
                    cs[1][siB][0],
                    cs[1][siB][1],
                    cs[1][siB][2]
                )
            )

        # Check if non-positive
        changes[0][siA] = _calc_change(ss[0][siA], cs[0][siA])
        changes[1][siB] = _calc_change(ss[1][siB], cs[1][siB])
        do_swaps[0][siA] = (changes[0][siA] <= 0)
        do_swaps[1][siB] = (changes[1][siB] <= 0)

        # If positive, use the uniforms
        if not do_swaps[0][siA]:
            do_swaps[0][siA] = (uniforms[0][siA] < np.exp(
                -betas[bi] * changes[0][siA]
            ))
        if not do_swaps[1][siB]:
            do_swaps[1][siB] = (uniforms[1][siB] < np.exp(
                -betas[bi] * changes[1][siB]
            ))

        # Perform swaps, update energy info
        if do_swaps[0][siA]:
            ss[0][siA][cs[0][siA][0]][cs[0][siA][1]][cs[0][siA][2]].s *= -1
            energies[0][siA] += changes[0][siA]
        if do_swaps[1][siB]:
            ss[1][siB][cs[1][siB][0]][cs[1][siB][1]][cs[1][siB][2]].s *= -1
            energies[1][siB] += changes[1][siB]

        # Update overlaps
        if coords_same:
            if not (do_swaps[0][siA] and do_swaps[1][siB]):
                spin_overlap[bi] -= 2 * (
                    ss[1][siB][cs[1][siB][0]][cs[1][siB][1]][cs[1][siB][2]].s
                    *ss[0][siA][cs[0][siA][0]][cs[0][siA][1]][cs[0][siA][2]].s 
                )
            link_overlap[bi] += (
                _calc_ss_overlap(
                    ss[0][siA],
                    ss[1][siB],
                    cs[0][siA][0],
                    cs[0][siA][1],
                    cs[0][siA][2]
                )
                - site_overlap[bi]
            )
        else:
            if do_swaps[0][siA]:
                spin_overlap[bi] -= 2 * (
                    ss[0][siA][cs[0][siA][0]][cs[0][siA][1]][cs[0][siA][2]].s
                    *ss[1][siB][cs[0][siA][0]][cs[0][siA][1]][cs[0][siA][2]].s 
                )
            if do_swaps[1][siB]:
                spin_overlap[bi] -= 2 * (
                    ss[0][siA][cs[1][siB][0]][cs[1][siB][1]][cs[1][siB][2]].s
                    *ss[1][siB][cs[1][siB][0]][cs[1][siB][1]][cs[1][siB][2]].s 
                )
            link_overlap[bi] += ((
                _calc_ss_overlap(
                    ss[0][siA],
                    ss[1][siB],
                    cs[0][siA][0],
                    cs[0][siA][1],
                    cs[0][siA][2]
                )
                + _calc_ss_overlap(
                    ss[0][siA],
                    ss[1][siB],
                    cs[1][siB][0],
                    cs[1][siB][1],
                    cs[1][siB][2]
                )
            ) - site_overlap[bi])


@time_logging.print_time
@njit
def ptmc(
        size : int,
        samples : int,
        sweeps : int,
        betas : np.ndarray,
        global_move_period : int,
        equilibriate_sweeps : int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Runs a Parallel-Tempering Monte Carlo simulation of an
    edwards_anderson model in a 3D cube with edge lengths size. samples
    is the number of times binding energies are recalculated and
    randomized. During each sample, two sets of systems are run
    independently. Each of the sets has a system for each of the betas 
    provided, which are run using Metropolis-Hastings Monte Carlo for 
    sweeps steps, and an attempt is made every global_move_period to 
    exchange adjacent betas of the systems. Also, the sweeps are allowed 
    to reach equilibrium for some number of sweeps.

    Returns some arrays with encountered values of spin and link overlap
    """
    #
    # Memory allocation, startup
    #
    states = np.zeros(
        (2, len(betas), size, size, size),
        dtype=_STATE_DTYPE
    )

    # Store the indices into betas of where to get beta of state at some
    # index. Stands for Beta _ Relocation
    beta_reloc = np.zeros((2, len(betas)), dtype=IDX_DTYPE)
    for i in range(len(betas)):
        beta_reloc[0][i] = beta_reloc[1][i] = i

    # Will initialize quantities of interest in loop
    energies = np.zeros((2, len(betas)), dtype=BINDING_DTYPE)
    # Note that for overlaps, we'll save scaling for later
    spin_overlap = np.zeros(len(betas), dtype=BINDING_DTYPE)
    link_overlap = np.zeros(len(betas), dtype=BINDING_DTYPE)

    # Logistics, counts, useful variables, ephemera arrays
    do_swaps = np.zeros((2, len(betas)), dtype=boolean)
    changes = np.zeros((2, len(betas)), dtype=BINDING_DTYPE)
    site_overlap_tmp = np.zeros(len(betas), dtype=BINDING_DTYPE)
    idx_trace = np.zeros((len(betas), 2), dtype=IDX_DTYPE)
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
    rand_coords = np.random.randint(0, size, size=(*mc_shape, 3))
    mc_uniforms = np.random.rand(*mc_shape)
    swap_idxs = np.random.randint(0, len(betas) - 1, size=swap_shape)
    swap_uniforms = np.random.rand(*swap_shape)
    print("...finished randoms.")

    # Output quantities of interest
    # Note that they are stored according to temperature and sample 
    # index and sweep index for multiple layers of averaging.
    spin_overlap_hist = np.zeros((len(betas), samples, sweeps))
    link_overlap_hist = np.zeros((len(betas), samples, sweeps))

    #
    # Sampling loop
    #
    print("Beginning Simulation with "+str(tot_mc_iters)+" Monte Carlo Steps")
    for sai in range(samples): # Sample index
        print("Running sample " + str(sai) + "...")

        #
        # Initalize
        #
        _init_rand_states(states)
        _calc_energies(states, energies)
        _calc_overlaps(
            states, spin_overlap, link_overlap, beta_reloc, idx_trace
        )
        print("Init Complete")

        #
        # Sweeping loop
        #
        for swi in range(real_sweeps): # Sweep index
            # Metropolis-Hastings Monte-Carlo
            for ssi in range(site_count): # Sub-sweep index
                _mc_step(
                    states,
                    betas,
                    beta_reloc,
                    rand_coords[sai][swi][ssi],
                    mc_uniforms[sai][swi][ssi],
                    do_swaps,
                    changes,
                    energies,
                    spin_overlap,
                    link_overlap,
                    site_overlap_tmp
                )

            # Record data after equilibrium phase
            if swi >= equilibriate_sweeps:
                # Record overlaps for histogram later
                for beta_idx in range(len(betas)):
                    spin_overlap_hist[beta_reloc[0][beta_idx]][sai][swi] = (
                        spin_overlap[beta_reloc[0][beta_idx]]
                    )
                    link_overlap_hist[beta_reloc[0][beta_idx]][sai][swi] = (
                        link_overlap[beta_reloc[0][beta_idx]]
                    )

            # Periodic Temperature Swap
            quo, rem = np.divmod(swi, global_move_period)
            if rem == 0:
                # Attempt swaps equal to number of betas
                for swt in range(len(betas)): # Swap trial
                    # -=- A -=-
                    # Choose a pair to swap
                    swap_idx = swap_idxs[sai][quo][swt][0]
                    low_i = np.where(beta_reloc[0]==swap_idx)[0][0]
                    high_i = np.where(beta_reloc[0]==(swap_idx+1))[0][0]
                    # Accept or reject based on boltzmann factor
                    exponent = (
                        -energies[0][low_i]*betas[beta_reloc[0][high_i]]
                        -energies[0][high_i]*betas[beta_reloc[0][low_i]]
                        +energies[0][low_i]*betas[beta_reloc[0][low_i]]
                        +energies[0][high_i]*betas[beta_reloc[0][high_i]]
                    )
                    if (
                        (exponent >= 1)
                        or (swap_uniforms[sai][quo][swt][0] <= np.exp(exponent))
                    ):
                        beta_reloc[0][low_i], beta_reloc[0][high_i] = (
                            beta_reloc[0][high_i],
                            beta_reloc[0][low_i]
                        )

                    # -=- B -=-
                    # Choose a pair to swap
                    swap_idx = swap_idxs[sai][quo][swt][1]
                    low_i = np.where(beta_reloc[1]==swap_idx)[0][0]
                    high_i = np.where(beta_reloc[1]==(swap_idx+1))[0][0]
                    # Accept or reject based on boltzmann factor
                    exponent = (
                        -energies[1][low_i]*betas[beta_reloc[1][high_i]]
                        -energies[1][high_i]*betas[beta_reloc[1][low_i]]
                        +energies[1][low_i]*betas[beta_reloc[1][low_i]]
                        +energies[1][high_i]*betas[beta_reloc[1][high_i]]
                    )
                    if (
                        (exponent >= 1)
                        or (swap_uniforms[sai][quo][swt][1] <= np.exp(exponent))
                    ):
                        beta_reloc[1][low_i], beta_reloc[1][high_i] = (
                            beta_reloc[1][high_i],
                            beta_reloc[1][low_i]
                        )

        print("...finished sample.")

    return spin_overlap_hist, link_overlap_hist


def ea_main():
    print("Starting simulation.")
    print(edwards_anderson(5, 1000, np.array([1])))


def ptmc_main():
    # Parse Arguments
    print("Parsing arguments...")
    parser = argparse.ArgumentParser(
        description="Run a Parallel-Tempering Monte Carlo simulation of "
                    "an Edwards-Anderson model in 3D."
    )
    parser.add_argument("size", type=int, help="Side length of cube.")
    parser.add_argument(
        "samples",
        type=int,
        help="Number of binding configs to sample"
    )
    parser.add_argument(
        "sweeps",
        type=int,
        help="Metropolis-Hastings iterations per binding config."
    )
    parser.add_argument(
        "global_move_period",
        type=int,
        help="Attempt temperature swapping every this iterations."
    )
    parser.add_argument(
        "equilibriate_sweeps",
        type=int,
        help="Iterations to calm down."
    )
    parser.add_argument(
        "out_name",
        type=str,
        help="Filename to output results to, should be a .csv."
    )
    args = parser.parse_args()
    size = args.size
    samples = args.samples
    sweeps = args.sweeps
    global_move_period = args.global_move_period
    equilibriate_sweeps = args.equilibriate_sweeps
    out_name = args.out_name
    print("...finished parsing arguments.")

    # Run simulation
    print("Starting Parallel Tempering Monte Carlo...")
    betas = np.linspace(0.2, 1.1, 10)
    spin_overlap_hist, link_overlap_hist = ptmc(
        size,
        samples,
        sweeps,
        betas,
        global_move_period,
        equilibriate_sweeps
    )
    print("...finished Monte Carlo.")

    # Write to file
    print(f'Writing to file "{out_name}"...')
    output_data = pd.DataFrame({
        "Beta" : [],
        "SampleID" : [],
        "SweepID" : [],
        "SpinOverlap" : [],
        "LinkOverlap" : []
    })
    for beta, s1, l1 in zip(betas, spin_overlap_hist, link_overlap_hist):
        for sai in range(len(s1)): # Sample index
            for swi in range(len(s1[sai])): # Sweep index
                output_data = output_data.append({
                    "Beta" : beta,
                    "SampleID" : sai,
                    "SweepID" : swi,
                    "SpinOverlap" : s1[sai][swi],
                    "LinkOverlap" : l1[sai][swi]
                }, ignore_index=True)
    output_data.to_csv(out_name, index=False)
    print("...finished writing to file.")


if __name__ == "__main__":
    ptmc_main()

