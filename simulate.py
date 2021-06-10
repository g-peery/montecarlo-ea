"""
Methods to run simulations and output data to a file for later analysis.
Directory operations only work on Windows, systems with '\\' separators.

Note that the global_move_period should be small.

Notable flaws: 
    1. Energy prediction results don't make sense.

Author: Gabriel Peery
Date: 6/9/2021
"""
import argparse
from json import dump
from numba import boolean, njit, prange
import numpy as np
import os
import time_logging
from typing import Tuple


SPIN_DTYPE = np.dtype("i1")
BINDING_DTYPE = np.dtype("f4")
_STATE_DTYPE = np.dtype([
    ("s", SPIN_DTYPE),    # Spin
    ("r", BINDING_DTYPE), # Right binding
    ("d", BINDING_DTYPE), # Down binding
    ("i", BINDING_DTYPE)  # In binding
])

IDX_DTYPE = np.dtype("i1") # Change if using more temperatures

HIST_COUNTER_DTYPE = np.dtype("u8") # Expected need up to 12.15e9

OVERLAP_DTYPE = np.dtype("i2")


@njit
def get_blank_states(size : int, beta_count : int) -> np.ndarray:
    """Generates a blank template for states to go in of some size. Will
    be 3D.
    """
    return np.zeros(
        (2, beta_count, size, size, size),
        dtype=_STATE_DTYPE
    )


@njit(parallel=True)
def init_rand_states(states : np.ndarray):
    """Draws binding energies according to a Gaussian distribution with
    variance 1 and center 0 and randomizes initial spins.

    Effects:
        Changes bindings and spins of states, both pairs of sets.
    """
    size = len(states[0][0])
    rand_spins = np.random.choice(np.array([-1, 1]), size=(2, size, size, size))
    rand_binding = np.random.normal(0.0, 1.0, size=(3, size, size, size))
    for stateA, stateB in zip(states[0], states[1]):
        for x in prange(size):
            for y in prange(size):
                for z in prange(size):
                    stateA[x][y][z].s = rand_spins[0][x][y][z]
                    stateB[x][y][z].s = rand_spins[1][x][y][z]
                    stateA[x][y][z].r = rand_binding[0][x][y][z]
                    stateB[x][y][z].r = rand_binding[0][x][y][z]
                    stateA[x][y][z].d = rand_binding[1][x][y][z]
                    stateB[x][y][z].d = rand_binding[1][x][y][z]
                    stateA[x][y][z].i = rand_binding[2][x][y][z]
                    stateB[x][y][z].i = rand_binding[2][x][y][z]


@njit(parallel=True)
def calc_energies(
        ss : np.ndarray,
        energies : np.ndarray
    ):
    """For each pair of states in ss, calculate the energies and put 
    them in the provided array.
    
    Effects:
        Modifies supplied energy array.
    """
    size = len(ss[0][0])
    # Loop over state indices
    for i in prange(len(ss[0])):
        energies[0][i] = energies[1][i] = 0
        # Loop over grid
        for x in prange(size):
            for y in prange(size):
                for z in prange(size):
                    energies[0][i] -= (ss[0][i][x][y][z].s * (
                        (ss[0][i][x][y][z].r*ss[0][i][(x+1)%size][y][z].s)
                        + (ss[0][i][x][y][z].d*ss[0][i][x][(y+1)%size][z].s)
                        + (ss[0][i][x][y][z].i*ss[0][i][x][y][(z+1)%size].s)
                    ))
                    energies[1][i] -= (ss[1][i][x][y][z].s * (
                        (ss[1][i][x][y][z].r*ss[1][i][(x+1)%size][y][z].s)
                        + (ss[1][i][x][y][z].d*ss[1][i][x][(y+1)%size][z].s)
                        + (ss[1][i][x][y][z].i*ss[1][i][x][y][(z+1)%size].s)
                    ))


@njit
def calc_energy_change(st : np.ndarray, c : np.ndarray) -> float:
    """Returns the change in energy that would occur if the site in the
    state st at coordinate c were flipped.
    """
    s = len(st)
    return 2 * st[c[0]][c[1]][c[2]].s * (
        st[c[0]][c[1]][c[2]].r*st[(c[0]+1)%s][c[1]][c[2]].s
        + st[c[0]][c[1]][c[2]].d*st[c[0]][(c[1]+1)%s][c[2]].s
        + st[c[0]][c[1]][c[2]].i*st[c[0]][c[1]][(c[2]+1)%s].s
        + st[(c[0]-1)%s][c[1]][c[2]].r*st[(c[0]-1)%s][c[1]][c[2]].s
        + st[c[0]][(c[1]-1)%s][c[2]].d*st[c[0]][(c[1]-1)%s][c[2]].s
        + st[c[0]][c[1]][(c[2]-1)%s].i*st[c[0]][c[1]][(c[2]-1)%s].s
    )


@njit
def calc_ss_overlap(
        stA : np.ndarray,
        stB : np.ndarray,
        x : int,
        y : int,
        z : int
    ) -> float:
    """Calculates the contribution of a single site to the link overlap,
    in all directions. Returns that contribution.
    """
    s = len(stA)
    return (
        stA[x][y][z].s * stB[x][y][z].s * (
        (stA[(x+1)%s][y][z].s*stB[(x+1)%s][y][z].s)
        + (stA[x][(y+1)%s][z].s*stB[x][(y+1)%s][z].s)
        + (stA[x][y][(z+1)%s].s*stB[x][y][(z+1)%s].s)
        + (stA[(x-1)%s][y][z].s*stB[(x-1)%s][y][z].s)
        + (stA[x][(y-1)%s][z].s*stB[x][(y-1)%s][z].s)
        + (stA[x][y][(z-1)%s].s*stB[x][y][(z-1)%s].s)
        )
    )


@njit
def _calc_sh_overlap(
        stA : np.ndarray,
        stB : np.ndarray,
        x : int,
        y : int,
        z : int
    ) -> float:
    """Calculates the contribution of a single site to the link overlap,
    but only in the right, down, and in directions (half). Returns that 
    contribution.
    """
    s = len(stA)
    return (
        stA[x][y][z].s * stB[x][y][z].s * (
        (stA[(x+1)%s][y][z].s*stB[(x+1)%s][y][z].s)
        + (stA[x][(y+1)%s][z].s*stB[x][(y+1)%s][z].s)
        + (stA[x][y][(z+1)%s].s*stB[x][y][(z+1)%s].s)
        )
    )


@njit
def calc_link_overlap(
        sA : np.ndarray,
        sB : np.ndarray
    ) -> int:
    """For a pairs of states sA and sB, calculate the link overlap index 
    by index. Return it.
    """
    size = len(sA)
    link_overlap = 0
    for x in range(size):
        for y in range(size):
            for z in range(size):
                link_overlap += _calc_sh_overlap(sA, sB, x, y, z)
    return link_overlap


@njit(parallel=True)
def calc_overlaps(
        ss : np.ndarray,
        spin_overlap : np.ndarray,
        link_overlap : np.ndarray,
        bts : np.ndarray
    ):
    """For pairs of states in ss, calculate the overlaps index by index.
    
    Effects:
        Modifies supplied overlap arrays.
    """
    size = len(ss[0][0])
    # Loop over indices into beta, as well as qoi, arrays
    for bi in prange(len(ss[0])):
        spin_overlap[bi] = link_overlap[bi] = 0
        for x in prange(size):
            for y in prange(size):
                for z in prange(size):
                    spin_overlap[bi] += (
                        ss[0][bts[0][bi]][x][y][z].s 
                        * ss[1][bts[1][bi]][x][y][z].s
                    )
                    link_overlap[bi] += _calc_sh_overlap(
                        ss[0][bts[0][bi]], ss[1][bts[1][bi]], x, y, z
                    )


@njit
def _update_obs_tbls(
        spin_tbl : np.ndarray,
        link_tbl : np.ndarray,
        spin_overlap : np.ndarray,
        link_overlap : np.ndarray
    ):
    """Updates the observation tables according to overlaps."""
    spin_offset = len(spin_tbl[0]) // 2
    link_offset = len(link_tbl[0]) // 2
    for bi, (spin, link) in enumerate(zip(spin_overlap, link_overlap)):
        spin_tbl[bi][spin + spin_offset] += 1
        link_tbl[bi][link + link_offset] += 1


@njit
def _mc_iter(
        # Arguments pertaining to only this iteration
        bi : int, # Index into betas
        si : int, # Index into this context's states
        c : np.ndarray, # Coordinate
        uniform : float,
        states : np.ndarray, # Only one half of copies
        # Global arguments
        betas : np.ndarray
    ) -> float:
    """Performs a single iteration of a monte carlo step with given
    indices into states and betas.

    Returns: 
        Resulting change in energy
    Effects:
        Flips the spin if accepted
    """
    # Check if non-positive
    energy_change = calc_energy_change(states[si], c)
    do_swap = (energy_change <= 0)

    # If positive, use the uniform
    if not do_swap:
        do_swap = (uniform < np.exp(-betas[bi] * energy_change))

    # Perform swaps, update energy and spin overlap info
    if do_swap:
        states[si][c[0]][c[1]][c[2]].s *= -1
    else:
        energy_change = 0.0

    return energy_change


@njit(parallel=True)
def mc_step(
        ss : np.ndarray,
        betas : np.ndarray,
        # Randoms
        cs : np.ndarray,
        uniforms : np.ndarray,
        # Observables
        energies : np.ndarray,
        # Index transforms
        bts : np.ndarray
    ):
    """Given states ss, information to get their temperatures, and some
    random values to use, performs a Metropolis-Hastings Monte Carlo
    step on all of them and updates quantities accordingly. The cs array 
    is of random coordinates.

    Effects:
        May swap spins in states.
        Updates energies
    """
    # Loop over indices into betas (and overlap arrays)
    for bi in prange(len(ss[0])):
        siA = bts[0][bi]
        siB = bts[1][bi]
        # A
        energies[0][siA] += _mc_iter(
            bi,
            siA,
            cs[0][bi],
            uniforms[0][bi],
            ss[0],
            betas
        )
        # B
        energies[1][siB] += _mc_iter(
            bi,
            siB,
            cs[1][bi],
            uniforms[1][bi],
            ss[1],
            betas
        )


@njit
def attempt_swap(
        bsi : np.ndarray, # Beta swap index i_beta
        energies : np.ndarray,
        betas : np.ndarray,
        bts : np.ndarray,
        stb : np.ndarray,
        swap_uniform : float,
        accepted : np.ndarray,
        attempts : np.ndarray
    ):
    """Given a random index at which to swap temperatures in the betas
    array, attempts a swap.
    """
    # Get indices into beta array of temperatures
    # that will swap
    low_i = bsi
    high_i = bsi + 1
    attempts[low_i] += 1
    # Accept or reject based on boltzmann factor
    exponent = (
        (betas[low_i] - betas[high_i])
        * (energies[bts[low_i]] - energies[bts[high_i]])
    )
    # Actual swapping
    if (exponent >= 0.0) or (swap_uniform <= np.exp(exponent)):
        accepted[low_i] += 1
        # Update state->beta first
        stb[bts[low_i]] = high_i
        stb[bts[high_i]] = low_i
        # Then swap beta->state
        bts[low_i], bts[high_i] = bts[high_i], bts[low_i]


START_POWER = 5

@time_logging.print_time
@njit
def ptmc(
        size : int,
        samples : int,
        sweeps : int,
        betas : np.ndarray,
        global_move_period : int,
        warmup_sweeps : int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs a Parallel-Tempering Monte Carlo simulation of an
    edwards_anderson model in a 3D cube with edge lengths size. samples
    is the number of times binding energies are recalculated and
    randomized. During each sample, two sets of systems are run
    independently. Each of the sets has a system for each of the betas 
    provided, which are run using Metropolis-Hastings Monte Carlo for 
    sweeps steps, and an attempt is made every global_move_period to 
    exchange adjacent betas of the systems. Also, the sweeps are allowed 
    to reach equilibrium for some number of sweeps.

    Returns some arrays that act as tables for link overlap time
    evolution, spin and link overlap histograms, and acceptance
    frequencies.
    """
    #
    # Memory allocation, startup
    #
    states = get_blank_states(size, len(betas))

    # Store maps between indices of beta and states to avoid huge
    # relocations.
    bts = np.zeros((2, len(betas)), dtype=IDX_DTYPE) # Beta to State
    stb = np.zeros((2, len(betas)), dtype=IDX_DTYPE) # State to Beta
    for i in range(len(betas)):
        bts[0][i] = bts[1][i] = stb[0][i] = stb[1][i] = i

    # Will initialize quantities of interest in loop
    # Indices here match states so that we don't have to swap with betas
    energies = np.zeros((2, len(betas)), dtype=BINDING_DTYPE)
    # Note that for overlaps, we'll save scaling for later
    # Indices match beta indices, but they are recalculated each save
    spin_overlap = np.zeros(len(betas), dtype=OVERLAP_DTYPE)
    link_overlap = np.zeros(len(betas), dtype=OVERLAP_DTYPE)

    # Logistics, counts, useful variables, ephemera arrays
    site_count = size ** 3
    real_sweeps = warmup_sweeps + sweeps
    sub_sweep_count = site_count * real_sweeps
    tot_mc_iters = samples * sub_sweep_count * len(betas) * 2
    mc_shape = (real_sweeps, site_count, 2, len(betas))
    swap_shape = (
        np.ceil(real_sweeps / global_move_period),
        len(betas),
        2
    )
    quo = rem = 0 # For Modular Operations

    #
    # Output quantities of interest (qoi)
    #
    # First is the prediction of link overlap from average energy
    # Second is the observed link overlap
    # Starts as a sum of averages, then averaged again at the end.
    time_ev_tbls = np.zeros(
        (2, len(betas), int(np.ceil(np.log2(sweeps) - START_POWER + 1))),
        dtype=BINDING_DTYPE
    )
    # Start as sums, turned to averages when put in time_ev_tbls
    avg_energies = np.zeros(len(betas), dtype=BINDING_DTYPE)
    # Will be normalized upon becoming average
    bind_count = 3 * site_count
    avg_ql = np.zeros(len(betas), dtype=BINDING_DTYPE)
    pred_norms = np.zeros(len(betas), dtype=BINDING_DTYPE)
    for bi in range(len(betas)):
        pred_norms[bi] = 1.0 / (bind_count * betas[bi])
    # Will be initialized in loop
    era = 0
    era_len = 0
    era_counter = 0

    # Each subtable is count of occurrences in bins.
    # Both start at counts, will be normalized and averaged at end.
    # Will iterate through, checking if less than until end
    q_bin_count = 2 * site_count + 1
    ql_bin_count = 2 * bind_count + 1
    # First is of spin overlap, second is link overlap
    spin_tbl = np.zeros((len(betas), q_bin_count), dtype=HIST_COUNTER_DTYPE)
    link_tbl = np.zeros((len(betas), ql_bin_count), dtype=HIST_COUNTER_DTYPE)

    # Acceptance frequencies
    accepted = np.zeros(len(betas) - 1, dtype=HIST_COUNTER_DTYPE)
    attempts = np.zeros(len(betas) - 1, dtype=HIST_COUNTER_DTYPE)

    #
    # Sampling loop
    #
    print("Beginning Simulation with "+str(tot_mc_iters)+" Monte Carlo Steps")
    for sai in range(samples): # Sample index
        print("Running sample " + str(sai) + "...")

        #
        # Output control
        #
        era = 0
        era_len = 2 ** START_POWER
        era_counter = 0
        for bi in range(len(betas)):
            avg_energies[bi] = avg_ql[bi] = 0.0

        #
        # Initalize
        #
        init_rand_states(states)
        calc_energies(states, energies)
        calc_overlaps(states, spin_overlap, link_overlap, bts)

        #
        # Generate randoms - this could take up lots of memory
        #
        print("Generating randoms...")
        rand_coords = np.random.randint(0, size, size=(*mc_shape, 3))
        mc_uniforms = np.random.rand(*mc_shape)
        swap_idxs = np.random.randint(0, len(betas) - 1, size=swap_shape)
        swap_uniforms = np.random.rand(*swap_shape)
        print("...finished randoms.")

        #
        # Sweeping loop
        #
        for swi in range(real_sweeps): # Sweep index
            #
            # Metropolis-Hastings Monte-Carlo
            #
            for ssi in range(site_count): # Sub-sweep index
                mc_step(
                    states,
                    betas,
                    rand_coords[swi][ssi],
                    mc_uniforms[swi][ssi],
                    energies,
                    bts
                )

            #
            # Record data after warmup
            #
            if swi >= warmup_sweeps:
                # Recalculate overlaps
                calc_overlaps(states, spin_overlap, link_overlap, bts)

                # Time Evolution Tables
                # Contribute to average (starting as sums)
                for bi in range(len(betas)):
                    avg_energies[bi] += (
                        energies[0][bts[0][bi]]
                        + energies[1][bts[1][bi]]
                    )
                    avg_ql[bi] += link_overlap[bi]
                era_counter += 1
                # Check if time to update table
                # Has the effect of doubling save period each time
                if era_counter == era_len:
                    # Add to table
                    for bi in range(len(betas)):
                        time_ev_tbls[0][bi][era] += avg_energies[bi] / era_len
                        #time_ev_tbls[0][bi][era] += 1 - (
                            #pred_norms[bi] * np.abs(avg_energies[bi]) / era_len
                        #)
                        time_ev_tbls[1][bi][era] += avg_ql[bi] / (
                            era_len * bind_count
                        )
                        # Reset averages after every era
                        avg_energies[bi] = avg_ql[bi] = 0.0
                    # Increments
                    era += 1
                    era_len *= 2
                    era_counter = 0

                # Observation Tables (Later Histogram Tables)
                _update_obs_tbls(
                    spin_tbl,
                    link_tbl,
                    spin_overlap,
                    link_overlap
                )

            #
            # Periodic Temperature Swap
            #
            quo, rem = np.divmod(swi, global_move_period)
            if rem == 0:
                # Attempt swaps equal to number of betas
                for swt in range(len(betas)): # Swap trial
                    # Swap A
                    attempt_swap(
                            swap_idxs[quo][swt][0],
                            energies[0],
                            betas,
                            bts[0],
                            stb[0],
                            swap_uniforms[quo][swt][0],
                            accepted,
                            attempts
                        )
                    # Swap B
                    attempt_swap(
                            swap_idxs[quo][swt][1],
                            energies[1],
                            betas,
                            bts[1],
                            stb[1],
                            swap_uniforms[quo][swt][1],
                            accepted,
                            attempts
                        )

        # Endpoint of Time Evolution Table
        if era_counter != 0:
            # Add to table
            for bi in range(len(betas)):
                time_ev_tbls[0][bi][era] += avg_energies[bi] / era_counter
                #time_ev_tbls[0][bi][era] += 1 - (
                    #pred_norms[bi] * np.abs(avg_energies[bi]) / era_counter
                #)
                time_ev_tbls[1][bi][era] += avg_ql[bi] / (
                    era_counter * bind_count
                )
                    
    #
    # Return, prepare data first
    #
    # Turn into average
    time_ev_tbls /= samples
    for x in range(len(time_ev_tbls[0])):
        for y in range(len(time_ev_tbls[0][x])):
            time_ev_tbls[0][x][y] = 1 - (
                pred_norms[x] * np.abs(time_ev_tbls[0][x][y])
            )
    # Turn into average
    spin_hist_tbl = spin_tbl / (samples * sweeps)
    link_hist_tbl = link_tbl / (samples * sweeps)
    return time_ev_tbls, spin_hist_tbl, link_hist_tbl, (accepted / attempts)


def alt_plot(link_overlap_hist):
    l1 = link_overlap_hist[0]
    ql_avgs = [0] * len(l1[0])
    for sai in range(len(l1)):
        for swi in range(len(l1[sai])):
            ql_avgs[swi] += l1[sai][swi]
    with open(out_name, "w") as fd:
        fd.write("Avg\n")
        for swi in range(len(l1[0])):
            ql_avgs[swi] /= (27 * 6 * len(l1))
            fd.write(f"{ql_avgs[swi]}\n")

    
def old_file_write(out_name, betas, spin_overlap_hist, link_overlap_hist):
    # Write to file
    print(f'Writing to file "{out_name}"...')
    with open(out_name, "w") as fd:
        fd.write("Beta,SampleID,SweepID,SpinOverlap,LinkOverlap\n")
        for beta, s1, l1 in zip(betas, spin_overlap_hist, link_overlap_hist):
            for sai in range(len(s1)): # Sample index
                for swi in range(len(s1[sai])): # Sweep index
                    fd.write(
                        f"{beta:.3},{sai},{swi},{s1[sai][swi]},{l1[sai][swi]}\n"
                    );
    print("...finished writing to file.")


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
        "warmup_sweeps",
        type=int,
        help="Iterations to calm down."
    )
    parser.add_argument(
        "min_temp",
        type=float,
        help="Minimum temperature"
    )
    parser.add_argument(
        "max_temp",
        type=float,
        help="Maximum temperature"
    )
    parser.add_argument(
        "temp_count",
        type=int,
        help="Number of temperatures in range"
    )
    parser.add_argument(
        "out_name",
        type=str,
        help="Directory to output results to."
    )
    args = parser.parse_args()
    size = args.size
    samples = args.samples
    sweeps = args.sweeps
    global_move_period = args.global_move_period
    warmup_sweeps = args.warmup_sweeps
    min_temp = args.min_temp
    max_temp = args.max_temp
    temp_count = args.temp_count
    out_name = args.out_name
    print("...finished parsing arguments.")

    # Prepare directory
    print("Preparing directory...")
    if not os.path.exists(out_name):
        os.makedirs(out_name)
    # Write config.json
    with open(out_name + r"\config.json", "w") as config:
        dump({
            "size" : size,
            "samples" : samples,
            "sweeps" : sweeps,
            "global_move_period" : global_move_period,
            "warmup_sweeps" : warmup_sweeps,
            "min_temp" : max_temp,
            "max_temp" : max_temp,
            "temp_count" : temp_count
        }, config)
    # Write meta.csv
    temps = np.linspace(min_temp, max_temp, temp_count)
    betas = temps ** (-1)
    with open(out_name + r"\meta.csv", "w") as meta:
        meta.write("Index,Beta,Temperature\n")
        for idx, (beta, temp) in enumerate(zip(betas, temps)):
            meta.write(f"{idx},{beta},{temp}\n")
    print("...finished.")

    # Run simulation
    time_ev_tbls, spin_tbl, link_tbl, acceptance = ptmc(
        size,
        samples,
        sweeps,
        betas,
        global_move_period,
        warmup_sweeps
    )

    # Write link_energy_predict.csv
    print("Writing results...")
    with open(out_name + r"\link_energy_predict.csv", "w") as lep:
        for row in time_ev_tbls[0]:
            lep.write(",".join([str(item) for item in row]) + "\n")
    # Write link_time.csv
    with open(out_name + r"\link_time.csv", "w") as lt:
        for row in time_ev_tbls[1]:
            lt.write(",".join([str(item) for item in row]) + "\n")
    # Write spin.csv
    with open(out_name + r"\spin.csv", "w") as spin_fd:
        for row in spin_tbl:
            spin_fd.write(",".join([str(item) for item in row]) + "\n")
    # Write link.csv
    with open(out_name + r"\link.csv", "w") as link_fd:
        for row in link_tbl:
            link_fd.write(",".join([str(item) for item in row]) + "\n")
    # Write accept.csv
    with open(out_name + r"\accept.csv", "w") as accept_fd:
        for ratio in acceptance:
            accept_fd.write(f"{ratio}\n")
    print("...finished.")


if __name__ == "__main__":
    ptmc_main()

