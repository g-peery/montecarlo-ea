"""
Methods to run simulations and output data to a file for later analysis.

At present, link and spin overlaps are calculated according to which
pairs of states are at the same indices, regardless of temperatures.
Will likely want to change after this one is made consistent.

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
from numba import boolean, njit, prange
import numpy as np
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
def calc_change(st : np.ndarray, c : np.ndarray) -> float:
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
        link_overlap : np.ndarray
    ):
    """For pairs of states in ss, calculate the overlaps index by index.
    Requires an array to store index transformations for reasons of
    parallelization.
    
    Effects:
        Modifies supplied overlap arrays.
    """
    size = len(ss[0][0])
    # Loop over indices into state array
    for si in prange(len(ss[0])):
        spin_overlap[si] = link_overlap[si] = 0
        for x in prange(size):
            for y in prange(size):
                for z in prange(size):
                    spin_overlap[si] += (
                        ss[0][si][x][y][z].s 
                        * ss[1][si][x][y][z].s
                    )
                    link_overlap[si] += _calc_sh_overlap(
                        ss[0][si], ss[1][si], x, y, z
                    )


@njit(parallel=True)
def mc_step(
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
        stb : np.ndarray
    ):
    """Given states ss, information to get their temperatures, and some
    random values to use, performs a Metropolis-Hastings Monte Carlo
    step on all of them and updates quantities accordingly. The cs array 
    is of random coordinates. Also requires arrays to keep booleans and 
    changes in energy for parallelization.

    Effects:
        May swap spins in states.
        Updates energies, spin_overlap, link_overlap.
        Modifies parallelization utility arrays
    """
    # Loop over indices into the states and qoi arrays
    for si in prange(len(ss[0])):
        # Check if non-positive
        changes[0][si] = calc_change(ss[0][si], cs[0][si])
        changes[1][si] = calc_change(ss[1][si], cs[1][si])
        do_swaps[0][si] = (changes[0][si] <= 0)
        do_swaps[1][si] = (changes[1][si] <= 0)

        # If positive, use the uniforms
        if not do_swaps[0][si]:
            do_swaps[0][si] = (uniforms[0][si] < np.exp(
                -betas[stb[0][si]] * changes[0][si]
            ))
        if not do_swaps[1][si]:
            do_swaps[1][si] = (uniforms[1][si] < np.exp(
                -betas[stb[1][si]] * changes[1][si]
            ))

        # Perform swaps, update energy and spin overlap info
        if do_swaps[0][si]:
            spin_overlap[si] -= 2 * (
                ss[0][si][cs[0][si][0]][cs[0][si][1]][cs[0][si][2]].s 
                * ss[1][si][cs[0][si][0]][cs[0][si][1]][cs[0][si][2]].s 
            )
            ss[0][si][cs[0][si][0]][cs[0][si][1]][cs[0][si][2]].s *= -1
            energies[0][si] += changes[0][si]
        if do_swaps[1][si]:
            spin_overlap[si] -= 2 * (
                ss[0][si][cs[1][si][0]][cs[1][si][1]][cs[1][si][2]].s
                * ss[1][si][cs[1][si][0]][cs[1][si][1]][cs[1][si][2]].s
            )
            ss[1][si][cs[1][si][0]][cs[1][si][1]][cs[1][si][2]].s *= -1
            energies[1][si] += changes[1][si]

        # Update link overlap
        if do_swaps[0][si] or do_swaps[1][si]:
            link_overlap[si] = calc_link_overlap(ss[0][si], ss[1][si])


@njit
def attempt_swap(
        bsi : np.ndarray, # Beta swap index i_beta
        energies : np.ndarray,
        betas : np.ndarray,
        bts : np.ndarray,
        stb : np.ndarray,
        swap_uniform : float
    ):
    """Given a random index at which to swap temperatures in the betas
    array, attempts a swap.
    """
    # Get indices into beta array of temperatures
    # that will swap
    low_i = bsi
    high_i = bsi + 1
    # Accept or reject based on boltzmann factor
    exponent = (
        -energies[bts[low_i]]*betas[high_i]
        - energies[bts[high_i]]*betas[low_i]
        + energies[bts[low_i]]*betas[low_i]
        + energies[bts[high_i]]*betas[high_i]
    )
    # Actual swapping
    if (exponent >= 1.0) or (swap_uniform <= np.exp(exponent)):
        # Update state->beta first
        stb[bts[low_i]] = high_i
        stb[bts[high_i]] = low_i
        # Then swap beta->state
        bts[low_i], bts[high_i] = bts[high_i], bts[low_i]


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
    spin_overlap = np.zeros(len(betas), dtype=BINDING_DTYPE)
    link_overlap = np.zeros(len(betas), dtype=BINDING_DTYPE)

    # Logistics, counts, useful variables, ephemera arrays
    do_swaps = np.zeros((2, len(betas)), dtype=boolean)
    changes = np.zeros((2, len(betas)), dtype=BINDING_DTYPE)
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

    # Generate randoms from start
    print("Generating randoms...")
    rand_coords = np.random.randint(0, size, size=(*mc_shape, 3))
    mc_uniforms = np.random.rand(*mc_shape)
    swap_idxs = np.random.randint(0, len(betas) - 1, size=swap_shape)
    swap_uniforms = np.random.rand(*swap_shape)
    print("...finished randoms.")

    # Output quantities of interest (qoi)
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
        init_rand_states(states)
        calc_energies(states, energies)
        calc_overlaps(states, spin_overlap, link_overlap)

        #
        # Sweeping loop
        #
        for swi in range(real_sweeps): # Sweep index
            # Metropolis-Hastings Monte-Carlo
            for ssi in range(site_count): # Sub-sweep index
                mc_step(
                    states,
                    betas,
                    bts,
                    rand_coords[sai][swi][ssi],
                    mc_uniforms[sai][swi][ssi],
                    do_swaps,
                    changes,
                    energies,
                    spin_overlap,
                    link_overlap,
                    stb
                )

            # Record data after equilibrium phase
            if swi >= equilibriate_sweeps:
                # Record overlaps for histogram later
                for bi in range(len(betas)):
                    spin_overlap_hist[bi][sai][swi] = (spin_overlap[bi])
                    link_overlap_hist[bi][sai][swi] = (link_overlap[bi])

            # Periodic Temperature Swap
            quo, rem = np.divmod(swi, global_move_period)
            if rem == 0:
                # Attempt swaps equal to number of betas
                for swt in range(len(betas)): # Swap trial
                    # Swap A
                    attempt_swap(
                            swap_idxs[sai][quo][swt][0],
                            energies[0],
                            betas,
                            bts[0],
                            stb[0],
                            swap_uniforms[sai][quo][swt][0]
                        )
                    # Swap B
                    attempt_swap(
                            swap_idxs[sai][quo][swt][1],
                            energies[1],
                            betas,
                            bts[1],
                            stb[1],
                            swap_uniforms[sai][quo][swt][1]
                        )

    return spin_overlap_hist, link_overlap_hist


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
    with open(out_name, "w") as fd:
        fd.write("Beta,SampleID,SweepID,SpinOverlap,LinkOverlap\n")
        for beta, s1, l1 in zip(betas, spin_overlap_hist, link_overlap_hist):
            for sai in range(len(s1)): # Sample index
                for swi in range(len(s1[sai])): # Sweep index
                    fd.write(
                        f"{beta:.3},{sai},{swi},{s1[sai][swi]},{l1[sai][swi]}\n"
                    );
    print("...finished writing to file.")


if __name__ == "__main__":
    ptmc_main()

