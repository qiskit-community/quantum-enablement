import numpy as np
import time
from json.encoder import JSONEncoder
from json.decoder import JSONDecoder

from qiskit_serverless import distribute_task, get_arguments, get, save_result
from qiskit_addon_sqd.subsampling import postselect_and_subsample
from qiskit_addon_sqd.configuration_recovery import recover_configurations

from qiskit_addon_sqd.fermion import solve_fermion

# --- fan‑out target: 1 CPU + 2 GB per call -------------
@distribute_task(target={"cpu": 1, "mem": 2 * 1024**3})
def _solve_fermion_worker(i0, i1, i2, i3, i4, i5, i6):
    energy_sci, coeffs_sci, avg_occs, spin = solve_fermion(i0, i1, i2, open_shell=i3, spin_sq=i4, max_davidson=i5)
    energy_sci += i6
    return (energy_sci, coeffs_sci, avg_occs, spin)

rng = np.random.default_rng(12345)

args                = get_arguments()
data                = args["data"]

iterations          = args["iterations"]
n_batches           = args["n_batches"] 
samples_per_batch   = args["samples_per_batch"] 
max_davidson_cycles = args["max_davidson_cycles"] 

i_data = JSONDecoder().decode(data)

(bitstring_matrix_full, probs_arr_full, hcore, eri, open_shell, spin_sq, max_davidson_cycles, nuclear_repulsion_energy, num_elec_a, num_elec_b) = i_data

bitstring_matrix_full = np.array(bitstring_matrix_full)
probs_arr_full        = np.array(probs_arr_full)
hcore                 = np.array(hcore)
eri                   = np.array(eri)

# Self-consistent configuration recovery loop 
e_hist = np.zeros((iterations, n_batches))  # energy history 
s_hist = np.zeros((iterations, n_batches))  # spin history 
occupancy_hist = []
avg_occupancy = None
for i in range(iterations):
    print(f"Starting configuration recovery iteration {i}")
    # On the first iteration, we have no orbital occupancy information from the
    # solver, so we just post-select from the full bitstring set based on hamming weight.
    if avg_occupancy is None:
        bs_mat_tmp = bitstring_matrix_full
        probs_arr_tmp = probs_arr_full

    # If we have average orbital occupancy information, we use it to refine the full set of noisy configurations
    else:
        bs_mat_tmp, probs_arr_tmp = recover_configurations(
            bitstring_matrix_full,
            probs_arr_full,
            avg_occupancy,
            num_elec_a,
            num_elec_b,
            rand_seed=rng,
        )

    # Throw out configurations with incorrect particle number in either the spin-up or spin-down systems
    batches = postselect_and_subsample(
        bs_mat_tmp,
        probs_arr_tmp,
        hamming_right=num_elec_a,
        hamming_left=num_elec_b,
        samples_per_batch=samples_per_batch,
        num_batches=n_batches,
        rand_seed=rng,
    )

    # Parallelize eigenstate solvers
    packages = [(batch, hcore, eri, open_shell, spin_sq, max_davidson_cycles, nuclear_repulsion_energy) for batch in batches]

    # fan‑out: spawn one worker per input tuple
    refs = [_solve_fermion_worker(*seven_inputs) for seven_inputs in packages]

    # fan‑in: block until every worker finishes
    batch_outputs = get(refs)                  

    e_tmp = np.zeros(n_batches)
    s_tmp = np.zeros(n_batches)
    occs_tmp = []
    coeffs = []
    for j, output in enumerate(batch_outputs):
        e_tmp[j] = output[0]
        s_tmp[j] = output[3]
        occs = np.array(output[2])
        occs_tmp.append(occs.tolist())
        coeffs.append(output[1])
 
    avg_occupancy = np.mean(occs_tmp, axis=0).tolist()

    # Track optimization history
    e_hist[i, :] = e_tmp
    s_hist[i, :] = s_tmp
    occupancy_hist.append(avg_occupancy)

o_data = (e_hist.tolist(), s_hist.tolist(), occupancy_hist)

# Encode JSON
out_e = JSONEncoder().encode(o_data)

# JSON-safe package
save_result({"outputs": out_e})     # single JSON blob returned to client
