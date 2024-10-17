from collections import deque
from datetime import datetime, timezone
from dataclasses import asdict
import itertools
import json
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Batch, EstimatorV2 as Estimator, QiskitRuntimeService, RuntimeEncoder
from qiskit_ibm_runtime.options.utils import remove_dict_unset_values
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from circuit import ExampleCircuit

def create_logical_circuit(num_qubits, depth, seed=0):
    logical_circuit = ExampleCircuit(num_qubits, depth // 2)  # Halved depth
            
    # Compute-uncompute construct (doubles the depth)
    inverse = logical_circuit.inverse()
    logical_circuit.barrier()
    logical_circuit.compose(inverse, inplace=True)

    # Parameter values
    rng = np.random.default_rng(seed=seed)
    parameter_values = rng.uniform(-np.pi, np.pi, size=logical_circuit.num_parameters)
    parameter_values[0] = 0.3  # Fix interaction strength (specific to MBL circuit)
    logical_circuit.assign_parameters(parameter_values, inplace=True)

    return logical_circuit, parameter_values

def create_wt2_logical_operator(num_qubits):
    wt2 = deque(["I"] * (num_qubits - 2) + ["Z", "Z"])
    logical_observables = []
    
    coeff = 1/num_qubits
    for _ in range(num_qubits):
        wt2_term = "".join(literal for literal in wt2)
        logical_observables.append((wt2_term, coeff))
        wt2.rotate(-1)

    return SparsePauliOp.from_list(logical_observables)

def heatmap_plotter(
        rel_err,
        widths,
        depths,
        filename,
        title,
        directory,
        xlabel="2-qubit depth",
        ylabel="Number of qubits",
):
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
    nrows = len(widths)
    ncols = len(depths)

    ax = sns.heatmap(rel_err, annot=True, cbar=False, vmin=0, vmax=100, cmap="binary", fmt=".0f")
    # Drawing the frame
    ax.axhline(y=0, color='k', linewidth=2) 
    
    ax.axhline(y=nrows, color='k', linewidth=2) 
    
    ax.axvline(x=0, color='k', linewidth = 2) 
    
    ax.axvline(x=ncols, color='k', linewidth=2) 

    xticks = np.array(list(range(ncols))) + 0.5
    ax.set_xticks(ticks=xticks)
    ax.set_xticklabels(labels=depths)

    yticks = np.array(list(range(nrows))) + 0.5
    ax.set_yticks(ticks=yticks)
    ax.set_yticklabels(labels=widths[::-1])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.savefig(
        f"{directory}/{filename}",
        bbox_inches="tight",
        dpi=200,
    )

    plt.close()

if __name__ == "__main__":
    num_qubits = [4, 8, 16]
    depths = [4, 8, 16, 32, 64]
    extrapolators = ["exponential", "polynomial_degree_2", "linear", ("exponential", "linear")]
    
    # service = QiskitRuntimeService()
    # backend = service.least_busy(min_num_qubits=127)
    # backend = FakeSherbrooke()
    backend = AerSimulator(method="matrix_product_state")

    # create all circuits and observables
    logical_circuits = []
    logical_observables = []
    circuit_parameters = []
    for q in num_qubits:
        for d in depths:
            logical_circuit, parameter_values = create_logical_circuit(num_qubits=q, depth=d, seed=0)
            logical_circuits.append(logical_circuit)
            circuit_parameters.append(parameter_values.tolist())

            logical_observable = create_wt2_logical_operator(num_qubits=q)
            logical_observables.append(logical_observable)

    # optimize circuit and observables
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    physical_circuits = pm.run(logical_circuits)

    physical_observables = [
        logical_observables[idx].apply_layout(layout=physical_circuits[idx].layout)
        for idx in range(len(logical_observables))
    ]

    pubs = list(zip(physical_circuits, physical_observables))

    timestamp = datetime.now(timezone.utc)
    estimator_options = {}
    with Batch(backend=backend) as batch:
        estimator = Estimator(mode=batch)
        options = estimator.options
        options.default_shots = 1000
        options.optimization_level = 0
        options.resilience_level = 0
        options.resilience.zne_mitigation = True  # Activates ZNE error mitigation only
        options.resilience.measure_mitigation = True # Activates measurement error mitigation
        
        jobs = {}
        for extrapolator in extrapolators:
            options.resilience.zne.extrapolator = extrapolator
            job = estimator.run(pubs)
            jobs[str(extrapolator)] = job
            options_dict = asdict(options)
            remove_dict_unset_values(options_dict)
            estimator_options[str(extrapolator)] = options_dict
            print(f"  - {job.job_id()} ({extrapolator})")


    results = {}
    tmp = list(itertools.product(num_qubits, depths))
    expvals_all = {str(extrapolator): np.empty(len(tmp)) for extrapolator in extrapolators}
    for idx, (qubit, depth) in enumerate(tmp):
        label = f"pub_{idx}"
        results[label] = {}
        results[label]["num_qubits"] = qubit
        results[label]["depth"] = depth
        
        # physical circuit is encoded using `RuntimeEncoder`. Use `RuntimeDecoder` to decode
        encoded_physical_circuit = RuntimeEncoder().encode(physical_circuits[idx])
        results[label]["physical_circuit"] = encoded_physical_circuit

        obs = physical_observables[idx]
        paulis = obs.paulis.to_labels()
        coeffs = [str(coeff) for coeff in obs.coeffs]
        results[label]["physical_observable"] = {
            "paulis": paulis,
            "coeffs": coeffs
        }
        
        # layout
        layout = physical_circuits[idx].layout
        final_layout = None if layout is None else layout.final_index_layout(filter_ancillas=True)
        results[label]["final_qubit_layout"] = final_layout
        results[label]["circuit_parameters"] = circuit_parameters[idx]
        
        results[label]["expvals"] = {}
        results[label]["job_ids"] = {}

        for extrapolator in extrapolators:
            extrapolator = str(extrapolator)
            job = jobs[extrapolator]
            
            primtive_results = job.result()
            results[label]["job_ids"][extrapolator] = job.job_id()

            pub_result = primtive_results[idx]
            evs = pub_result.data.evs.tolist()
            expvals_all[extrapolator][idx] = evs

            results[label]["expvals"][extrapolator] = evs
            
    all_data = {
        "num_qubits_list": num_qubits,
        "depths": depths,
        "extrapolators": extrapolators,
        "estimator_options": estimator_options,
        "batch_id": batch.session_id,
        "job_ids": [job.job_id() for job in jobs.values()],
        "backend_name": backend.name,
        "timestamp": timestamp.isoformat(),
        "results": results
    }
    
    
    name_signature = f"challenge_3b_{timestamp.strftime('%Y_%m_%d_%H_%M_%S_%f')}"
    name_signature = "3b"
    if not os.path.exists(name_signature):
        os.mkdir(name_signature)

    with open(f"{name_signature}/data_{name_signature}.json", "w") as jf:
        json.dump(all_data, jf, indent=2, sort_keys=False)
    
    # plot heatmaps
    for extrapolator in extrapolators:
        extrapolator_str = str(extrapolator)
        
        evs = np.flipud(
            expvals_all[extrapolator_str].reshape((len(num_qubits), len(depths)))
        )
        rel_err = 100 * (1 - evs)
        
        if isinstance(extrapolator, str):
            extrapolator = [extrapolator]
        
        heatmap_plotter(
            rel_err,
            widths=num_qubits,
            depths=depths,
            filename=f"rel_err_{name_signature}_{'_'.join(extrapolator)}.png",
            title=r'Error (\%): ' + r'$\overline{\langle{Z_{q}}\rangle}$' + f' {extrapolator}',
            directory=name_signature,
            xlabel="2-qubit depth",
            ylabel="Number of qubits",
        )
