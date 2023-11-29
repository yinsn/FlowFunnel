import subprocess
from typing import Optional

import click

from .parallel import calculate_mpi_and_parallel_params


@click.command()
@click.option("--job", type=click.Path(exists=True), required=False)
@click.option("--mpi", type=click.Path(exists=True), required=False)
def run_flow_funnel(run: Optional[str], mpi: Optional[str]) -> None:
    """
    Execute a distributed parallel Python program using MPI.

    This function takes a path to a Python file and runs it in a distributed environment using MPI.
    The number of processes (`mpirun_np`) and the number of jobs (`n_jobs`) are calculated by
    `calculate_mpi_and_parallel_params`.

    Args:
        job (Optional[str]): The file path of the Python script to be run.
                             is displayed and the program exits.
        mpi (Optional[str]): The file path of the MPI script to be run. Use `use-hwthread-cpus`.
    """
    if mpi is not None:
        print(f"\nmpirun -np 1 python3 {mpi}")
        subprocess.run(
            ["mpirun", "--allow-run-as-root", "--use-hwthread-cpus", "python3", mpi]
        )
    if run is not None:
        mpirun_np, n_jobs = calculate_mpi_and_parallel_params()
        print(f"\nmpirun -np {mpirun_np} python3 {run} --n_jobs {n_jobs}")
        subprocess.run(
            [
                "mpirun",
                "--allow-run-as-root",
                "-np",
                str(mpirun_np),
                "python3",
                run,
                "--n_jobs",
                str(n_jobs),
            ]
        )


if __name__ == "__main__":
    run_flow_funnel()
