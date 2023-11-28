import subprocess
from typing import Optional

import click

from .parallel import calculate_mpi_and_parallel_params


@click.command()
@click.option("--run", type=click.Path(exists=True), required=False)
def run_flow_funnel(run: Optional[str]) -> None:
    """
    Execute a distributed parallel Python program using MPI.

    This function takes a path to a Python file and runs it in a distributed environment using MPI.
    The number of processes (`mpirun_np`) and the number of jobs (`n_jobs`) are calculated by
    `calculate_mpi_and_parallel_params`.

    Args:
        run (Optional[str]): The file path of the Python script to be run. If not provided, an error message
                             is displayed and the program exits.

    Returns:
        None: This function does not return anything but executes the MPI command with subprocess.
    """
    if run is None:
        click.echo(
            "Error: Please provide a Python file path to run. Usage: flowfunnel --run [PYTHON_FILE_PATH]"
        )
    else:
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
