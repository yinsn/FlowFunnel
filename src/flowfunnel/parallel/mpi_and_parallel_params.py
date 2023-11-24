from .logical_processors import get_logical_processors_count


def calculate_mpi_and_parallel_params() -> tuple:
    """
    Calculate parameters for MPI and Parallel execution based on a given even number.

    This function finds two factors of the given even number that are closest to each other.
    The larger factor is suitable for the 'mpirun -np' parameter (number of MPI processes),
    and the smaller factor is suitable for the 'n_jobs' parameter in Parallel.

    Args:
    even_number (int): A positive even number to be factorized for MPI and Parallel parameters.

    Returns:
    tuple: A tuple of two integers, where the first integer (larger) is recommended for 'mpirun -np',
           and the second integer (smaller) for 'Parallel' 'n_jobs'.
    """
    even_number = get_logical_processors_count()
    if even_number % 2 != 0:
        raise ValueError("The number must be an even number.")

    for i in range(int(even_number**0.5), 0, -1):
        if even_number % i == 0:
            return (even_number // i, i)

    return (even_number, 1)
