from datetime import datetime, timedelta
from typing import List, Tuple


def generate_week_pairs(
    start_date: str, end_date: str, window_size: int, interval: int
) -> List[Tuple[str, str]]:
    """
    Generate a list of date pairs between start_date and end_date as week-like windows.

    Week-like windows are defined by window_size, which denotes the number of days for each date range.
    The next window starts after the interval days from the start of the previous window.
    The last window may be shorter if it reaches the end_date.

    Args:
        start_date: The starting date in 'YYYYMMDD' format.
        end_date: The ending date in 'YYYYMMDD' format, ensuring no window exceeds this date.
        window_size: The size of the window, representing a week or other period, in days.
        interval: The number of days to skip before beginning the next window after one ends.

    Returns:
        A list of tuples, each with start and end dates of a window.

    Raises:
        ValueError: If start_date is not before end_date.
    """

    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    current = start
    pairs = []

    while True:
        week_end = min(current + timedelta(days=window_size - 1), end)
        pairs.append((current.strftime("%Y%m%d"), week_end.strftime("%Y%m%d")))

        if week_end == end:
            break

        current += timedelta(days=interval)

    return pairs
