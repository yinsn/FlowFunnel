from datetime import datetime, timedelta
from typing import Dict, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from ..dataloaders import generate_week_pairs


class RollingWindowVisualizer:
    """
    A visualizer for rolling window data analysis.

    This class plots trends and rates from given rolling window data over a specified time range.

    Attributes:
        rolling_data (Dict[str, np.ndarray]): A dictionary where keys are data labels and values are numpy arrays of data.
        window_size (int): The size of the rolling window.
        interval (Optional[int]): The interval between each window. Defaults to half of the window_size if None.
        weeks (List[Tuple[str, str]]): Pairs of start and end dates for each window.
        start_date (datetime): The starting date of the data range.
        end_date (datetime): The ending date of the data range.
        date_range (List[datetime]): List of dates in the specified range at the given interval.

    Args:
        rolling_data (Dict[str, np.ndarray]): A dictionary containing rolling data.
        start_date (str): The starting date in 'YYYYMMDD' format.
        end_date (str): The ending date in 'YYYYMMDD' format.
        window_size (int): The size of the rolling window.
        interval (Optional[int]): The interval between each window.
    """

    def __init__(
        self,
        rolling_data: Dict[str, np.ndarray],
        start_date: str,
        end_date: str,
        window_size: int,
        interval: Optional[int] = None,
    ) -> None:
        self.rolling_data = rolling_data
        self.window_size = window_size
        if interval is None:
            self.interval = window_size // 2
        else:
            self.interval = interval
        self.weeks = generate_week_pairs(
            start_date=start_date,
            end_date=end_date,
            window_size=self.window_size,
            interval=self.interval,
        )
        self.start_date = datetime.strptime(self.weeks[0][0], "%Y%m%d")
        self.end_date = datetime.strptime(self.weeks[-1][0], "%Y%m%d")
        self.date_range = [
            self.start_date + timedelta(days=x)
            for x in range(0, (self.end_date - self.start_date).days, self.interval)
        ]

    def plot_growth_trend(self) -> None:
        """
        Plots the growth trend from the rolling data.

        This method visualizes the growth trends over time using the date range and interval specified in the class.
        """
        growth_trend_keys = [
            key for key in self.rolling_data.keys() if "growth_trend" in key
        ]
        growth_trends = {key: self.rolling_data[key] for key in growth_trend_keys}

        plt.figure(figsize=(10, 5))
        for key in growth_trends:
            plt.plot(self.date_range, growth_trends[key], label=key)

        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=self.interval))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()

        plt.legend()
        plt.title("Growth Trends Over Time")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.show()

    def plot_transition_rate(self) -> None:
        """
        Plots the transition rate from the rolling data.

        This method visualizes the transition rates over time using the date range and interval specified in the class.
        """
        transition_rate_keys = [
            key for key in self.rolling_data.keys() if "transition_rate" in key
        ]
        transition_rates = {key: self.rolling_data[key] for key in transition_rate_keys}

        plt.figure(figsize=(10, 5))
        for key in transition_rates:
            plt.plot(self.date_range, transition_rates[key], label=key)

        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=self.interval))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()

        plt.legend()
        plt.title("Transition Rates Over Time")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.show()
