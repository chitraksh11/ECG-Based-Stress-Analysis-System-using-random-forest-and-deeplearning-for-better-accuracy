import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_stress_levels_over_time(stress_data, time_period='daily'):
    """
    Generates a line plot showing the user's stress levels over time.
    
    :param stress_data: A DataFrame containing 'date' and 'stress_level' columns.
    :param time_period: 'daily' for daily stress levels, 'historical' for a broader time range.
    """
    try:
        plt.figure(figsize=(10, 6))

        if time_period == 'daily':
            plt.title('Daily Stress Levels')
            sns.lineplot(x='date', y='stress_level', data=stress_data, marker='o')
        elif time_period == 'historical':
            plt.title('Historical Stress Levels')
            sns.lineplot(x='date', y='stress_level', data=stress_data)

        plt.xlabel('Date')
        plt.ylabel('Stress Level')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        logging.info(f"Successfully plotted {time_period} stress levels.")
    except Exception as e:
        logging.error(f"Failed to plot {time_period} stress levels.", exc_info=True)