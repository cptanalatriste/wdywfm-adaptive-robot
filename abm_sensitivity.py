import itertools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from matplotlib.axes import Axes
from typing import List, Dict, Tuple

from abm_analysis import SIMULATION_SCENARIOS, run_parallel_simulations, PLOT_STYLE, \
    SET_FALL_LENGTH_COMMAND
from abm_statistics import ADAPTIVE_SUPPORT_COLUMN

EVACUATION_TIME_COLUMN = "evacuation_time"

PASSENGER_BONUS_COLUMN = "passenger_bonus"  # type:str
FALL_LENGTH_COLUMN = "fall_length"  # type:str

FALL_LENGTHS = range(30, 630, 30)  # type: List[int]
PASSENGER_BONUS = range(2, 42, 2)  # type: List[int]
SENSITIVITY_SAMPLES = 30  # type: int

# FALL_LENGTHS = [60, 600]  # type: List[int]
# PASSENGER_BONUS = [5, 50]  # type: List[int]
# SENSITIVITY_SAMPLES = 30  # type: int

SENSITIVITY_DATA_FILE = "data/sensitivity_analysis.csv"  # type:str

SET_PASSENGER_BONUS_COMMAND = "set ROBOT_REQUEST_BONUS {}"  # type:str


def get_heatmap(annotated=False, figure_size=(10, 7)):
    # type: (bool, Tuple) -> None
    experiment_data = pd.read_csv(SENSITIVITY_DATA_FILE, index_col=[0])  # type: pd.DataFrame
    experiment_data = experiment_data.dropna()

    consolidated_data = experiment_data.groupby(
        [PASSENGER_BONUS_COLUMN, FALL_LENGTH_COLUMN]).mean()  # type: pd.DataFrame

    heatmap_data = consolidated_data.pivot_table(values=EVACUATION_TIME_COLUMN,
                                                 index=FALL_LENGTH_COLUMN,
                                                 columns=PASSENGER_BONUS_COLUMN)

    print(heatmap_data)

    _, axes = plt.subplots(figsize=figure_size)
    seaborn.heatmap(heatmap_data, annot=annotated, fmt=".0f", cmap="YlGnBu", ax=axes)  # type: Axes
    axes.set(xlabel="Helping Effect Increase (%)", ylabel="Fall length (s)")
    new_x_labels = [int(100 * float(label.get_text())) for label in axes.get_xticklabels()]  # type: List[int]
    axes.set_xticklabels(new_x_labels)

    plt.savefig("img/sensitivity_analysis.eps", format="eps", bbox_inches='tight', pad_inches=0)
    plt.savefig("img/sensitivity_analysis.png", format="png", bbox_inches='tight', pad_inches=0)

    plt.show()


def generate_data_for_analysis():
    sensitivity_data = []  # type: List[Dict[str, float]]

    for fall_length, passenger_bonus in itertools.product(FALL_LENGTHS, PASSENGER_BONUS):

        passenger_bonus = passenger_bonus / 100.0  # type:float
        print("fall_length {} passenger_bonus {}".format(fall_length, passenger_bonus))

        post_setup_commands = SIMULATION_SCENARIOS[ADAPTIVE_SUPPORT_COLUMN]  # type: List[str]
        post_setup_commands.append(SET_FALL_LENGTH_COMMAND.format(fall_length))
        post_setup_commands.append(SET_PASSENGER_BONUS_COMMAND.format(passenger_bonus))

        scenario_times = run_parallel_simulations(SENSITIVITY_SAMPLES,
                                                  setup_commands=post_setup_commands)  # type: List[float]
        for evacuation_time in scenario_times:
            sensitivity_data.append({FALL_LENGTH_COLUMN: fall_length,
                                     PASSENGER_BONUS_COLUMN: passenger_bonus,
                                     EVACUATION_TIME_COLUMN: evacuation_time})

    results_dataframe = pd.DataFrame(sensitivity_data)
    results_dataframe.to_csv(SENSITIVITY_DATA_FILE)

    print("Results written to {}".format(SENSITIVITY_DATA_FILE))


if __name__ == "__main__":
    plt.style.use(PLOT_STYLE)
    # generate_data_for_analysis()
    get_heatmap(annotated=True)
    get_heatmap(annotated=False)

