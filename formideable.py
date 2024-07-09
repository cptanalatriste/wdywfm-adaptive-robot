import os

import numpy as np
import pandas as pd
from typing import List, Tuple

from abm_statistics import ADAPTIVE_SUPPORT_COLUMN, NO_SUPPORT_COLUMN
from netlogo_config import ExperimentRun, ENABLE_STAFF_COMMAND, ENABLE_PASSENGER_COMMAND

RESULTS_FILE_PREFIX = "exp_rvar_0.5_50_50_{}"
DATA_FILE_FORMAT = "formideable_results/" + RESULTS_FILE_PREFIX + ".csv"  # type: str

# Temporary placeholders. Fall length and samples are extracted from filename
FALL_LENGTHS = [30, 60]  # type: List[int]
SAMPLES = 0  # type:int

RANDOM_SEED_COLUMN = "random_seed"  # type: str
STAFF_NUMBER_COLUMN = "random_staff"  # type: str
PASSANGER_NUMBER_COLUMN = "random_pass"  # type: str

SIMULATION_SCENARIOS = {
    NO_SUPPORT_COLUMN: [],
    ADAPTIVE_SUPPORT_COLUMN: [
        # (SET_GENERATE_FRAMES_COMMAND.format("TRUE"), False),
        # (SET_ENABLE_LOGGING_COMMAND.format("TRUE"), False),
        (ENABLE_PASSENGER_COMMAND, False),
        (ENABLE_STAFF_COMMAND, False)]
}


def get_runs_from_file(commands_per_scenario, fall_length):
    # type: (List[Tuple[str, bool]], int) -> List[ExperimentRun]

    path = DATA_FILE_FORMAT.format(fall_length)  # type: str
    print("Getting simulation parameters from {}".format(path))

    file_name, extension = os.path.splitext(path)  # type: Tuple
    name_tokens = file_name.split("_")  # type: List[str]

    fall_length_from_file = int(name_tokens[-1])  # type: int
    if fall_length != fall_length_from_file:
        raise ValueError("Provided length {} does not correspond to file {}".format(fall_length, file_name))

    data_frame = pd.read_csv(path)  # type: pd.DataFrame

    samples = len(data_frame)  # type: int
    random_seeds = data_frame[RANDOM_SEED_COLUMN].to_list()  # type: List[int]
    staff_number = data_frame[STAFF_NUMBER_COLUMN].to_list()  # type: List[int]
    passenger_number = data_frame[PASSANGER_NUMBER_COLUMN].to_list()  # type: List[int]

    # Temporary workaround
    normal_staff_number = [1 for staff in range(1, samples + 1)]  # type: List[int]

    experiment_runs = []  # type: List[ExperimentRun]
    for simulation_id in range(samples):
        experiment_runs.append(ExperimentRun(simulation_id=simulation_id, commands_per_scenario=commands_per_scenario,
                                             random_seed=random_seeds[simulation_id],
                                             normal_staff_number=normal_staff_number[simulation_id],
                                             staff_number=staff_number[simulation_id],
                                             passenger_number=passenger_number[simulation_id],
                                             fall_length=fall_length_from_file))

    return experiment_runs


class NaiveTypeAnalyser(object):

    def __init__(self, shared_identity_probability):
        # type: (float) -> None
        self.shared_identity_probability = shared_identity_probability

    def obtain_probabilities(self, sensor_data):
        # type: (np.ndarray) -> np.ndarray
        return np.array([self.shared_identity_probability])
