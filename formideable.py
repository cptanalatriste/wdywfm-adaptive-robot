import os
import random

import numpy as np
import pandas as pd
from typing import List, Tuple

from abm_statistics import ADAPTIVE_SUPPORT_COLUMN, NO_SUPPORT_COLUMN, ONLY_STAFF_SUPPORT_COLUMN, \
    ONLY_PASSENGER_SUPPORT_COLUMN
from netlogo_config import (ExperimentRun, ENABLE_STAFF_COMMAND, ENABLE_PASSENGER_COMMAND, SET_GENERATE_FRAMES_COMMAND,
                            SET_ENABLE_LOGGING_COMMAND)

SAMPLES = 50  # type: int
# SAMPLES = 1  # type: int
# FALL_LENGTHS = [minutes * 30 for minutes in range(1, 21)]  # type: List[int]
FALL_LENGTHS = [minutes * 30 for minutes in range(1, 2)]

# For the replication of the TOSEM experiments
NORMAL_STAFF_NUMBER = 8  # type: int
STAFF_NUMBER = 1  # type: int
PASSENGER_NUMBER = 800  # type: int
# RESULTS_FILE_PREFIX = "{}_fall_50_samples_experiment_results."

# For loading experiment config from CSV files.
USE_FORMIDEABLE_FILES = True  # type: bool

RESULTS_FILE_PREFIX = "exp_rvar_0.5_50_50_{}"  # type: str
RESULT_FILE_SUFFIX = "_from_FORMIDEABLE.csv"  # type: str
RANDOM_SEED_COLUMN = "random_seed"  # type: str
STAFF_NUMBER_COLUMN = "random_staff"  # type: str
PASSANGER_NUMBER_COLUMN = "random_pass"  # type: str

IDEA_RESULTS_FILE_PREFIX = "exp_rvar_0.5_50_50_{}_gambit_results"  # type: str
IDEA_RESULT_FILE_SUFFIX = "_from_IDEA.csv"  # type: str
RANDOM_SEED_COLUMN_FORMAT = "{}_seed"  # type: str
STAFF_NUMBER_COLUMN_FORMAT = "{}_staff"  # type: str
PASSANGER_NUMBER_COLUMN_FORMAT = "{}_passengers"  # type: str

DATA_DIRECTORY = "/home/cgc87/github/formal-robot-assisted-evacuation/workspace/data/"


SIMULATION_SCENARIOS = {
    NO_SUPPORT_COLUMN: [
        (SET_GENERATE_FRAMES_COMMAND.format("TRUE"), False),
        (SET_ENABLE_LOGGING_COMMAND.format("TRUE"), False),
    ],
    ONLY_STAFF_SUPPORT_COLUMN: [
        (ENABLE_STAFF_COMMAND, False),
        (SET_GENERATE_FRAMES_COMMAND.format("TRUE"), False),
        (SET_ENABLE_LOGGING_COMMAND.format("TRUE"), False)],
    ONLY_PASSENGER_SUPPORT_COLUMN: [
        (ENABLE_PASSENGER_COMMAND, False),
        (SET_GENERATE_FRAMES_COMMAND.format("TRUE"), False),
        (SET_ENABLE_LOGGING_COMMAND.format("TRUE"), False)],
    ADAPTIVE_SUPPORT_COLUMN: [
        (SET_GENERATE_FRAMES_COMMAND.format("TRUE"), False),
        (SET_ENABLE_LOGGING_COMMAND.format("TRUE"), False),
        (ENABLE_PASSENGER_COMMAND, False),
        (ENABLE_STAFF_COMMAND, False)]
}


def get_runs(experiment_name, commands_per_scenario, fall_length):
    # type: (str, List[Tuple[str, bool]], int) -> List[ExperimentRun]

    min_seed = -2147483648  # type: int
    max_seed = 2147483647  # type: int

    experiment_runs = []  # type: List[ExperimentRun]
    for simulation_id in range(SAMPLES):
        random_seed = random.randrange(min_seed, max_seed)
        experiment_runs.append(ExperimentRun(experiment_name=experiment_name,
                                             simulation_id=simulation_id,
                                             commands_per_scenario=commands_per_scenario,
                                             random_seed=random_seed,
                                             normal_staff_number=NORMAL_STAFF_NUMBER,
                                             staff_number=STAFF_NUMBER,
                                             passenger_number=PASSENGER_NUMBER,
                                             fall_length=fall_length))

    return experiment_runs


def get_runs_from_file(experiment_name, commands_per_scenario, fall_length):
    # type: (str, List[Tuple[str, bool]], int) -> List[ExperimentRun]

    prefix = RESULTS_FILE_PREFIX
    if not USE_FORMIDEABLE_FILES:
        prefix = IDEA_RESULTS_FILE_PREFIX

    data_file_format = DATA_DIRECTORY + prefix + ".csv"  # type: str
    path = data_file_format.format(fall_length)  # type: str
    print("Getting simulation parameters from {}".format(path))

    file_name, extension = os.path.splitext(path)  # type: Tuple
    name_tokens = file_name.split("_")  # type: List[str]

    if USE_FORMIDEABLE_FILES:
        fall_length_from_file = int(name_tokens[-1])  # type: int
        if fall_length != fall_length_from_file:
            raise ValueError("Provided length {} does not correspond to file {}".format(fall_length, file_name))

    data_frame = pd.read_csv(path)  # type: pd.DataFrame

    samples = SAMPLES  # type: int
    # samples = len(data_frame)  # type: int

    random_seed_column = RANDOM_SEED_COLUMN
    staff_number_column = STAFF_NUMBER_COLUMN
    passenger_number_column = PASSANGER_NUMBER_COLUMN
    if not USE_FORMIDEABLE_FILES:
        random_seed_column = RANDOM_SEED_COLUMN_FORMAT.format(experiment_name)
        staff_number_column = STAFF_NUMBER_COLUMN_FORMAT.format(experiment_name)
        passenger_number_column = PASSANGER_NUMBER_COLUMN_FORMAT.format(experiment_name)

    random_seeds = data_frame[random_seed_column].to_list()  # type: List[int]
    staff_number = data_frame[staff_number_column].to_list()  # type: List[int]
    passenger_number = data_frame[passenger_number_column].to_list()  # type: List[int]

    # Temporary workaround
    normal_staff_number = [1 for staff in range(1, samples + 1)]  # type: List[int]

    experiment_runs = []  # type: List[ExperimentRun]
    for simulation_id in range(samples):
        experiment_runs.append(ExperimentRun(experiment_name=experiment_name,
                                             simulation_id=simulation_id,
                                             commands_per_scenario=commands_per_scenario,
                                             random_seed=random_seeds[simulation_id],
                                             normal_staff_number=normal_staff_number[simulation_id],
                                             staff_number=staff_number[simulation_id],
                                             passenger_number=passenger_number[simulation_id],
                                             fall_length=fall_length))

    return experiment_runs


class NaiveTypeAnalyser(object):

    def __init__(self, shared_identity_probability):
        # type: (float) -> None
        self.shared_identity_probability = shared_identity_probability

    def obtain_probabilities(self, sensor_data):
        # type: (np.ndarray) -> np.ndarray
        return np.array([self.shared_identity_probability])
