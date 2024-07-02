import logging
import pickle

import numpy as np
import pandas as pd
from numpy import save
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from typing import List, Tuple, Optional

from abm_analysis import run_parallel_simulations
from netlogo_config import SET_STAFF_SUPPORT_COMMAND, SET_PASSENGER_SUPPORT_COMMAND, SET_FALL_LENGTH_COMMAND, \
    SET_ENABLE_LOGGING_COMMAND
from analyser import NeuralNetworkTypeAnalyser
from prob_calibration import start_probability_calibration
from synthetic_runner import encode_training_data, TYPE_ANALYSER_MODEL_FILE, ENCODER_FILE

MAX_EPOCHS = 500  # type: int

EARLY_STOPPING_PATIENCE = int(MAX_EPOCHS * 0.10)  # type: int
TRAINING_BATCH_SIZE = 2048  # type: int
LEARNING_RATE = 0.001  # type: float
# UNITS_PER_LAYER = [16, 16]  # type: List[int]
UNITS_PER_LAYER = None  # For plain Logistic Regression

TRAINING_DATA_DIRECTORY = "data/training"
CALIBRATION_SENSOR_DATA_FILE = "{}/sensor_data_validation.npy".format(TRAINING_DATA_DIRECTORY)
CALIBRATION_PERSON_TYPE_FILE = "{}/person_type_validation.npy".format(TRAINING_DATA_DIRECTORY)
NETLOGO_DATA_FILE_PREFIX = "request-for-help-results"  # type:str
REQUEST_RESULT_COLUMN = "offer-help"  # type:str

SIMULATION_RUNS = 100  # type:int
FALL_LENGTH = 30  # type:int

ENABLE_DATA_COLLECTION_COMMAND = "set ENABLE_DATA_COLLECTION TRUE"  # type:str
DISABLE_FRAME_GENERATION_COMMAND = "  set ENABLE_FRAME_GENERATION FALSE"  # type:str

CONFIGURATION_COMMANDS = [
    SET_STAFF_SUPPORT_COMMAND.format("FALSE"),
    SET_PASSENGER_SUPPORT_COMMAND.format("FALSE"),
    SET_FALL_LENGTH_COMMAND.format(FALL_LENGTH),
    ENABLE_DATA_COLLECTION_COMMAND,
    SET_ENABLE_LOGGING_COMMAND.format("FALSE"),
    DISABLE_FRAME_GENERATION_COMMAND,
]  # type: List[str]


def get_netlogo_dataset():
    # type:() -> Tuple[np.ndarray, np.ndarray]

    dataframes = [pd.read_csv("{}/{}_{}.csv".format(TRAINING_DATA_DIRECTORY,
                                                    dataframe_index,
                                                    NETLOGO_DATA_FILE_PREFIX))
                  for dataframe_index in range(0, 97)]  # type: List[pd.DataFrame]

    netlogo_dataframe = pd.concat(dataframes, axis=0)  # type: pd.DataFrame

    netlogo_sensor_data = netlogo_dataframe.drop(REQUEST_RESULT_COLUMN, axis=1)  # type: pd.DataFrame
    netlogo_person_type = netlogo_dataframe[REQUEST_RESULT_COLUMN]  # type: pd.DataFrame

    return netlogo_sensor_data.values, netlogo_person_type.values


def generate_training_data(simulation_runs=None, configuration_commands=None):
    # type: (int, List[str]) -> None
    print("Generating training data from {} simulation runs".format(SIMULATION_RUNS))
    _ = run_parallel_simulations(simulation_runs, configuration_commands, gui=False)


def start_training(max_epochs, training_batch_size, learning_rate, units_per_layer,
                   early_stopping_patience):
    # type: (int, int,float, Optional[List[int]], int) -> None

    target_accuracy = None
    under_sample = False  # type: bool
    calculate_weights = True  # type: bool
    number_of_bins = 20  # type: int
    # calibration_method = "sigmoid"  # type:str
    calibration_method = "isotonic"  # type:str

    sensor_data, person_type = get_netlogo_dataset()  # type: Tuple[np.ndarray, np.ndarray]
    sensor_data_training, sensor_data_test, person_type_training, person_type_test = train_test_split(sensor_data,
                                                                                                      person_type,
                                                                                                      test_size=0.33,
                                                                                                      stratify=person_type,
                                                                                                      random_state=0)
    sensor_data_training = encode_training_data(sensor_data_training)  # type:np.ndarray

    sensor_data_training, sensor_data_validation, person_type_training, person_type_validation = train_test_split(
        sensor_data_training,
        person_type_training,
        stratify=person_type_training,
        test_size=0.33,
        random_state=0)

    # _ = train_type_analyser(sensor_data_training, person_type_training,
    #                         sensor_data_validation, person_type_validation,
    #                         training_batch_size, target_accuracy,
    #                         units_per_layer, max_epochs,
    #                         learning_rate=learning_rate,
    #                         patience=early_stopping_patience,
    #                         balance_data=under_sample,
    #                         calculate_weights=calculate_weights)  # type: SyntheticTypeAnalyser

    type_analyser = NeuralNetworkTypeAnalyser(model_file=TYPE_ANALYSER_MODEL_FILE)

    with open(ENCODER_FILE, "rb") as encoder_file:
        encoder = pickle.load(encoder_file)  # type: OneHotEncoder
        sensor_data_test = encoder.transform(sensor_data_test)
        logging.info("Test dataset inputs encoded")

    save(CALIBRATION_SENSOR_DATA_FILE, sensor_data_validation)
    save(CALIBRATION_PERSON_TYPE_FILE, person_type_validation)
    logging.info("Data for calibration fitting saved at {} and {}".format(CALIBRATION_SENSOR_DATA_FILE,
                                                                          CALIBRATION_PERSON_TYPE_FILE))

    start_probability_calibration(TYPE_ANALYSER_MODEL_FILE, CALIBRATION_SENSOR_DATA_FILE, CALIBRATION_PERSON_TYPE_FILE,
                                  sensor_data_test, person_type_test, number_of_bins, calibration_method)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # generate_training_data(SIMULATION_RUNS, CONFIGURATION_COMMANDS)
    start_training(MAX_EPOCHS, TRAINING_BATCH_SIZE, LEARNING_RATE, UNITS_PER_LAYER,
                   EARLY_STOPPING_PATIENCE)
