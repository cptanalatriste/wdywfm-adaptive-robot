import logging
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
from typing import Dict, Tuple

import abm_gamemodel
from abm_trainer import CALIBRATION_SENSOR_DATA_FILE, CALIBRATION_PERSON_TYPE_FILE
from analyser import NeuralNetworkTypeAnalyser, CalibratedTypeAnalyser
from formideable import NaiveTypeAnalyser
from controller import AutonomicManagerController
from environment import NetlogoEvacuationEnvironment
from prob_calibration import get_calibrated_model
from synthetic_runner import TYPE_ANALYSER_MODEL_FILE, ENCODER_FILE

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Running inference on CPU

USE_NAIVE_ANALYSER = True  # type: bool
PROJECT_DIRECTORY = "/home/cgc87/github/wdywfm-adaptive-robot/"  # type:str


def run_scenario(robot_controller, emergency_environment):
    # type: ( AutonomicManagerController,  NetlogoEvacuationEnvironment) -> Tuple[str, float]

    current_sensor_data = emergency_environment.reset()  # type: np.ndarray

    model_filename = "efg/simulation_{}_game_model.efg".format(emergency_environment.simulation_id)  # type:str

    robot_controller.measure_distance(emergency_environment)
    robot_action, identity_probability = robot_controller.sensor_data_callback(current_sensor_data,
                                                                               model_filename)  # type:Tuple[str, float]

    logging.debug("robot_action {} identity_probability {}".format(robot_action, identity_probability))

    return robot_action, identity_probability


def get_calibrated_analyser():
    # type: () -> CalibratedTypeAnalyser
    base_type_analyser = NeuralNetworkTypeAnalyser(
        model_file=PROJECT_DIRECTORY + TYPE_ANALYSER_MODEL_FILE)  # type: NeuralNetworkTypeAnalyser
    type_analyser, _, _ = get_calibrated_model(base_type_analyser,
                                               PROJECT_DIRECTORY + CALIBRATION_SENSOR_DATA_FILE,
                                               PROJECT_DIRECTORY + CALIBRATION_PERSON_TYPE_FILE,
                                               method="isotonic")  # type: CalibratedTypeAnalyser
    return type_analyser


def get_naive_analyser():
    # type: () -> NaiveTypeAnalyser
    return NaiveTypeAnalyser(shared_identity_probability=0.8)


def main():
    parser = ArgumentParser("Get a robot action from the adaptive controller",
                            formatter_class=ArgumentDefaultsHelpFormatter)  # type: ArgumentParser
    parser.add_argument("simulation_id")

    parser.add_argument("helper_gender")
    parser.add_argument("helper_culture")
    parser.add_argument("helper_age")
    parser.add_argument("fallen_gender")
    parser.add_argument("fallen_culture")
    parser.add_argument("fallen_age")
    parser.add_argument("helper_fallen_distance")
    parser.add_argument("staff_fallen_distance")

    arguments = parser.parse_args()
    configuration = vars(arguments)  # type:Dict

    # This is the type analyser used for TOSEM
    type_analyser = get_calibrated_analyser()

    # This is the type analyser used with constant probability.
    if USE_NAIVE_ANALYSER:
        type_analyser = get_naive_analyser()
    robot_controller = AutonomicManagerController(type_analyser,
                                                  abm_gamemodel.generate_game_model)

    emergency_environment = \
        NetlogoEvacuationEnvironment(configuration,
                                     PROJECT_DIRECTORY + ENCODER_FILE)  # type: NetlogoEvacuationEnvironment

    robot_action, identity_probability = run_scenario(robot_controller, emergency_environment)  # type:Tuple[str, float]
    print("{} (Prob={})".format(robot_action, identity_probability))


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    # logging.basicConfig(level=logging.INFO)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    main()
