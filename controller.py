import logging

import numpy as np
from gambit import Rational
from gambit.nash import NashSolution
from typing import Optional, Tuple, List, Dict

import analyser
import solver
from abm_gamemodel import generate_game_model
from game import InteractionGame
from gamemodel import CALL_STAFF_ROBOT_ACTION, ASK_FOR_HELP_ROBOT_ACTION


class AbstractRobotController(object):

    def sensor_data_callback(self, sensor_data):
        raise NotImplementedError("Subclasses must override sensor_data_callback")


class ProSelfRobotController(AbstractRobotController):

    def sensor_data_callback(self, sensor_data):
        return CALL_STAFF_ROBOT_ACTION


class ProSocialRobotController(AbstractRobotController):

    def sensor_data_callback(self, sensor_data):
        return ASK_FOR_HELP_ROBOT_ACTION


class AutonomicManagerController(AbstractRobotController):

    def __init__(self, type_analyser, model_generator):
        self.type_analyser = type_analyser  # type: analyser.NeuralNetworkTypeAnalyser
        self.external_solver = solver.ExternalSubGamePerfectSolver()  # type: solver.ExternalSubGamePerfectSolver
        self.interaction_game = None  # type: Optional[InteractionGame]

        self.model_generator = model_generator  # type: generate_game_model

        self.robot_payoff_with_support = None  # type: Optional[Rational]
        self.robot_payoff_call_staff = None  # type: Optional[Rational]

    def get_shared_identity_probability(self, sensor_data):
        # type: (np.ndarray) -> float

        type_probabilities = self.type_analyser.obtain_probabilities(sensor_data)  # type: np.ndarray
        shared_identity_prob = type_probabilities.item()  # type: float

        return shared_identity_prob

    def sensor_data_callback(self, sensor_data, model_filename=None):
        # type: (np.ndarray, Optional[str]) -> Tuple[Optional[str], float]

        group_identity_prob = self.get_shared_identity_probability(sensor_data)  # type: float
        logging.info("group_identity_prob :  %.4f " % group_identity_prob)
        logging.info("self.robot_payoff_with_support {}".format(self.robot_payoff_with_support))
        logging.info("self.robot_payoff_call_staff {}".format(self.robot_payoff_call_staff))

        self.model_interaction(zero_responder_prob=group_identity_prob, filename=model_filename)
        equilibria = self.external_solver.solve(self.interaction_game.game_tree)  # type: List[NashSolution]

        if len(equilibria) == 0:
            logging.warning("No equilibria found! Aborting")
            return None, group_identity_prob

        if len(equilibria) > 1:
            logging.warning("Multiple equilibria found! Aborting")
            return None, group_identity_prob
        strategy_profile = equilibria[0]  # type: NashSolution

        robot_strategy = self.interaction_game.get_robot_strategy(strategy_profile)  # type: Dict[str, float]
        robot_action = max(robot_strategy, key=robot_strategy.get)  # type: str

        return robot_action, group_identity_prob

    def model_interaction(self, zero_responder_prob, filename):
        # type: (float, Optional[str]) -> None

        zero_responder_ratio = zero_responder_prob.as_integer_ratio()  # type: Tuple [int, int]
        selfish_ratio = (1 - zero_responder_prob).as_integer_ratio()  # type: Tuple [int, int]

        self.interaction_game = self.model_generator(zero_responder_ratio=zero_responder_ratio,
                                                     selfish_ratio=selfish_ratio,
                                                     robot_payoff_with_support=self.robot_payoff_with_support,
                                                     robot_payoff_call_staff=self.robot_payoff_call_staff,
                                                     filename=filename)

    @staticmethod
    def get_robot_payoff_with_support(staff_fallen_distance, helper_fallen_distance):
        if staff_fallen_distance < 0 or staff_fallen_distance > helper_fallen_distance:
            return Rational(3, 1)

        return Rational(2, 1)

    @staticmethod
    def get_robot_payoff_call_staff(staff_fallen_distance, helper_fallen_distance):
        if staff_fallen_distance < 0:
            return Rational(-1, 1)
        elif staff_fallen_distance > helper_fallen_distance:
            return Rational(0, 1)
        else:
            return Rational(1, 1)

    def measure_distance(self, environment):
        self.robot_payoff_with_support = self.get_robot_payoff_with_support(environment.staff_fallen_distance,
                                                                            environment.helper_fallen_distance)
        self.robot_payoff_call_staff = self.get_robot_payoff_call_staff(environment.staff_fallen_distance,
                                                                        environment.helper_fallen_distance)


def main():
    manager = AutonomicManagerController(analyser.NeuralNetworkTypeAnalyser(model_file="trained_model.h5"))
    sample_sensor_reading = np.zeros(shape=(1, 31))  # type: np.ndarray
    robot_action, identity_probability = manager.sensor_data_callback(sample_sensor_reading)
    print(robot_action)
    print(identity_probability)


if __name__ == "__main__":
    main()
