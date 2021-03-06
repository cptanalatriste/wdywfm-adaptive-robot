import logging

import numpy as np
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from analyser import SyntheticTypeAnalyser
from controller import ProSelfRobotController, AutonomicManagerController, ProSocialRobotController
from environment import EmergencyEvacuationEnvironment
from samplegame import PERSONAL_IDENTITY_TYPE, GROUP_IDENTITY_TYPE

SEED = 0
NUM_SCENARIOS = 10
INTERACTIONS_PER_SCENARIO = 10

TYPE_TO_CLASS = {
    PERSONAL_IDENTITY_TYPE: 0,
    GROUP_IDENTITY_TYPE: 1
}


class EarlyStoppingByTarget(Callback):

    def __init__(self, monitor='val_acc', target=0.8, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.target = target
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current_value = logs.get(self.monitor)
        if current_value is None:
            logging.error("Early stopping requires %s available!" % self.monitor)
            raise RuntimeWarning()

        if current_value >= self.target:
            if self.verbose > 0:
                logging.info("Epoch %s: early stopping, target accuracy reached" % str(epoch))
            self.model.stop_training = True


def get_dataset(selfish_type_weight, zeroresponder_type_weight, total_samples):
    features, target = make_classification(n_samples=total_samples,
                                           n_features=100,
                                           n_informative=3,
                                           n_redundant=0,
                                           n_classes=2,
                                           weights=[selfish_type_weight, zeroresponder_type_weight],
                                           random_state=0)

    return features, target


def plot_training(training_history, metric):
    training_metric = training_history.history[metric]
    validation_metric = training_history.history["val_" + metric]

    epoch = range(1, len(training_metric) + 1)
    plt.plot(epoch, training_metric, "r--")
    plt.plot(epoch, validation_metric, "b-")

    plt.legend(["Training " + metric, "Validation " + metric])
    plt.xlabel("Epoch")
    plt.ylabel(metric)

    plt.show()


def get_type_analyser(sensor_data_train, person_type_train, batch_size, target_accuracy, epochs=100):
    logging.info("Training data: : %.4f" % len(sensor_data_train))
    _, num_features = sensor_data_train.shape
    type_analyser = SyntheticTypeAnalyser(num_features)

    zero_responder_index = np.where(person_type_train == TYPE_TO_CLASS[GROUP_IDENTITY_TYPE])[0]
    selfish_index = np.where(person_type_train == TYPE_TO_CLASS[PERSONAL_IDENTITY_TYPE])[0]

    logging.info("Training data -> Zero-responders: %d" % len(zero_responder_index))
    logging.info("Training data -> Selfish: %d" % len(selfish_index))

    if len(zero_responder_index) > len(selfish_index):
        logging.info("Imbalanced dataset favouring zero-responders. Undersampling...")
        zeroresponder_sample_index = np.random.choice(zero_responder_index, size=len(selfish_index), replace=False)

        sensor_data_train = np.vstack(
            (sensor_data_train[zeroresponder_sample_index, :], sensor_data_train[selfish_index, :]))
        person_type_train = np.hstack((person_type_train[zeroresponder_sample_index], person_type_train[selfish_index]))

    validation_accuracy_monitor = 'val_acc'
    if target_accuracy is not None:
        logging.info("Training for target accuracy %s" % str(target_accuracy))
        early_stopping_callback = EarlyStoppingByTarget(monitor=validation_accuracy_monitor, target=target_accuracy,
                                                        verbose=1)
    else:
        logging.info("Training for best accuracy")
        early_stopping_callback = EarlyStopping(monitor=validation_accuracy_monitor, patience=20)
    callbacks = [early_stopping_callback,
                 ModelCheckpoint(filepath="trained_model.h5", monitor=validation_accuracy_monitor, save_best_only=True)]

    training_history = type_analyser.train(sensor_data_train,
                                           person_type_train,
                                           epochs,
                                           batch_size,
                                           callbacks)
    plot_training(training_history, "acc")

    return type_analyser


def run_scenario(robot_controller, emergency_environment, num_scenarios):
    robot_payoffs = []

    for scenario in range(num_scenarios):

        current_sensor_data = emergency_environment.reset()
        done = False
        scenario_payoff = 0

        while not done:
            logging.info("Data Index: %d " % emergency_environment.data_index)
            robot_action = robot_controller.sensor_data_callback(current_sensor_data)
            logging.info("robot_action: %s" % robot_action)

            current_sensor_data, robot_payoff, done = emergency_environment.step(robot_action)
            scenario_payoff += robot_payoff

        robot_payoffs.append(scenario_payoff)

    logging.info("Scenarios: %.4f " % len(robot_payoffs))
    logging.info("Mean payoffs: %.4f " % np.mean(robot_payoffs))
    logging.info("Std payoffs: %.4f " % np.std(robot_payoffs))
    logging.info("Max payoffs: %.4f " % np.max(robot_payoffs))
    logging.info("Min payoffs: %.4f " % np.mean(robot_payoffs))

    return robot_payoffs


def main():
    np.random.seed(SEED)

    zeroresponder_type_weight = 0.8  # According to: "Modelling social identification and helping in evacuation simulation"
    selfish_type_weight = 1 - zeroresponder_type_weight
    # target_accuracy = 0.65
    target_accuracy = None
    max_epochs = 200
    interactions_per_scenario = INTERACTIONS_PER_SCENARIO
    total_samples = 10000
    training_batch_size = 100
    num_scenarios = NUM_SCENARIOS

    sensor_data, person_type = get_dataset(selfish_type_weight, zeroresponder_type_weight, total_samples)
    sensor_data_train, sensor_data_test, person_type_train, person_type_test = train_test_split(sensor_data,
                                                                                                person_type,
                                                                                                test_size=0.33,
                                                                                                random_state=0)

    type_analyser = get_type_analyser(sensor_data_train, person_type_train, training_batch_size, target_accuracy,
                                      max_epochs)
    robot_controller = AutonomicManagerController(type_analyser)
    # robot_controller = PessimisticRobotController()
    # robot_controller = OptimisticRobotController()

    emergency_environment = EmergencyEvacuationEnvironment(sensor_data_test, person_type_test,
                                                           interactions_per_scenario)

    _ = run_scenario(robot_controller, emergency_environment, num_scenarios)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
