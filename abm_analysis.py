""""
Module for agent-based modelling analysis.

This script generates the violin plot used in the journal submission (plot_results function).

This module relies on Python 3+ for some statistical analysis.
"""

import math
import multiprocessing
import random
import time
import traceback
from multiprocessing import Pool
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from abm_statistics import NO_SUPPORT_COLUMN, ONLY_STAFF_SUPPORT_COLUMN, ONLY_PASSENGER_SUPPORT_COLUMN, \
    ADAPTIVE_SUPPORT_COLUMN, get_dataframe, plot_results, test_kruskal_wallis

PLOT_STYLE = 'seaborn-darkgrid'

NETLOGO_PROJECT_DIRECTORY = "/home/cgc87/github/robot-assisted-evacuation/"  # type:str
NETLOGO_MODEL_FILE = NETLOGO_PROJECT_DIRECTORY + "/impact2.10.7/v2.11.0.nlogo"  # type:str
NETLOGO_HOME = "/home/cgc87/netlogo-5.3.1-64"  # type:str
NETLOGO_VERSION = "5"  # type:str

TURTLE_PRESENT_REPORTER = "count turtles"  # type:str
EVACUATED_REPORTER = "number_passengers - count agents + 1"  # type:str
DEAD_REPORTER = "count agents with [ st_dead = 1 ]"  # type:str

SEED_SIMULATION_REPORTER = "seed-simulation {}"

RESULTS_CSV_FILE = "data/{}_fall_{}_samples_experiment_results.csv"  # type:str

SET_SIMULATION_ID_COMMAND = "set SIMULATION_ID {}"  # type:str
SET_STAFF_SUPPORT_COMMAND = "set REQUEST_STAFF_SUPPORT {}"  # type: str
SET_PASSENGER_SUPPORT_COMMAND = "set REQUEST_BYSTANDER_SUPPORT {}"  # type: str
SET_FALL_LENGTH_COMMAND = "set DEFAULT_FALL_LENGTH {}"  # type:str
SET_ENABLE_LOGGING_COMMAND = "set ENABLE_LOGGING {}"  # type:str
SET_GENERATE_FRAMES_COMMAND = "set ENABLE_FRAME_GENERATION {}"  # type:str
SET_NUMBER_PASSENGERS_COMMAND = "set number_passengers {}"  # type:str
SET_NUMBER_NORMAL_STAFF_COMMAND = "set _number_normal_staff_members {}"  # type:str
SET_NUMBER_STAFF_COMMAND = "set _number_staff_members {}"  # type:str

ENABLE_STAFF_COMMAND = SET_STAFF_SUPPORT_COMMAND.format("TRUE")  # type:str
ENABLE_PASSENGER_COMMAND = SET_PASSENGER_SUPPORT_COMMAND.format("TRUE")  # type:str

NETLOGO_MINIMUM_SEED = -2147483648  # type:int
NETLOGO_MAXIMUM_SEED = 2147483647  # type:int

SIMULATION_SCENARIOS = {NO_SUPPORT_COLUMN: [],
                        ONLY_STAFF_SUPPORT_COLUMN: [ENABLE_STAFF_COMMAND],
                        ONLY_PASSENGER_SUPPORT_COLUMN: [ENABLE_PASSENGER_COMMAND],
                        ADAPTIVE_SUPPORT_COLUMN: [ENABLE_PASSENGER_COMMAND,
                                                  ENABLE_STAFF_COMMAND]}  # type: Dict[str, List[Tuple]]

# Settings for experiments
SAMPLES = 100  # type:int
MAX_NETLOGO_TICKS = 2000  # type: int
FALL_LENGTHS = [minutes * 30 for minutes in range(1, 21)]  # type: List[int]

# TODO(cgavidia): Uncomment for test runs
SAMPLES = 2
FALL_LENGTHS = [minutes * 60 for minutes in range(3, 5)]
SIMULATION_SCENARIOS = {ADAPTIVE_SUPPORT_COLUMN: [
    (SET_GENERATE_FRAMES_COMMAND.format("TRUE"), False),
    (SET_ENABLE_LOGGING_COMMAND.format("TRUE"), False),
    (ENABLE_PASSENGER_COMMAND, False),
    (ENABLE_STAFF_COMMAND, False)]}
NETLOGO_MINIMUM_SEED = 0  # type:int
NETLOGO_MAXIMUM_SEED = 10  # type:int


class ExperimentRun(object):

    def __init__(self, simulation_id, commands_per_scenario, random_seed, normal_staff_number, staff_number,
                 passenger_number, fall_length):
        # type: (int, List[Tuple[str, bool]], int, int, int, int, int) -> None

        self.simulation_id = simulation_id  # type: int
        self.base_scenario_commands = commands_per_scenario  # type: List[Tuple[str, bool]]

        self.random_seed = random_seed  # type: int
        self.normal_staff_number = normal_staff_number  # type: int
        self.staff_number = staff_number  # type: int
        self.passenger_number = passenger_number  # type: int
        self.fall_length = fall_length  # type: int

    def get_random_seed_command(self):
        # type: () -> str
        return SEED_SIMULATION_REPORTER.format(self.random_seed)

    def get_pre_setup_commands(self):
        # type: () -> List[str]
        pre_setup_commands = [command for command, before_setup in self.base_scenario_commands
                              if before_setup]  # type: List[str]

        pre_setup_commands.append(SET_NUMBER_NORMAL_STAFF_COMMAND.format(self.normal_staff_number))
        pre_setup_commands.append(SET_NUMBER_STAFF_COMMAND.format(self.staff_number))
        pre_setup_commands.append(SET_NUMBER_PASSENGERS_COMMAND.format(self.passenger_number))

        return pre_setup_commands

    def get_post_setup_commands(self):
        # type: () -> List[str]
        post_setup_commands = [command for command, before_setup in self.base_scenario_commands if
                               not before_setup]  # type: List[str]

        post_setup_commands.append(SET_FALL_LENGTH_COMMAND.format(self.fall_length))

        return post_setup_commands


def get_experiment_runs(samples, fall_length, commands_per_scenario):
    # type: (int, int, List[Tuple[str, bool]]) -> List[ExperimentRun]
    experiment_runs = []  # type: List[ExperimentRun]

    random_seeds = [random.randint(NETLOGO_MINIMUM_SEED, NETLOGO_MAXIMUM_SEED) for _ in
                    range(samples)]  #  type: List[int]
    normal_staff_number = [staff for staff in range(1, samples + 1)]  # type: List[int]
    staff_number = [staff for staff in range(1, samples + 1)]  # type: List[int]
    passenger_number = [150 * index for index in range(1, samples + 1)]  # type: List[int]

    for simulation_id in range(samples):
        experiment_runs.append(ExperimentRun(simulation_id=simulation_id, commands_per_scenario=commands_per_scenario,
                                             random_seed=random_seeds[simulation_id],
                                             normal_staff_number=normal_staff_number[simulation_id],
                                             staff_number=staff_number[simulation_id],
                                             passenger_number=passenger_number[simulation_id],
                                             fall_length=fall_length))

    return experiment_runs


def run_simulation(experiment_run):
    # type: (ExperimentRun) -> Optional[float]
    from pyNetLogo import NetLogoException

    simulation_id = experiment_run.simulation_id
    pre_setup_commands = experiment_run.get_pre_setup_commands()  # type: List[str]
    post_setup_commands = experiment_run.get_post_setup_commands()  # type: List[str]

    try:
        random_seed = netlogo_link.report(experiment_run.get_random_seed_command())  # type:str

        if len(pre_setup_commands) > 0:
            for pre_setup_command in pre_setup_commands:
                netlogo_link.command(pre_setup_command)
                print("id:{} seed:{} {} executed before setup".format(simulation_id, random_seed, pre_setup_command))
        else:
            print("id:{} seed:{} no pre-setup commands".format(simulation_id, random_seed))

        netlogo_link.command("setup")

        if len(post_setup_commands) > 0:
            for pre_setup_command in post_setup_commands:
                netlogo_link.command(pre_setup_command)
                print("id:{} seed:{} {} executed after setup".format(simulation_id, random_seed, pre_setup_command))
        else:
            print("id:{} seed:{} no post-setup commands".format(simulation_id, random_seed))

        metrics_dataframe = netlogo_link.repeat_report(
            netlogo_reporter=[TURTLE_PRESENT_REPORTER, EVACUATED_REPORTER, DEAD_REPORTER],
            reps=MAX_NETLOGO_TICKS)  # type: pd.DataFrame

        evacuation_finished = metrics_dataframe[
            metrics_dataframe[TURTLE_PRESENT_REPORTER] == metrics_dataframe[DEAD_REPORTER]]

        evacuation_time = evacuation_finished.index.min()  # type: float
        print("id:{} seed:{} evacuation time {}".format(simulation_id, random_seed, evacuation_time))
        if math.isnan(evacuation_time):
            metrics_dataframe.to_csv("data/nan_df.csv")
            print("DEBUG!!! info to data/nan_df.csv")

        return evacuation_time
    except NetLogoException:
        traceback.print_exc()
        raise
    except Exception:
        traceback.print_exc()

    return None


def initialize(gui):
    # type: (bool) -> None
    global netlogo_link
    import pyNetLogo

    netlogo_link = pyNetLogo.NetLogoLink(netlogo_home=NETLOGO_HOME,
                                         netlogo_version=NETLOGO_VERSION,
                                         gui=gui)  # type: pyNetLogo.NetLogoLink
    netlogo_link.load_model(NETLOGO_MODEL_FILE)


def start_experiments(fall_length, experiment_configurations, results_file):
    # type: (int, Dict[str, List[Tuple[str, bool]]], str) -> None

    start_time = time.time()  # type: float

    experiment_data = {}  # type: Dict[str, List[float]]
    for experiment_name, experiment_commands in experiment_configurations.items():
        scenario_times = run_parallel_simulations(SAMPLES,
                                                  fall_length,
                                                  setup_commands=experiment_commands)  # type:List[float]
        experiment_data[experiment_name] = scenario_times

    end_time = time.time()  # type: float
    print("Simulation finished after {} seconds".format(end_time - start_time))

    experiment_results = pd.DataFrame(experiment_data)  # type:pd.DataFrame
    experiment_results.to_csv(results_file)

    print("Data written to {}".format(results_file))


def run_simulation_with_dict(dict_parameters):
    # type: (Dict) -> float
    return run_simulation(**dict_parameters)


def run_parallel_simulations(samples, fall_length, setup_commands, gui=False):
    # type: (int, int, List[Tuple[str, bool]], bool) -> List[float]

    initialise_arguments = (gui,)  # type: Tuple
    simulation_parameters = get_experiment_runs(samples, fall_length, setup_commands)  # type: List[ExperimentRun]

    results = []  # type: List[float]
    executor = Pool(initializer=initialize,
                    initargs=initialise_arguments)  # type: multiprocessing.pool.Pool

    for simulation_output in executor.map(func=run_simulation,
                                          iterable=simulation_parameters):
        if simulation_output:
            results.append(simulation_output)

    executor.close()
    executor.join()

    return results


def simulate_and_store(fall_length):
    # type: (int) -> None
    results_file_name = RESULTS_CSV_FILE.format(fall_length, SAMPLES)  # type:str

    start_experiments(fall_length, SIMULATION_SCENARIOS, results_file_name)


def get_current_file_metrics(current_file):
    # type: (str) -> Dict[str, float]
    results_dataframe = get_dataframe(current_file)  # type: pd.DataFrame
    metrics_dict = {}  # type: Dict[str, float]

    for scenario in SIMULATION_SCENARIOS.keys():
        metrics_dict["{}_mean".format(scenario)] = results_dataframe[scenario].mean()
        metrics_dict["{}_std".format(scenario)] = results_dataframe[scenario].std()
        metrics_dict["{}_median".format(scenario)] = results_dataframe[scenario].median()
        metrics_dict["{}_min".format(scenario)] = results_dataframe[scenario].min()
        metrics_dict["{}_max".format(scenario)] = results_dataframe[scenario].max()

    return metrics_dict


def perform_analysis(fall_length):
    # type: (int) -> Dict[str, float]

    current_file = RESULTS_CSV_FILE.format(fall_length, SAMPLES)  # type:str
    plt.style.use(PLOT_STYLE)
    plot_results(csv_file=current_file)
    current_file_metrics = get_current_file_metrics(current_file)  # type: Dict[str, float]
    current_file_metrics["fall_length"] = fall_length

    current_file_metrics.update(
        test_kruskal_wallis(current_file, list(SIMULATION_SCENARIOS.keys())))

    # alternative = "less"  # type:str
    # for scenario_under_analysis in SIMULATION_SCENARIOS.keys():
    #     for alternative_scenario in SIMULATION_SCENARIOS.keys():
    #         if alternative_scenario != scenario_under_analysis:
    #             scenario_description = "{}_{}_{}".format(scenario_under_analysis, alternative, alternative_scenario)
    #             current_file_metrics[scenario_description] = test_mann_whitney(
    #                 first_scenario_column=scenario_under_analysis,
    #                 second_scenario_column=alternative_scenario,
    #                 alternative=alternative,
    #                 csv_file=current_file)

    return current_file_metrics


if __name__ == "__main__":
    for length in FALL_LENGTHS:
        simulate_and_store(length)

    metrics = pd.DataFrame([perform_analysis(length) for length in FALL_LENGTHS])  # type: pd.DataFrame
    metrics_file = "data/metrics.csv"  # type: str
    metrics.to_csv(metrics_file)
    print("Consolidates metrics written to {}".format(metrics_file))
