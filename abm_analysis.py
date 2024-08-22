""""
Module for agent-based modelling analysis.

This script generates the violin plot used in the journal submission (plot_results function).

This module relies on Python 3+ for some statistical analysis.
"""

import math
import multiprocessing
import time
import traceback
from multiprocessing import Pool
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from argparse import ArgumentParser
from abm_statistics import NO_SUPPORT_COLUMN, ONLY_STAFF_SUPPORT_COLUMN, ONLY_PASSENGER_SUPPORT_COLUMN, \
    ADAPTIVE_SUPPORT_COLUMN, get_dataframe, plot_results, test_kruskal_wallis
import formideable
from netlogo_config import NETLOGO_MODEL_FILE, NETLOGO_HOME, NETLOGO_VERSION, TURTLE_PRESENT_REPORTER, \
    EVACUATED_REPORTER, DEAD_REPORTER, ENABLE_STAFF_COMMAND, ENABLE_PASSENGER_COMMAND, MAX_NETLOGO_TICKS, ExperimentRun

PLOT_STYLE = 'seaborn-darkgrid'

LOAD_CONFIG_FROM_FILE = False  # type: bool
USE_FORMIDEABLE_CONFIG = False  # type: bool
SHOW_PLOTS = True # type: bool

RESULTS_CSV_FILE = "data/{}_fall_{}_samples_experiment_results.csv"  # type:str

SIMULATION_SCENARIOS = {NO_SUPPORT_COLUMN: [],
                        ONLY_STAFF_SUPPORT_COLUMN: [ENABLE_STAFF_COMMAND],
                        ONLY_PASSENGER_SUPPORT_COLUMN: [ENABLE_PASSENGER_COMMAND],
                        ADAPTIVE_SUPPORT_COLUMN: [ENABLE_PASSENGER_COMMAND,
                                                  ENABLE_STAFF_COMMAND]}  # type: Dict[str, List[Tuple]]

# Settings for experiments
SAMPLES = 100  # type:int
FALL_LENGTHS = [minutes * 30 for minutes in range(1, 21)]  # type: List[int]


# TODO(cgavidia): Uncomment for test runs
# SAMPLES = 2
# FALL_LENGTHS = [minutes * 60 for minutes in range(3, 5)]
# NETLOGO_MINIMUM_SEED = 0  # type:int
# NETLOGO_MAXIMUM_SEED = 10  # type:int


def run_simulation(experiment_run):
    # type: (ExperimentRun) -> Optional[ExperimentRun]
    from pyNetLogo import NetLogoException

    simulation_id = experiment_run.get_netlogo_simulation_id()
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

        experiment_run.evacuation_time = evacuation_time
        return experiment_run
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
        experiment_runs = run_parallel_simulations(setup_commands=experiment_commands,
                                                   fall_length=fall_length,
                                                   experiment_name=experiment_name)  # type:List[ExperimentRun]
        experiment_data[experiment_name] = [run.evacuation_time for run in experiment_runs]
        experiment_data["{}_seed".format(experiment_name)] = [run.random_seed for run in experiment_runs]
        experiment_data["{}_passengers".format(experiment_name)] = [run.passenger_number for run in experiment_runs]
        experiment_data["{}_staff".format(experiment_name)] = [run.staff_number for run in experiment_runs]
        experiment_data["{}_normal_staff".format(experiment_name)] = [run.normal_staff_number for run in
                                                                      experiment_runs]

    end_time = time.time()  # type: float
    print("Simulation finished after {} seconds".format(end_time - start_time))

    experiment_results = pd.DataFrame(experiment_data)  # type:pd.DataFrame
    experiment_results.to_csv(results_file)

    print("Data written to {}".format(results_file))


def run_simulation_with_dict(dict_parameters):
    # type: (Dict) -> float
    return run_simulation(**dict_parameters)


def run_parallel_simulations(setup_commands, fall_length, experiment_name="", gui=False):
    # type: (List[Tuple[str, bool]], int, str, bool) -> List[ExperimentRun]

    initialise_arguments = (gui,)  # type: Tuple

    if LOAD_CONFIG_FROM_FILE:
        simulation_parameters = formideable.get_runs_from_file(experiment_name,
                                                               setup_commands, fall_length)  # type: List[ExperimentRun]
    else:
        # Running FormIDEAble experiments. Adjustment for TOSEM is pending.
        simulation_parameters = formideable.get_runs(experiment_name,
                                                     setup_commands, fall_length)  # type: List[ExperimentRun]

    results = []  # type: List[ExperimentRun]
    executor = Pool(initializer=initialize,
                    initargs=initialise_arguments)  # type: multiprocessing.pool.Pool

    for simulation_output in executor.map(func=run_simulation,
                                          iterable=simulation_parameters):
        if simulation_output:
            results.append(simulation_output)

    executor.close()
    executor.join()

    return results


def simulate_and_store(fall_length, results_file_name=None):
    # type: (int, Optional[str]) -> None

    simulation_scenarios = SIMULATION_SCENARIOS
    samples = SAMPLES
    if USE_FORMIDEABLE_CONFIG:
        simulation_scenarios = formideable.SIMULATION_SCENARIOS
        samples = formideable.SAMPLES

    if results_file_name is None:
        results_file_name = RESULTS_CSV_FILE.format(fall_length, samples)  # type:str

    start_experiments(fall_length, simulation_scenarios, results_file_name)


def get_current_file_metrics(current_file):
    # type: (str) -> Dict[str, float]
    results_dataframe = get_dataframe(current_file)  # type: pd.DataFrame
    metrics_dict = {}  # type: Dict[str, float]

    for scenario in SIMULATION_SCENARIOS.keys():
        if scenario in results_dataframe.columns.tolist():
            metrics_dict["{}_mean".format(scenario)] = results_dataframe[scenario].mean()
            metrics_dict["{}_std".format(scenario)] = results_dataframe[scenario].std()
            metrics_dict["{}_median".format(scenario)] = results_dataframe[scenario].median()
            metrics_dict["{}_min".format(scenario)] = results_dataframe[scenario].min()
            metrics_dict["{}_max".format(scenario)] = results_dataframe[scenario].max()

    return metrics_dict


def perform_analysis(fall_length, current_file=None):
    # type: (int, Optional[str]) -> Dict[str, float]

    samples = SAMPLES
    if USE_FORMIDEABLE_CONFIG:
        samples = formideable.SAMPLES

    if current_file is None:
        current_file = RESULTS_CSV_FILE.format(fall_length, samples)  # type:str
    plt.style.use(PLOT_STYLE)
    plot_results(csv_file=current_file, show_plots=SHOW_PLOTS)
    current_file_metrics = get_current_file_metrics(current_file)  # type: Dict[str, float]
    current_file_metrics["fall_length"] = fall_length

    test_results = test_kruskal_wallis(current_file,
                                       list(SIMULATION_SCENARIOS.keys()))  # type: Optional[Dict[str, bool]]
    if test_results is not None:
        current_file_metrics.update(test_results)

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


def main():
    fall_lengths = FALL_LENGTHS
    file_name_format = None
    print("USE_FORMIDEABLE_CONFIG={}".format(USE_FORMIDEABLE_CONFIG))
    if USE_FORMIDEABLE_CONFIG:
        fall_lengths = formideable.FALL_LENGTHS

        prefix = formideable.RESULTS_FILE_PREFIX
        suffix = formideable.RESULT_FILE_SUFFIX

        if not formideable.USE_FORMIDEABLE_FILES:
            prefix = formideable.IDEA_RESULTS_FILE_PREFIX
            suffix = formideable.IDEA_RESULT_FILE_SUFFIX

        file_name_format = "data/formideable/" + prefix + suffix

    for length in fall_lengths:
        simulate_and_store(length, file_name_format.format(length))

    metrics = pd.DataFrame([perform_analysis(length, file_name_format.format(length)) for length in
                            fall_lengths])  # type: pd.DataFrame

    metrics_file = "data/metrics.csv"  # type: str
    metrics.to_csv(metrics_file)
    print("Consolidates metrics written to {}".format(metrics_file))


if __name__ == "__main__":
    parser = ArgumentParser()  # type: ArgumentParser
    parser.add_argument("data_directory")
    parser.add_argument("samples")

    parser.add_argument("--load_file")
    parser.add_argument("--formideable_config")
    parser.add_argument("--naive")
    parser.add_argument("--formideable_files")
    parser.add_argument("--show_plots")


    arguments = parser.parse_args()

    formideable.DATA_DIRECTORY = arguments.data_directory
    formideable.SAMPLES = int(arguments.samples)

    LOAD_CONFIG_FROM_FILE = False
    if arguments.load_file == "1":
        LOAD_CONFIG_FROM_FILE = True

    USE_FORMIDEABLE_CONFIG = False
    if arguments.formideable_config == "1":
        USE_FORMIDEABLE_CONFIG = True

    formideable.USE_FORMIDEABLE_FILES = False
    if arguments.formideable_files == "1":
        formideable.USE_FORMIDEABLE_FILES = True

    SHOW_PLOTS = False
    if arguments.show_plots == "1":
        SHOW_PLOTS = True

    print(arguments)

    main()
