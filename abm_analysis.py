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

from abm_statistics import NO_SUPPORT_COLUMN, ONLY_STAFF_SUPPORT_COLUMN, ONLY_PASSENGER_SUPPORT_COLUMN, \
    ADAPTIVE_SUPPORT_COLUMN, get_dataframe, plot_results, test_kruskal_wallis
import formideable
from netlogo_config import NETLOGO_MODEL_FILE, NETLOGO_HOME, NETLOGO_VERSION, TURTLE_PRESENT_REPORTER, \
    EVACUATED_REPORTER, DEAD_REPORTER, ENABLE_STAFF_COMMAND, ENABLE_PASSENGER_COMMAND, MAX_NETLOGO_TICKS, ExperimentRun

PLOT_STYLE = 'seaborn-darkgrid'

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
        experiment_runs = run_parallel_simulations(setup_commands=experiment_commands)  # type:List[ExperimentRun]
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


def run_parallel_simulations(setup_commands, gui=False):
    # type: (List[Tuple[str, bool]], bool) -> List[ExperimentRun]

    initialise_arguments = (gui,)  # type: Tuple
    # Running FormIDEAble experiments. Adjustment for TOSEM is pending.
    simulation_parameters = formideable.get_runs_from_file(setup_commands)  # type: List[ExperimentRun]

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


def simulate_and_store(fall_length):
    # type: (int) -> None
    # Uncomment for TOSEM
    # results_file_name = RESULTS_CSV_FILE.format(fall_length, SAMPLES)  # type:str
    results_file_name = RESULTS_CSV_FILE.format(fall_length, formideable.SAMPLES)  # type:str

    # Uncomment for TOSEM experiments
    # start_experiments(fall_length, SIMULATION_SCENARIOS, results_file_name)

    start_experiments(fall_length, formideable.SIMULATION_SCENARIOS, results_file_name)


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

    # Uncomment for TOSEM
    # current_file = RESULTS_CSV_FILE.format(fall_length, SAMPLES)  # type:str
    current_file = RESULTS_CSV_FILE.format(fall_length, formideable.SAMPLES)  # type:str
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
    # Uncomment for TOSEM
    # for length in FALL_LENGTHS:
    for length in formideable.FALL_LENGTHS:
        simulate_and_store(length)

    # Uncomment for TOSEM
    # metrics = pd.DataFrame([perform_analysis(length) for length in FALL_LENGTHS])  # type: pd.DataFrame
    metrics = pd.DataFrame([perform_analysis(length) for length in formideable.FALL_LENGTHS])  # type: pd.DataFrame

    metrics_file = "data/metrics.csv"  # type: str
    metrics.to_csv(metrics_file)
    print("Consolidates metrics written to {}".format(metrics_file))
