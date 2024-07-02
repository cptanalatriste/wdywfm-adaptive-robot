from itertools import combinations

import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats.stats import KruskalResult
from typing import List, Dict

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels import api as sm

NO_SUPPORT_COLUMN = "no-support"  # type:str
ONLY_STAFF_SUPPORT_COLUMN = "staff-support"  # type:str
ONLY_PASSENGER_SUPPORT_COLUMN = "passenger-support"  # type:str
ADAPTIVE_SUPPORT_COLUMN = "adaptive-support"

# Using https://www.stat.ubc.ca/~rollin/stats/ssize/n2.html
# And https://www.statology.org/pooled-standard-deviation-calculator/
# function to calculate Cohen's d for independent samples
# Inspired by: https://machinelearningmastery.com/effect-size-measures-in-python/

def cohen_d_from_metrics(mean_1, mean_2, std_dev_1, std_dev_2):
    # type: (float, float, float, float) -> float
    pooled_std_dev = np.sqrt((std_dev_1 ** 2 + std_dev_2 ** 2) / 2)
    return (mean_1 - mean_2) / pooled_std_dev


def calculate_sample_size(mean_1, mean_2, std_dev_1, std_dev_2, alpha=0.05, power=0.8):
    # type: (float, float, float, float, float, float) -> float
    analysis = sm.stats.TTestIndPower()  # type: sm.stats.TTestIndPower
    effect_size = cohen_d_from_metrics(mean_1, mean_2, std_dev_1, std_dev_2)
    result = analysis.solve_power(effect_size=effect_size,
                                  alpha=alpha,
                                  power=power,
                                  alternative="two-sided")
    return result


def get_dataframe(csv_file):
    # type: (str) -> pd.DataFrame
    results_dataframe = pd.read_csv(csv_file, index_col=[0])  # type: pd.DataFrame
    results_dataframe = results_dataframe.dropna()

    return results_dataframe


def plot_results(csv_file, samples_in_title=False):
    # type: (str, bool) -> None
    file_description = Path(csv_file).stem  # type: str
    results_dataframe = get_dataframe(csv_file)  # type: pd.DataFrame
    results_dataframe = results_dataframe.rename(columns={
        NO_SUPPORT_COLUMN: "No Support",
        ONLY_STAFF_SUPPORT_COLUMN: "Proself-Oriented",
        ONLY_PASSENGER_SUPPORT_COLUMN: "Prosocial-Oriented",
        ADAPTIVE_SUPPORT_COLUMN: "Adaptive"
    })

    print("Metrics for dataset {}".format(csv_file))
    print(results_dataframe.describe())

    title = ""
    order = ["No Support", "Prosocial-Oriented", "Proself-Oriented", "Adaptive"]  # type: List[str]
    order = [column for column in order if column in results_dataframe.columns.tolist()]
    if samples_in_title:
        title = "{} samples".format(len(results_dataframe))

    plot_axis = sns.violinplot(data=results_dataframe, order=order)
    plot_axis.set_title(title)
    plot_axis.set_xlabel("IDEA variant")
    plot_axis.set_ylabel("Evacuation time")

    plt.savefig("img/" + file_description + "_violin_plot.png", bbox_inches='tight', pad_inches=0)
    plt.savefig("img/" + file_description + "_violin_plot.eps", bbox_inches='tight', pad_inches=0)
    plt.show()

    _ = sns.stripplot(data=results_dataframe, order=order, jitter=True).set_title(title)
    plt.savefig("img/" + file_description + "_strip_plot.png", bbox_inches='tight', pad_inches=0)
    plt.savefig("img/" + file_description + "_strip_plot.eps", bbox_inches='tight', pad_inches=0)
    plt.show()


def test_kruskal_wallis(csv_file, column_list, threshold=0.05, method_for_adjusting="bonferroni"):
    # type: (str, List[str], float, str) -> Dict[str, bool]

    import scikit_posthocs as sp

    print("CURRENT ANALYSIS: Analysing file {}".format(csv_file))
    results_dataframe = get_dataframe(csv_file)  # type: pd.DataFrame

    data_as_list = [results_dataframe[column_name].values for column_name in column_list]  # type: List[List[float]]

    null_hypothesis = "KRUSKAL-WALLIS TEST: the population median of all of the groups are equal."  # type: str
    alternative_hypothesis = "ALTERNATIVE HYPOTHESIS: " \
                             "the population median of all of the groups are NOT equal."  # type:str

    kruskal_result = stats.kruskal(data_as_list[0], data_as_list[1], data_as_list[2],
                                   data_as_list[3])  # type: KruskalResult
    print("statistic={} , p-value={}".format(kruskal_result[0], kruskal_result[1]))

    result = {}  # type: Dict
    for first_scenario_index, second_scenario_index in combinations(range(0, len(column_list)), 2):
        first_scenario_description = column_list[first_scenario_index]  # type: str
        second_scenario_description = column_list[second_scenario_index]  # type: str
        result["{}_{}".format(first_scenario_description, second_scenario_description)] = False

    if kruskal_result[1] < threshold:
        print("REJECT NULL HYPOTHESIS: {}".format(null_hypothesis))
        print("Performing Post-Hoc pairwise Dunn's test ({} correction)".format(method_for_adjusting))
        print(alternative_hypothesis)

        p_values_dataframe = sp.posthoc_dunn(data_as_list, p_adjust=method_for_adjusting)
        print(p_values_dataframe)

        for first_scenario_index, second_scenario_index in combinations(range(0, len(column_list)), 2):
            first_scenario_description = column_list[first_scenario_index]  # type: str
            second_scenario_description = column_list[second_scenario_index]  # type: str

            p_value = p_values_dataframe.loc[first_scenario_index + 1][second_scenario_index + 1]
            if p_value < threshold:
                result["{}_{}".format(first_scenario_description, second_scenario_description)] = True
                print("{} (median {}) is significantly different than {} (median {}), with p-value={}".format(
                    first_scenario_description, np.median(data_as_list[first_scenario_index]),
                    second_scenario_description, np.median(data_as_list[second_scenario_index]),
                    p_value
                ))
    else:
        print("FAILS TO REJECT NULL HYPOTHESIS: {}".format(null_hypothesis))

    return result


def test_mann_whitney(first_scenario_column, second_scenario_column, csv_file, alternative="two-sided"):
    # type: (str, str, str, str) -> bool
    print("CURRENT ANALYSIS: Analysing file {}".format(csv_file))
    results_dataframe = get_dataframe(csv_file)  # type: pd.DataFrame

    first_scenario_data = results_dataframe[first_scenario_column].values  # type: List[float]
    first_scenario_mean = np.mean(first_scenario_data).item()  # type:float
    first_scenario_stddev = np.std(first_scenario_data).item()  # type:float

    second_scenario_data = results_dataframe[second_scenario_column].values  # type: List[float]
    second_scenario_mean = np.mean(second_scenario_data).item()  # type:float
    second_scenario_stddev = np.std(second_scenario_data).item()  # type:float

    print("{}-> mean = {} std = {} len={}".format(first_scenario_column, first_scenario_mean, first_scenario_stddev,
                                                  len(first_scenario_data)))
    print("{}-> mean = {} std = {} len={}".format(second_scenario_column, second_scenario_mean, second_scenario_stddev,
                                                  len(second_scenario_data)))
    print("Sample size: {}".format(
        calculate_sample_size(first_scenario_mean, second_scenario_mean, first_scenario_stddev,
                              second_scenario_stddev)))

    null_hypothesis = "MANN-WHITNEY RANK TEST: " + \
                      "The distribution of {} times is THE SAME as the distribution of {} times".format(
                          first_scenario_column, second_scenario_column)  # type: str
    alternative_hypothesis = "ALTERNATIVE HYPOTHESIS: the distribution underlying {} is stochastically {} than the " \
                             "distribution underlying {}".format(first_scenario_column, alternative,
                                                                 second_scenario_column)  # type:str

    threshold = 0.05  # type:float
    u, p_value = mannwhitneyu(x=first_scenario_data, y=second_scenario_data, alternative=alternative)
    print("U={} , p={}".format(u, p_value))

    is_first_sample_less_than_second = False  # type: bool
    if p_value > threshold:
        print("FAILS TO REJECT NULL HYPOTHESIS: {}".format(null_hypothesis))
    else:
        print("REJECT NULL HYPOTHESIS: {}".format(null_hypothesis))
        print(alternative_hypothesis)
        is_first_sample_less_than_second = True

    return is_first_sample_less_than_second
