import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import logging
from utils.data_processing import read_file

logger = logging.getLogger(__file__)

def find_newer_report(
        path_to_reports_dir: str,
        first_report_name: str, 
        second_report_name: str,
        only_for_existence: bool = False,
        older_report_key: str = 'old',
        newer_report_key: str = 'new') -> dict or bool or None: 
    """
    Function to compare two report files and return the newer one.

    Args:
        path_to_reports_dir (str): The path to the directory containing the reports.
        first_report_name (str): The name of the first report.
        second_report_name (str): The name of the second report.
        only_for_existence (bool, optional): If True, the function will only check if the reports exist.
            Defaults to False.
        older_report_key (str, optional): The key for the older report in the returned dictionary.
            Defaults to 'old'.
        newer_report_key (str, optional): The key for the newer report in the returned dictionary.
            Defaults to 'new'.

    Returns:
        dict or bool or None: A dictionary with the older and newer report names,
            or True if only_for_existence
    """

    TIMESTAMP_PATTERN = r"_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\..+"
    TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S"

    report1_path = os.path.join(path_to_reports_dir, first_report_name)
    report2_path = os.path.join(path_to_reports_dir, second_report_name)

    if not os.path.exists(report1_path):

        logger.error(f"Report '{first_report_name}' does not exist in the reports directory.")
        return None

    if not os.path.exists(report2_path):

        logger.error(f"Report '{second_report_name}' does not exist in the reports directory.")
        return None
    
    match1 = re.search(TIMESTAMP_PATTERN, first_report_name)
    match2 = re.search(TIMESTAMP_PATTERN, second_report_name)

    if not match1:

        logger.error(f"Unable to extract timestamp from report name: {first_report_name}")
        return None

    if not match2:

        logger.error(f"Unable to extract timestamp from report name: {second_report_name}")
        return None
        

    if only_for_existence:
        return True

    timestamp1_str = match1.group(1)
    timestamp2_str = match2.group(1)

    logger.debug(timestamp1_str)
    logger.debug(timestamp2_str)

    datetime_first_report = datetime.strptime(timestamp1_str, TIMESTAMP_FORMAT)
    datetime_second_report = datetime.strptime(timestamp2_str, TIMESTAMP_FORMAT)

    result = {}

    if datetime_first_report > datetime_second_report:

        logger.info(f"Report '{first_report_name}' is newer than '{second_report_name}'.")
        result[older_report_key] = second_report_name
        result[newer_report_key] = first_report_name

        return result

    elif datetime_first_report < datetime_second_report:

        logger.info(f"Report '{second_report_name}' is newer than '{first_report_name}'.")
        result[older_report_key] = first_report_name
        result[newer_report_key] = second_report_name

        return result
    
    else:
        logger.error(f"Both reports have the same timestamp.")
        return None


def compare_reports(
        first_report_name: str, 
        second_report_name: str,
        path_to_reports_dir: str,
        only_for_existence: bool = False,
        topics_number: int = 5,
        must_match_topics_numbers: int = 3,
        no_topic_token: str = '-',
        topic_preffix_name: str = 'Word',
        cardinality_column: str = 'counts',
        old_report_column_prefix: str = 'Old',
        new_report_column_prefix: str = 'New',
        old_cluster_value: str = 'Old group',
        new_cluster_value: str = 'New group') -> pd.DataFrame or None:
    """
    Function to compare the topics of two reports. For each topic, the function will check if the
    topic is present in both reports, and if so, it will calulated the difference in counts.
    If the topic is no present in one of the reports, the function will assign the label
    old_cluster_value or new_cluster_value.


    Args:
        first_report_name (str): The name of the first report.
        second_report_name (str): The name of the second report.
        path_to_reports_dir (str): The path to the directory containing the reports.
        only_for_existence (bool, optional): If True, the function will only check if the reports exist.
            Defaults to False.
        topics_number (int, optional): The number of topics in the reports. Defaults to 5.
        must_match_topics_numbers (int, optional): The number of topics that must match in both reports.
            Defaults to 3.
        no_topic_token (str, optional): The token to use for topics that are not present in one of the reports.
            Defaults to '-'.
        topic_preffix_name (str, optional): The prefix for the topic names in the reports. Defaults to 'Word'.
        cardinality_column (str, optional): The name of the column containing the counts of the topics.
            Defaults to 'counts'.
        old_report_column_prefix (str, optional): The prefix for the columns of the first report.
            Defaults to 'Old'.
        new_report_column_prefix (str, optional): The prefix for the columns of the second report.
            Defaults to 'New'.
        old_cluster_value (str, optional): The value to assign to topics that are not present in the first report.
            Defaults to 'Old group'.
        new_cluster_value (str, optional): The value to assign to topics that are not present in the second report.
            Defaults to 'New group'.
        
    Returns:
        pd.DataFrame: A dataframe with the comparison results.
    """

    COMPARISON_COLUMN_NAME = 'Comparison'
    OLDER_REPORT_KEY = 'old'
    NEWER_REPORT_KEY = 'new'

    time_comp_result = find_newer_report(
        path_to_reports_dir=path_to_reports_dir,
        first_report_name=first_report_name,
        second_report_name=second_report_name,
        only_for_existence=only_for_existence,
        older_report_key=OLDER_REPORT_KEY,
        newer_report_key=NEWER_REPORT_KEY
    )

    if time_comp_result is None:
        logger.error("Can not compare reports!")
        return None

    elif isinstance(time_comp_result, bool):

        report1_name = first_report_name
        report2_name = second_report_name

    elif isinstance(time_comp_result, dict):

        report1_name = time_comp_result.get(OLDER_REPORT_KEY)
        report2_name = time_comp_result.get(NEWER_REPORT_KEY)

    else:
        logger.error(f'Unexpected error: Instance of time_com_result: {type(time_comp_result)}')
        return None
    
    old_report_path = os.path.join(
        path_to_reports_dir, report1_name
    )
    
    new_report_path = os.path.join(
        path_to_reports_dir, report2_name
    )

    topic_columns = [f"{topic_preffix_name}_{i}" for i in range(1, topics_number + 1)]
    columns_to_load = topic_columns + [cardinality_column, ]

    old_report_df = read_file(
        file_path=old_report_path,
        columns=columns_to_load
    )

    new_report_df = read_file(
        file_path=new_report_path,
        columns=columns_to_load
    )

    if list(old_report_df.columns) != list(new_report_df.columns):

        logger.error(f'Column names must match for comparison!')
        logger.error(f'Old report columns: {list(old_report_df.columns)}')
        logger.error(f'New report columns: {list(new_report_df.columns)}')

        return None

    report1 = old_report_df.copy()
    report2 = new_report_df.copy()

    rows_for_result_df = []

    for idx2, row2 in report2.iterrows():

        group2 = set(row2[topic_columns]).difference(no_topic_token)  
        cardinality2 = row2[cardinality_column]

        new_group_flag = True

        for idx1, row1 in report1.iterrows():

            group1 = set(row1[topic_columns]).difference(no_topic_token)  
            cardinality1 = row1[cardinality_column]

            if (len(group2.intersection(group1)) >= must_match_topics_numbers) \
                or ((group2.issubset(group1)) and (len(group2) != 0) and (len(group1) != 0)) \
                or ((group1.issubset(group2)) and (len(group2) != 0) and (len(group1) != 0)) \
                or (group1 == group2):

                growth = cardinality2 - cardinality1
                growth_str = f'+{growth}' if growth >= 0 else str(growth)

                presented_in_both = list(row1) + list(row2) + [growth_str]
                rows_for_result_df.append(presented_in_both)

                report1 = report1[report1.index != row1.name]
                new_group_flag = False
                break
            
        if new_group_flag:

            new_group_row = [''] * 6 + list(row2) + [new_cluster_value]
            rows_for_result_df.append(new_group_row)

    for _, row1 in report1.iterrows():

        new_group_row = list(row1) + [''] * 6 + [old_cluster_value]
        rows_for_result_df.append(new_group_row)

    result_columns = [f'{old_report_column_prefix}_{col}' for col in report1.columns] \
        + [f'{new_report_column_prefix}_{col}' for col in report2.columns] \
        + [COMPARISON_COLUMN_NAME]

    result_df = pd.DataFrame(
        rows_for_result_df,
        columns=result_columns)

    return result_df


def find_latest_two_reports(
        path_to_reports_dir: str) -> list[str]:
    """
    Finds the two latest reports in the directory.
    """
    report_files = []

    timestamp_pattern = r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}"

    files = os.listdir(path_to_reports_dir)

    for file_ in files:
        match = re.search(timestamp_pattern, file_)
        if match:
            report_files.append(file_)

    report_files.sort(reverse=True)

    return report_files[:2]


def find_latested_n_exec_report(
    path_to_dir: str,
    cluster_exec_prefix: str = 'cluster_exec',
    n_reports: int = 1) -> str or tuple[str, str]:
    """
    Find n latest cluster execution reports in the directory.
    It is assumed that the name of the report file is in the format:
    cluster_exec_2021_08_31_12_00_00.json

    Args:
        path_to_dir (str): Path to the directory with reports.
        cluster_exec_prefix (str, optional): Prefix of the report file name. Defaults to 'cluster_exec'.
        n_reports (int, optional): Number of reports to return. Defaults to 1.
    
    Returns:
        str or tuple[str, str]: Name of the report file or tuple of two names of the report files.
    """

    TIMESTAMP_PATTERN = r"_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\..+"
    TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S"

    only_exec_reports = sorted([
        file_ for file_ in os.listdir(path_to_dir) 
        if file_.startswith(cluster_exec_prefix)], reverse=True)

    try:
        reports_to_return = only_exec_reports[0] if n_reports == 1 else only_exec_reports[:n_reports]
    except IndexError:
        logger.error(f'No cluster execution reports in {path_to_dir}!')
        return None
    else:

        if (n_reports > 1) and n_reports != len(reports_to_return):

            logger.error(f'Only one cluster execution report is avaiable currently')
            return None
        
        return reports_to_return
