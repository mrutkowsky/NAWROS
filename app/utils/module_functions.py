import pandas as pd
import os
import yaml
import logging
import io
from flask import make_response
import json

logger = logging.getLogger(__file__)


def compare_columns(
    file_columns: str,
    required_columns: set) -> set:

    """
    Compare the columns of a file to a set of required columns.

    Args:
        file_columns (str): The columns of the file.
        required_columns (set): The set of required columns.

    Returns:
        set: A set of column names that are missing in the file.
    """

    required_columns_set = set(map(str.lower, required_columns))
    file_columns_set = set(map(str.lower, file_columns))

    column_compatibility = required_columns_set.issubset(file_columns_set) 
    
    return None if column_compatibility else required_columns_set.difference(file_columns_set)


def check_column_names(column_names: list) -> bool or str:
    """Asserts the column are aligned wiotg configuration file."""
    for i, col1 in enumerate(column_names):
        for j, col2 in enumerate(column_names):
            if i != j and col1.startswith(col2) and '.' in col1:
                return col2
    return False


def validate_file(
        file_path: str, 
        required_columns: list) -> bool or str:
    
    """
    Validate a file by checking if it has the required columns.

    Args:
        file_path: The path to the file.
        required_columns: The set of required columns.

    Returns:
        bool or str: Returns True if the file is valid, or an error message if the file is invalid.
    """
    
    _, tail = os.path.split(file_path)
    
    if tail.endswith('.xlsx'):

        try:
            file_df = pd.read_excel(
                file_path, 
                nrows=1)
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            return "Invalid file format or structure."
        else:
            
            try:
                first_line = file_df.iloc[0]
            except IndexError:
                return "Empty dataframe"
            else:
                if first_line.isnull().any():
                    return "Provided file has missing data in the first line of file."
                
            has_doubled_col = check_column_names(file_df.columns)
            if has_doubled_col:
                return f"Provided file has doubled column: {has_doubled_col}"
            
            logger.debug(f"Loaded filename df {file_df}")

            column_dismatch = compare_columns(
                file_columns=file_df.columns,
                required_columns=required_columns,
            )
        
            if column_dismatch:
                return f"File is missing required columns: {list(column_dismatch)}"
            
    else:

        try:
            file_df = pd.read_csv(
                file_path, 
                nrows=1)
             
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            return "Invalid file format or structure."
        else:

            if len(list(file_df.columns)) < len(required_columns):

                file_df = pd.read_csv(
                    file_path, 
                    nrows=1,
                    delimiter=";")
                
                if len(list(file_df.columns)) < len(required_columns):
                    return "Only comma (,) or semicolon (;) are allowed as separators."

            try:
                first_line = file_df.iloc[0]
            except IndexError:
                return "Empty dataframe"
            else:
                if first_line.isnull().any():
                    return "Provided file has missing data in the first line of file."
                
            has_doubled_col = check_column_names(file_df.columns)
            if has_doubled_col:
                return f"Provided file has doubled column: {has_doubled_col}"

            column_dismatch = compare_columns(
                file_columns=file_df.columns,
                required_columns=required_columns,
            )
            
            if column_dismatch:
                return f"File is missing required columns: {list(column_dismatch)}"
            
    return True


def validate_file_extension(
        filename: str,
        allowed_extensions: set) -> bool:
    
    """
    Validate the extension of a file.

    Args:
        filename (str): The name of the file.
        allowed_extensions (set): A set of allowed extensions.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    
    return True if os.path.splitext(filename)[-1].lower() in allowed_extensions else False


def read_config(  
        config_filename: str,
        path_to_dir: str = None) -> dict:
    
    """
    Read a YAML configuration file.

    Args:
        config_filename (str): The name of the configuration file.
        path_to_dir (str, optional): The path to the directory containing the configuration file.
            Defaults to None.

    Returns:
        dict: The configuration data as a dictionary.
    """
    
    path_to_configfile = os.path.join(
        path_to_dir, 
        config_filename) \
            if path_to_dir is not None else config_filename
    
    try:
    
        with open(path_to_configfile) as yaml_file:
            config = yaml.load(yaml_file, Loader=yaml.SafeLoader)

    except FileNotFoundError:
        return None

    else:
        return config


def get_report_ext(
    path_to_arf_dir: str,
    filename: str) -> dict:
    """Return the extension of the report file."""
    with open(os.path.join(path_to_arf_dir, filename), 'r', encoding='utf-8') as arf_json:
        return json.load(arf_json)