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

def validate_file(
        file_path, 
        required_columns,
        delimeter: str = ';'):
    
    """
    Validate a file by checking if it has the required columns.

    Args:
        file_path: The path to the file.
        required_columns: The set of required columns.
        delimeter (str, optional): The delimiter used in case of a CSV file. Defaults to r'[;,]'.

    Returns:
        bool or str: Returns True if the file is valid, or an error message if the file is invalid.
    """
    
    _, tail = os.path.split(file_path)
    
    if tail.endswith('.xlsx'):

        try:
            file_df = pd.read_excel(
                file_path, 
                nrows=1,
                usecols=required_columns)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
            return "Invalid file format or structure."
        else:

            if file_df.iloc[0].isnull().any():
                return "Empty dataframe or provided file has missing data in the first line of file."
            
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
                nrows=1,
                sep=delimeter)
             
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            return "Invalid file format or structure."
        else:

            if file_df.iloc[0].isnull().any():
                return "Empty dataframe or provided file has missing data in the first line of file."

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
    
    with open(path_to_configfile) as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.SafeLoader)

    return config

def return_valid_response_file(
        filename: str,
        report_format: str,
        mimetype: str,
        ext: str):

    resp = make_response()
    resp.headers["Content-Disposition"] = \
        f"attachment; filename={filename}.{ext}"
    resp.headers["Content-type"] = mimetype

    return resp

def get_report_ext(
    path_to_arf_dir: str,
    filename: str):

    with open(os.path.join(path_to_arf_dir, filename), 'r', encoding='utf-8') as arf_json:
        return json.load(arf_json)