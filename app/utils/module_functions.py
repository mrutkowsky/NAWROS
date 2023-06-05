import pandas as pd
import os
import yaml

def compare_columns(
    file_columns: str,
    required_columns: set) -> set:

    column_dismatch = set(map(str.lower, required_columns)).difference(
        set(map(str.lower, file_columns))) 
    
    return column_dismatch

def validate_file(
        file_path, 
        required_columns,
        delimeter: str = r'[;,]'):
    
    _, tail = os.path.split(file_path)
    
    if tail.endswith('.xlsx'):

        try:
            file_columns = pd.read_excel(
                file_path, 
                nrows=1).columns
        except pd.errors.ParserError:
            return "Invalid file format or structure."
        else:

            column_dismatch = compare_columns(
                file_columns=file_columns,
                required_columns=required_columns,
            )
        
            if column_dismatch:
                return list(column_dismatch)
            
    else:

        try:
            file_columns = pd.read_csv(
                file_path, 
                nrows=1,
                sep=delimeter,
                engine='python').columns 
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            return "Invalid file format or structure."
        else:

            column_dismatch = compare_columns(
                file_columns=file_columns,
                required_columns=required_columns,
            )
            
            if column_dismatch:
                return list(column_dismatch)
            
    return True

def validate_file_extension(
        filename: str,
        allowed_extensions: set) -> bool:
    
    return True if os.path.splitext(filename)[-1].lower() in allowed_extensions else False

def read_config(  
        config_filename: str,
        path_to_dir: str = None
) -> dict:
    
    path_to_configfile = os.path.join(
        path_to_dir, 
        config_filename) \
            if path_to_dir is not None else config_filename
    
    with open(path_to_configfile) as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.SafeLoader)

    return config