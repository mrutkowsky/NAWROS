import io
import xlsxwriter
import pandas as pd
from utils.data_processing import read_file


def write_file(df, document_type: str) -> pd.DataFrame:
    output = io.BytesIO()
    if document_type == 'excel':
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Filtered Data')
            writer.close()
    elif document_type == 'csv':
        df.to_csv(output, index=False)
    else:
        raise ValueError("Invalid document type. Supported types: 'excel', 'csv'")
    
    output.seek(0)
    return output

def show_columns_for_filtering(path: str):

    filtered_df = read_file(path)
    columns_to_exclude = ['x', 'y']
    filtered_df_excluded = filtered_df.drop(columns_to_exclude, axis=1)
    
    return filtered_df_excluded