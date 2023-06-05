import io
import xlsxwriter
import pandas as pd

def write_file(df) -> pd.DataFrame:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Filtered Data')
        writer.close()
    output.seek(0)
    return output
