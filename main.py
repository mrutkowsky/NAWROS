import pandas as pd
import glob
import re
import os
# Path to the folder where the files are located
folder_path = os.path.join(os.getcwd(), 'data')
file_extension = ['xlsx', 'csv', 'txt']
files = []
Columns = ["Questioned_Date", "Model_No", "OS", "SW_Version", "CSC", "Category", "Application_Name", "content"]
    
def clean_text(text):
    # Remove punctuation marks
    cleaned_text = re.sub(r"[^\w\s]", "", re.sub(r"[\t\n]", "", text))

    return cleaned_text

def get_data(folder_path):
    # Download files with extensions .xlsx, .csv and .txt
    for extension in file_extension:
        files.extend(glob.glob(f"{folder_path}/*.{extension}"))

    # Create an empty pandas table that will hold the merged data
    merged_data = pd.DataFrame(columns=Columns)
    # Iterate through all the files and add their contents to the pandas table
    for file in files:
        if file.endswith('.xlsx'):
            data = pd.read_excel(file)
        elif file.endswith('.csv'):
            data = pd.read_csv(file, delimiter=';')
        else:
            data = pd.read_csv(file, delimiter=';')
        merged_data = pd.concat([merged_data, data], ignore_index=True, )
        
    return merged_data['content'].apply(clean_text)

if __name__ == "__main__":
    get_data(folder_path)
