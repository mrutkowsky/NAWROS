import pandas as pd
import glob
import re
import os
# Ścieżka do folderu, w którym znajdują się pliki
folder_path = os.getcwd() + '\data'

def clean_text(text):
    # Usunięcie znaków interpunkcyjnych
    cleaned_text = re.sub(r"[^\w\s]", "", re.sub(r"[\t\n]", "", text))

    return cleaned_text

def get_data(folder_path):
    # Pobieranie plików z rozszerzeniami .xlsx, .csv i .txt
    file_extension = ['xlsx', 'csv', 'txt']
    files = []
    for extension in file_extension:
        files.extend(glob.glob(f"{folder_path}/*.{extension}"))

    # Tworzenie pustej tabeli pandas, która będzie przechowywać scalone dane
    merged_data = pd.DataFrame(columns=["Questioned_Date", "Model_No", "OS", "SW_Version", "CSC", "Category", "Application_Name", "content"])
    # Iteracja przez wszystkie pliki i dodawanie ich zawartości do tabeli pandas
    for file in files:
        if file.endswith('.xlsx'):
            data = pd.read_excel(file)

            # data['content'] = data['content'].apply(clean_text)
        elif file.endswith('.csv'):
            data = pd.read_csv(file, delimiter=';')
            # data['content'] = data['content'].apply(clean_text)
        else:
            data = pd.read_csv(file, delimiter=';')
            # data['content'] = data['content'].apply(clean_text)

        merged_data = pd.concat([merged_data, data], ignore_index=True, )
    print(merged_data)
    return merged_data['content'].apply(clean_text)

if __name__ == "__main__":
    get_data(folder_path)