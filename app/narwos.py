import os
import shutil
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, session
import logging
from utils.module_functions import \
    validate_file, \
    validate_file_extension, \
    read_config
from utils.data_processing import process_data_from_choosen_files, save_raport_to_csv
import plotly.express as px
import json
import plotly
from utils.cluster import get_clusters_for_choosen_files
from utils.c_tf_idf_module import get_topics_from_texts


app = Flask(__name__)

app.config['CONFIG_FILE'] = 'CONFIG.yaml'

CONFIGURATION = read_config(
    app.config['CONFIG_FILE']
)

print(CONFIGURATION)

DIRECTORIES = CONFIGURATION.get('DIRECTORIES')
FILES = CONFIGURATION.get('FILES')
EMPTY_CONTENT_SETTINGS = CONFIGURATION.get('EMPTY_CONTENT_SETTINGS')
PIPELINE = CONFIGURATION.get('PIPELINE')
INPUT_FILES_SETTINGS = CONFIGURATION.get('INPUT_FILES_SETTINGS')
LOGGER = CONFIGURATION.get('LOGGER')
ML = CONFIGURATION.get('ML')

DATA_FOLDER = DIRECTORIES.get('data')
CLEARED_DATA_DIR = DIRECTORIES.get('cleared_files')
VALID_FILES_DIR = DIRECTORIES.get('valid_files')
TMP_DIR = DIRECTORIES.get('tmp')
EMBEDDINGS_DIR = DIRECTORIES.get('embeddings')
EMPTY_CONTENT_DIR = DIRECTORIES.get('empty_content')
FAISS_VECTORS_DIR = DIRECTORIES.get('faiss_vectors')
RAPORTS_DIR = DIRECTORIES.get('raports')
CURRENT_DF_DIR = DIRECTORIES.get('current_df')
SWEARWORDS_DIR = DIRECTORIES.get('swearwords_dir')

EMBEDDED_FILES = FILES.get('embedded_files')
CURRENT_DF_FILE = FILES.get('current_df')

EMPTY_CONTENTS_EXT = EMPTY_CONTENT_SETTINGS.get('empty_content_ext')
EMPTY_CONTENTS_SUFFIX = EMPTY_CONTENT_SETTINGS.get('empty_content_suffix')

BATCH_SIZE = PIPELINE.get('batch_size')
CONTENT_COLUMN = PIPELINE.get('content_column')
CLEARED_FILE_EXT = PIPELINE.get('cleared_file_ext')

LOGGING_FORMAT = LOGGER.get('logging_format')
LOGGER_LEVEL = LOGGER.get('logger_level')

ALLOWED_EXTENSIONS = INPUT_FILES_SETTINGS.get('allowed_extensions')
REQUIRED_COLUMNS = INPUT_FILES_SETTINGS.get('required_columns')

EMBEDDINGS_MODEL = ML.get('embeddings').get('model')
SEED = ML.get('seed')
SENTIMENT_MODEL_NAME = ML.get('sentiment').get('model_name')

PATH_TO_VALID_FILES = os.path.join(
    DATA_FOLDER,
    VALID_FILES_DIR
)

PATH_TO_CLEARED_FILES = os.path.join(
    DATA_FOLDER,
    CLEARED_DATA_DIR
)

PATH_TO_EMPTY_CONTENTS = os.path.join(
    DATA_FOLDER,
    EMPTY_CONTENT_DIR
)

PATH_TO_TMP_DIR = os.path.join(
    DATA_FOLDER,
    TMP_DIR
)

PATH_TO_RAPORTS_DIR = os.path.join(
    DATA_FOLDER,
    RAPORTS_DIR
)

PATH_TO_CURRENT_DF = os.path.join(
    DATA_FOLDER,
    CURRENT_DF_DIR,
    CURRENT_DF_FILE
)

PATH_PL_SWEARWORDS = os.path.join(
    DATA_FOLDER,
    SWEARWORDS_DIR,
    FILES.get('polish_swearwords')
)

PATH_EN_SWEARWORDS = os.path.join(
    DATA_FOLDER,
    SWEARWORDS_DIR,
    FILES.get('english_swearwords')
)

logging.basicConfig(
    level=LOGGER_LEVEL,
    format=LOGGING_FORMAT)

logger = logging.getLogger(__name__)

logger.debug(f'Required columns: {REQUIRED_COLUMNS}')

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config["TEMP_FOLDER"] = TEMP_FOLDER
# app.config["VALIDATED_FILES_FOLDER"] = VALIDATED_FILES_FOLDER

@app.route("/")
def index():

    message = request.args.get("message")

    validated_files = os.listdir(
        PATH_TO_VALID_FILES)

    return render_template(
        "index.html", 
        files=validated_files,
        message=message)

@app.route('/upload_file', methods=['POST'])
def upload_file():
        # Check if files were uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No files found in the request.'}), 400

    uploaded_files = request.files.getlist('file')
    files_uploading_status = {
        "uploaded_succesfully": [],
        "uploading_failed": {}}

    for uploaded_file in uploaded_files:

        if not uploaded_file:
            continue

        if validate_file_extension(
            filename=uploaded_file.filename, 
            allowed_extensions=ALLOWED_EXTENSIONS):

            file_path = os.path.join(
                PATH_TO_TMP_DIR, 
                uploaded_file.filename)
            uploaded_file.save(file_path)

            validated_file_content = validate_file(
                file_path,
                required_columns=REQUIRED_COLUMNS)

            if isinstance(validated_file_content, bool):
                
                shutil.move(
                    os.path.join(
                    PATH_TO_TMP_DIR, 
                    uploaded_file.filename),
                    os.path.join(
                    PATH_TO_VALID_FILES, 
                    uploaded_file.filename),
                )
                 
                files_uploading_status['uploaded_succesfully'].append(uploaded_file.filename)
                 
            else:
                os.remove(os.path.join(
                    PATH_TO_TMP_DIR, 
                    uploaded_file.filename))
        
                files_uploading_status['uploading_failed'][uploaded_file.filename] = validated_file_content


        else:
            files_uploading_status['uploading_failed'][uploaded_file.filename] = 'Extension is not valid'

    return redirect(url_for("index", message=files_uploading_status))

@app.route('/delete_file', methods=['POST'])
def delete_file():

    filename = request.form.get('to_delete')

    try:
        file_path = os.path.join(
            PATH_TO_VALID_FILES, 
            filename)
    except OSError:
        return redirect(url_for("index", message=f'File {filename} does not exist.'))

    if os.path.exists(file_path):

        os.remove(file_path)
        return redirect(url_for("index", message=f'File {filename} deleted successfully.'))

    return redirect(url_for("index", message=f'Cannot delete file {filename}.'))


@app.route('/choose_files_for_clusters', methods=['POST'])
def choose_files_for_clusters():

    files_for_clustering = request.form.getlist('chosen_files')

    logger.debug(f'Chosen files: {files_for_clustering}')

    swearwords = open(PATH_EN_SWEARWORDS, 'r').read().split('\n') + \
        open(PATH_PL_SWEARWORDS, 'r').read().split('\n')

    process_data_from_choosen_files(
        chosen_files=files_for_clustering,
        path_to_valid_files=PATH_TO_VALID_FILES,
        path_to_cleared_files=PATH_TO_CLEARED_FILES,
        path_to_empty_content_dir=PATH_TO_EMPTY_CONTENTS,
        path_to_embeddings_dir=EMBEDDINGS_DIR,
        faiss_vectors_dirname=FAISS_VECTORS_DIR,
        embedded_files_filename=EMBEDDED_FILES,
        embeddings_model_name=EMBEDDINGS_MODEL,
        sentiment_model_name=SENTIMENT_MODEL_NAME,
        swearwords=swearwords,
        content_column_name=CONTENT_COLUMN,
        cleread_file_ext=CLEARED_FILE_EXT,
        empty_contents_suffix=EMPTY_CONTENTS_SUFFIX,
        empty_content_ext=EMPTY_CONTENTS_EXT,
        batch_size=BATCH_SIZE,
        seed=SEED)

    clusters_df = get_clusters_for_choosen_files(
        chosen_files=files_for_clustering,
        path_to_cleared_files=PATH_TO_CLEARED_FILES,
        path_to_embeddings_dir=EMBEDDINGS_DIR,
        faiss_vectors_dirname=FAISS_VECTORS_DIR,
        embedded_files_filename=EMBEDDED_FILES,
        cleared_files_ext=CLEARED_FILE_EXT)
    
    clusters_df.to_parquet(
        index=False, 
        path=PATH_TO_CURRENT_DF)

    clusters_topics_df = get_topics_from_texts(
        df=clusters_df
    )

    save_raport_to_csv(
        df=clusters_df,
        path_to_raports_dir=PATH_TO_RAPORTS_DIR,
        clusters_topics=clusters_topics_df,
        filename=f'clusterization_raport_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.csv')

    logger.debug(f'Files {files_for_clustering} processed successfully.')

    n_clusters = len(clusters_df['labels'].unique())

    return redirect(url_for("index", message=f"{n_clusters} clusters has been created successfully."))


@app.route('/show_clusters_submit', methods=['POST'])
def show_clusters_submit():

    show_plot = request.form.get('show_plot')

    if show_plot:
        return redirect(url_for('show_clusters', show_plot=show_plot))
    else:
        return redirect(url_for("index", message=f"Cannot show the clusters!"))

@app.route('/show_clusters', methods=['GET'])
def show_clusters():

    df = pd.read_csv(PATH_TO_CURRENT_DF)

    if isinstance(df, pd.DataFrame):

        scatter_plot = px.scatter(
            data_frame=df,
            x='x',
            y='y',
            color='labels'
        )

        fig_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template("clusters_viz.html", figure=fig_json)
    
    return 'Nothing to show here'

if __name__ == '__main__':
    app.run(debug=True)





