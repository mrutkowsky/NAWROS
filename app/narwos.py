import os
import shutil
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, session, make_response
import logging
from utils.module_functions import \
    validate_file, \
    validate_file_extension, \
    read_config
from utils.data_processing import process_data_from_choosen_files, save_raport_to_csv, get_stopwords, read_file, del_file_from_embeded
import plotly.express as px
import json
import plotly
from utils.cluster import get_clusters_for_choosen_files
from utils.c_tf_idf_module import get_topics_from_texts
from utils.filtering import write_file, show_columns_for_filtering
from copy import copy

app = Flask(__name__)

app.config['CONFIG_FILE'] = 'CONFIG.yaml'

CONFIGURATION = read_config(
    app.config['CONFIG_FILE']
)

DIRECTORIES = CONFIGURATION.get('DIRECTORIES')
FILES = CONFIGURATION.get('FILES')
EMPTY_CONTENT_SETTINGS = CONFIGURATION.get('EMPTY_CONTENT_SETTINGS')
PIPELINE = CONFIGURATION.get('PIPELINE')
INPUT_FILES_SETTINGS = CONFIGURATION.get('INPUT_FILES_SETTINGS')
LOGGER = CONFIGURATION.get('LOGGER')
ML = CONFIGURATION.get('ML')
FILTERING = CONFIGURATION.get('FILTERING')

DATA_FOLDER = DIRECTORIES.get('data')
CLEARED_DATA_DIR = DIRECTORIES.get('cleared_files')
VALID_FILES_DIR = DIRECTORIES.get('valid_files')
TMP_DIR = DIRECTORIES.get('tmp')
EMBEDDINGS_DIR = DIRECTORIES.get('embeddings')
EMPTY_CONTENT_DIR = DIRECTORIES.get('empty_content')
FAISS_VECTORS_DIR = DIRECTORIES.get('faiss_vectors')
RAPORTS_DIR = DIRECTORIES.get('raports')
CURRENT_DF_DIR = DIRECTORIES.get('current_df')
FILTERED_DF_DIR = DIRECTORIES.get('filtered_df')
STOPWORDS_DIR = DIRECTORIES.get('stop_words')
SWEARWORDS_DIR = DIRECTORIES.get('swearwords_dir')
STOPWORDS_DIR = DIRECTORIES.get('stop_words')

EMBEDDED_JSON = FILES.get('embedded_json')
CURRENT_DF_FILE = FILES.get('current_df')
FILTERED_DF_FILE = FILES.get('filtered_df')

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

FILTERING_DOWNLOAD_NAME = FILTERING.get('download_name')

UMAP = ML.get('UMAP')
HDBSCAN = ML.get('HDBSCAN')

RAPORT_CONFIG = CONFIGURATION.get('RAPORT_SETTINGS')
RAPORT_COLUMNS = RAPORT_CONFIG.get('columns')

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

PATH_TO_FAISS_VECTORS_DIR = os.path.join(
    EMBEDDINGS_DIR,
    FAISS_VECTORS_DIR
)

PATH_TO_CURRENT_DF = os.path.join(
    DATA_FOLDER,
    CURRENT_DF_DIR,
    CURRENT_DF_FILE
)

PATH_TO_FILTERED_DF = os.path.join(
    DATA_FOLDER,
    FILTERED_DF_DIR,
    FILTERED_DF_FILE
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

@app.route("/")
def index():

    message = request.args.get("message")
    success_upload = request.args.get("success_upload")
    failed_upload = request.args.get("failed_upload")

    if success_upload is not None:
        success_upload = json.loads(success_upload)

    if failed_upload is not None:
        failed_upload = json.loads(failed_upload)

    validated_files = os.listdir(
        PATH_TO_VALID_FILES)
        
    validated_files_to_show = [
        v_file for v_file in validated_files 
        if os.path.splitext(v_file)[-1].lower() in ALLOWED_EXTENSIONS
    ]

    return render_template(
        "index.html", 
        files=validated_files_to_show,
        message=message,
        success_upload=success_upload, 
        failed_upload=failed_upload)

@app.route('/upload_file', methods=['POST'])
def upload_file():
        # Check if files were uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No files found in the request.'}), 400

    uploaded_files = request.files.getlist('file')
    files_uploading_status = {
        "uploaded_successfully": [],
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
                 
                files_uploading_status['uploaded_successfully'].append(uploaded_file.filename)
                 
            else:
                os.remove(os.path.join(
                    PATH_TO_TMP_DIR, 
                    uploaded_file.filename))
        
                files_uploading_status['uploading_failed'][uploaded_file.filename] = validated_file_content


        else:
            files_uploading_status['uploading_failed'][uploaded_file.filename] = 'Extension is not valid'

    # successfully_uploaded_message = f"Files uploaded succesfully:<ol><li>{'</li><li>'.join(files_uploading_status.get('uploaded_succesfully'))}</li></ol>"
    # uploading_failed_message = f"Failed to upload files:<ol><li>{'</li><li>'.join([f'{key} - {value}' for key, value in files_uploading_status.get('uploading_failed').items()])}</li></ol>"
    # final_message = f"{successfully_uploaded_message}<br>{uploading_failed_message}"

    success_upload = copy(files_uploading_status['uploaded_successfully'])
    failed_upload = [f"{key} - {value}" for key, value in files_uploading_status.get('uploading_failed').items()]

    logger.info(success_upload)
    logger.info(failed_upload)

    return redirect(url_for("index", success_upload=json.dumps(success_upload), failed_upload=json.dumps(failed_upload)))

@app.route('/delete_file', methods=['POST'])
def delete_file():

    filename = request.form.get('to_delete')
    base_filename = os.path.splitext(filename)[0]

    logger.debug(f'Filename: {filename}, base name: {base_filename}')

    try:
        file_path = os.path.join(
            PATH_TO_VALID_FILES, 
            filename)
    except OSError:
        return redirect(url_for("index", message=f'File {filename} does not exist.'))
    else:

        os.remove(file_path)

        for data_dir in [PATH_TO_CLEARED_FILES, PATH_TO_FAISS_VECTORS_DIR]:

            logger.debug(f'Current data dir: {data_dir}')

            filename_search_dir = {os.path.splitext(file_)[0]: file_ for file_ in os.listdir(data_dir)}

            logger.debug(f'Current {data_dir} search dir: {filename_search_dir}')

            if base_filename in filename_search_dir:

                logger.debug(f'File name with extension from dict: {filename_search_dir.get(base_filename)}')

                file_path = os.path.join(
                    data_dir, 
                    filename_search_dir.get(base_filename))

                if os.path.exists(file_path):

                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f'Can not del {filename} from {data_dir} dir! - {e}')
                    else:
                        logger.info(f'Successfully deleted file from {data_dir}')

        
        deleted_successfully_from_json = del_file_from_embeded(
            filename_to_del=filename,
            path_to_embeddings_file=os.path.join(EMBEDDINGS_DIR, EMBEDDED_JSON)
        )

        if deleted_successfully_from_json:

            logger.info(f'Successfully deleted {filename} from JSON file')
        
        return redirect(url_for("index", message=f'File {filename} deleted successfully.'))
    


@app.route('/choose_files_for_clusters', methods=['POST'])
def choose_files_for_clusters():

    files_for_clustering = request.form.getlist('chosen_files')
    swearwords = open(PATH_EN_SWEARWORDS, 'r').read().split('\n') + \
        open(PATH_PL_SWEARWORDS, 'r').read().split('\n')
    logger.debug(f'Chosen files: {files_for_clustering}')

    process_data_from_choosen_files(
        chosen_files=files_for_clustering,
        path_to_valid_files=PATH_TO_VALID_FILES,
        path_to_cleared_files=PATH_TO_CLEARED_FILES,
        path_to_empty_content_dir=PATH_TO_EMPTY_CONTENTS,
        path_to_embeddings_dir=EMBEDDINGS_DIR,
        faiss_vectors_dirname=FAISS_VECTORS_DIR,
        embedded_files_filename=EMBEDDED_JSON,
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
        embedded_files_filename=EMBEDDED_JSON,
        cleared_files_ext=CLEARED_FILE_EXT,
        random_state=SEED,
        n_neighbors=UMAP.get('n_neighbors'),
        min_dist=UMAP.get('min_dist'),
        n_components=UMAP.get('n_components'),
        min_cluster_size=HDBSCAN.get('min_cluster_size'),
        min_samples=HDBSCAN.get('min_samples'),
        metric=HDBSCAN.get('metric'),                      
        cluster_selection_method=HDBSCAN.get('cluster_selection_method')
    ).astype({col: str for col in REQUIRED_COLUMNS})
    
    clusters_df.to_parquet(
        index=False, 
        path=PATH_TO_CURRENT_DF)

    clusters_topics_df = get_topics_from_texts(
        df=clusters_df,
        stop_words=get_stopwords(STOPWORDS_DIR)
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

    df = read_file(PATH_TO_CURRENT_DF)

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

@app.route('/show_filters_submit', methods=['POST'])
def show_filter_submit():

    show_filter = request.form.get('show_filter')
    if show_filter:
        return redirect(url_for('show_filter', show_filter=show_filter))
    else:
        return redirect(url_for("index", message=f"Cannot show the filtering!"))

@app.route('/show_filter', methods=['GET'])
def show_filter():

    if request.method == 'GET':
        if isinstance(RAPORT_COLUMNS, list):

            return render_template('filtering.html', columns=RAPORT_COLUMNS)
        
        return 'Nothing to show here'
    
@app.route('/apply_filter', methods=['POST'])
def apply_filter():

    filters = request.get_json()

    filtered_df = read_file(PATH_TO_CURRENT_DF, columns=RAPORT_COLUMNS)
    filtered_df = filtered_df.astype(str)

    logger.info(filtered_df)

    logger.info(
        f"""{filters} Columns of df to filter:{filtered_df.columns}""")
    
    query_string = ' & '.join(
        [f"{filter_dict.get('column')} == '{filter_dict.get('value')}'" for filter_dict in filters]
    )

    logger.info(f"{query_string}")

    filtered_df = filtered_df.query(query_string)

    logger.info(filtered_df)

    filtered_df.to_csv(
        index=False, 
        path_or_buf=PATH_TO_FILTERED_DF)
    
    # filtered_df_excluded = show_columns_for_filtering(PATH_TO_CURRENT_DF)
    message = f"{filters} filters has been applied successfully."

    print(message)

    return render_template(
        'filtering.html', 
        columns=RAPORT_COLUMNS, 
        message=message)


@app.route('/filter_download_report', methods=['POST'])
def filter_data_download_report():

    report_type = request.get_json()
    filtered_df = read_file(PATH_TO_FILTERED_DF)
    file_type = report_type['reportType']
    # Prepare the CSV file for download
    output = write_file(filtered_df, file_type)
    response = make_response(send_file(
        output,
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment = True,
        download_name = FILTERING_DOWNLOAD_NAME
    ))
    return response

if __name__ == '__main__':
    app.run(debug=True)





