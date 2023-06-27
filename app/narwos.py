import os
import shutil
import pandas as pd
from flask import Flask, request, render_template, send_file, redirect, url_for, make_response
import logging
from utils.module_functions import \
    validate_file, \
    validate_file_extension, \
    read_config
from utils.data_processing import process_data_from_choosen_files, \
    get_stopwords, \
    read_file, \
    del_file_from_embeded, \
    get_swearwords, \
    get_rows_cardinalities, \
    set_rows_cardinalities
import plotly.express as px
import json
import plotly
from utils.cluster import get_clusters_for_choosen_files, \
    get_cluster_labels_for_new_file, \
    cluster_recalculation_needed, \
    cns_after_clusterization
from utils.filtering import write_file
from utils.reports import compare_reports
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
CLUSTER_EXEC_REPORTS_DIR = DIRECTORIES.get('cluster_exec_reports')
COMPARING_REPORTS_DIR = DIRECTORIES.get('comparing_reports')
CURRENT_DF_DIR = DIRECTORIES.get('current_df')
FILTERED_DF_DIR = DIRECTORIES.get('filtered_df')
STOPWORDS_DIR = DIRECTORIES.get('stop_words')
SWEARWORDS_DIR = DIRECTORIES.get('swearwords_dir')
STOPWORDS_DIR = DIRECTORIES.get('stop_words')

EMBEDDED_JSON = FILES.get('embedded_json')
CURRENT_DF_FILE = FILES.get('current_df')
FILTERED_DF_FILE = FILES.get('filtered_df')
TOPICS_DF_FILE = FILES.get('topics_df')
ROWS_CARDINALITIES_FILE = FILES.get('rows_cardinalities')

EMPTY_CONTENTS_EXT = EMPTY_CONTENT_SETTINGS.get('empty_content_ext')
EMPTY_CONTENTS_SUFFIX = EMPTY_CONTENT_SETTINGS.get('empty_content_suffix')

BATCH_SIZE = PIPELINE.get('batch_size')
CONTENT_COLUMN = PIPELINE.get('content_column')
CLEARED_FILE_EXT = PIPELINE.get('cleared_file_ext')

LOGGING_FORMAT = LOGGER.get('logging_format')
LOGGER_LEVEL = LOGGER.get('logger_level')

ALLOWED_EXTENSIONS = INPUT_FILES_SETTINGS.get('allowed_extensions')
REQUIRED_COLUMNS = INPUT_FILES_SETTINGS.get('required_columns')

EMBEDDINGS_MODEL = ML.get('embeddings').get('model_name')
SEED = ML.get('seed')
SENTIMENT_MODEL_NAME = ML.get('sentiment').get('model_name')

FILTERING_DOWNLOAD_NAME = FILTERING.get('download_name')

UMAP = ML.get('UMAP')
DIM_REDUCER_MODEL_NAME = UMAP.get('dim_reducer_model_name')
REDUCER_2D_MODEL_NAME = UMAP.get('reducer_2d_model_name')

HDBSCAN_SETTINGS = ML.get('HDBSCAN_SETTINGS')
HDBSCAN_MODEL_NAME = HDBSCAN_SETTINGS.get('model_name')
RECALCULATE_CLUSTERS_TRESHOLD = ML.get('recalculate_clusters_treshold')

REPORT_CONFIG = CONFIGURATION.get('REPORT_SETTINGS')

BASE_REPORT_COLUMNS = REPORT_CONFIG.get('base_columns')
LABELS_COLUMN = REPORT_CONFIG.get('labels_column', 'labels')
CARDINALITIES_COLUMN = REPORT_CONFIG.get('cardinalities_column', 'counts')
SENTIMENT_COLUMN = REPORT_CONFIG.get('sentiment_column', 'sentiment')
FILENAME_COLUMN = REPORT_CONFIG.get('filename_column' 'filename')

ALL_REPORT_COLUMNS = BASE_REPORT_COLUMNS + [
    LABELS_COLUMN, 
    CARDINALITIES_COLUMN, 
    SENTIMENT_COLUMN, 
    FILENAME_COLUMN
]

CLUSTER_EXEC_FILENAME_PREFIX = REPORT_CONFIG.get('cluster_exec_filename_prefix')
CLUSTER_EXEC_FILENAME_EXT = REPORT_CONFIG.get('cluster_exec_filename_ext')

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

PATH_TO_CLUSTER_EXEC_REPORTS_DIR = os.path.join(
    DATA_FOLDER,
    CLUSTER_EXEC_REPORTS_DIR
)

PATH_TO_COMPARING_REPORTS_DIR = os.path.join(
    DATA_FOLDER,
    COMPARING_REPORTS_DIR
)

PATH_TO_FAISS_VECTORS_DIR = os.path.join(
    EMBEDDINGS_DIR,
    FAISS_VECTORS_DIR
)

PATH_TO_CURRENT_DF_DIR = os.path.join(
    DATA_FOLDER,
    CURRENT_DF_DIR,
)

PATH_TO_CURRENT_DF = os.path.join(
    DATA_FOLDER,
    CURRENT_DF_DIR,
    CURRENT_DF_FILE
)

PATH_TO_ROWS_CARDINALITIES = os.path.join(
    DATA_FOLDER,
    CURRENT_DF_DIR,
    ROWS_CARDINALITIES_FILE,
)

PATH_TO_FILTERED_DF = os.path.join(
    DATA_FOLDER,
    FILTERED_DF_DIR,
    FILTERED_DF_FILE
)

PATH_TO_SWEARWORDS_DIR = os.path.join(
    DATA_FOLDER,
    SWEARWORDS_DIR,
)

PATH_TO_STOPWORDS_DIR = os.path.join(
    DATA_FOLDER,
    STOPWORDS_DIR,
)

STOP_WORDS = get_stopwords(PATH_TO_STOPWORDS_DIR)
SWEAR_WORDS = get_swearwords(PATH_TO_SWEARWORDS_DIR)

logging.basicConfig(
    level=LOGGER_LEVEL,
    format=LOGGING_FORMAT)

logger = logging.getLogger(__name__)

logger.debug(f'Required columns: {REQUIRED_COLUMNS}')

def upload_and_validate_files(
        uploaded_files: list):

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

    success_upload = copy(files_uploading_status['uploaded_successfully'])
    failed_upload = [f"{key} - {value}" for key, value in files_uploading_status.get('uploading_failed').items()]

    return success_upload, failed_upload

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
    if len(request.files.getlist('file')) == 1:

        if request.files.getlist('file')[0].filename == '':
            return redirect(url_for("index", message=f'No file has been selected for uploading!'))
    
    logger.debug(f"Uploaded files: {request.files.getlist('file')}")

    uploaded_files = request.files.getlist('file')

    success_upload, failed_upload = upload_and_validate_files(
        uploaded_files=uploaded_files
    )

    logger.debug(success_upload)
    logger.debug(failed_upload)

    return redirect(url_for("index", success_upload=json.dumps(success_upload), failed_upload=json.dumps(failed_upload)))

@app.route('/delete_file', methods=['POST'])
def delete_file():

    filename = request.form.get('to_delete')

    if not filename:
        return redirect(url_for("index", message=f'No file has been selected for deletion!'))

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
                        return redirect(url_for("index", message=f'Can not del {filename} from {data_dir} dir!'))
                    else:
                        logger.info(f'Successfully deleted file from {data_dir}')
 
        deleted_successfully_from_json = del_file_from_embeded(
            filename_to_del=filename,
            path_to_embeddings_file=os.path.join(EMBEDDINGS_DIR, EMBEDDED_JSON)
        )

        if deleted_successfully_from_json:

            logger.info(f'Successfully deleted {filename} from JSON file')

        else:

            logger.error(f'Failed to delete {filename} from JSON file')
            return redirect(url_for("index", message=f'Failed to delete {filename} from JSON file.'))
        
        return redirect(url_for("index", message=f'File {filename} deleted successfully.'))
    
@app.route('/choose_files_for_clusters', methods=['POST'])
def choose_files_for_clusters():

    files_for_clustering = request.form.getlist('chosen_files')

    if not files_for_clustering:
        return redirect(url_for("index", message=f'No file has been selected for clustering!'))
    
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
        swearwords=SWEAR_WORDS,
        content_column_name=CONTENT_COLUMN,
        cleread_file_ext=CLEARED_FILE_EXT,
        empty_contents_suffix=EMPTY_CONTENTS_SUFFIX,
        empty_content_ext=EMPTY_CONTENTS_EXT,
        batch_size=BATCH_SIZE,
        seed=SEED)

    new_current_df = get_clusters_for_choosen_files(
        chosen_files=files_for_clustering,
        path_to_cleared_files=PATH_TO_CLEARED_FILES,
        path_to_embeddings_dir=EMBEDDINGS_DIR,
        path_to_current_df_dir=PATH_TO_CURRENT_DF_DIR,
        rows_cardinalities_file=ROWS_CARDINALITIES_FILE,
        faiss_vectors_dirname=FAISS_VECTORS_DIR,
        embedded_files_filename=EMBEDDED_JSON,
        cleared_files_ext=CLEARED_FILE_EXT,
        random_state=SEED,
        n_neighbors=UMAP.get('n_neighbors'),
        min_dist=UMAP.get('min_dist'),
        n_components=UMAP.get('n_components'),
        min_cluster_size=HDBSCAN_SETTINGS.get('min_cluster_size'),
        min_samples=HDBSCAN_SETTINGS.get('min_samples'),
        metric=HDBSCAN_SETTINGS.get('metric'),                      
        cluster_selection_method=HDBSCAN_SETTINGS.get('cluster_selection_method')
    ).astype({col: str for col in REQUIRED_COLUMNS})
    
    cns_after_clusterization(
        new_current_df=new_current_df,
        path_to_current_df_dir=PATH_TO_CURRENT_DF_DIR,
        path_to_cluster_exec_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR,
        topic_df_file_name=TOPICS_DF_FILE,
        current_df_filename=CURRENT_DF_FILE,
        stop_words=STOP_WORDS,
        labels_column=LABELS_COLUMN,
        cardinalities_column=CARDINALITIES_COLUMN,
        cluster_exec_filename_prefix=CLUSTER_EXEC_FILENAME_PREFIX,
        cluster_exec_filename_ext=CLUSTER_EXEC_FILENAME_EXT
    )

    logger.debug(f'Files {files_for_clustering} processed successfully.')

    n_clusters = len(new_current_df[LABELS_COLUMN].unique())

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
            color=LABELS_COLUMN
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
        if isinstance(ALL_REPORT_COLUMNS, list):

            return render_template('filtering.html', columns=ALL_REPORT_COLUMNS)
        
        return 'Nothing to show here'
    
@app.route('/apply_filter', methods=['POST'])
def apply_filter():

    filters = request.get_json()

    filtered_df = read_file(PATH_TO_CURRENT_DF, columns=ALL_REPORT_COLUMNS)
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

    return render_template(
        'filtering.html', 
        columns=ALL_REPORT_COLUMNS, 
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

@app.route('/update_clusters_new_file', methods=['POST'])
def update_clusters_new_file():
    
    uploaded_file = request.files['file']
    filename = uploaded_file.filename

    # if os.path.exists(os.path.join(PATH_TO_VALID_FILES, filename)):
    #     return redirect(url_for("index", message=f"File {filename} already exists in File Storage, choose another file"))
    
    # else:

    success_upload, _ = upload_and_validate_files(
        uploaded_files=[uploaded_file]
    )

    logger.debug(f'File {filename} has been validated')

    if success_upload:

        rows_cards_for_preprocessed = process_data_from_choosen_files(
            chosen_files=[filename],
            path_to_valid_files=PATH_TO_VALID_FILES,
            path_to_cleared_files=PATH_TO_CLEARED_FILES,
            path_to_empty_content_dir=PATH_TO_EMPTY_CONTENTS,
            path_to_embeddings_dir=EMBEDDINGS_DIR,
            faiss_vectors_dirname=FAISS_VECTORS_DIR,
            embedded_files_filename=EMBEDDED_JSON,
            embeddings_model_name=EMBEDDINGS_MODEL,
            sentiment_model_name=SENTIMENT_MODEL_NAME,
            swearwords=SWEAR_WORDS,
            content_column_name=CONTENT_COLUMN,
            cleread_file_ext=CLEARED_FILE_EXT,
            empty_contents_suffix=EMPTY_CONTENTS_SUFFIX,
            empty_content_ext=EMPTY_CONTENTS_EXT,
            batch_size=BATCH_SIZE,
            seed=SEED)
        
        n_of_rows_for_new_file = rows_cards_for_preprocessed.get(filename)

        rows_cardinalities_current_df = get_rows_cardinalities(
            path_to_cardinalities_file=PATH_TO_ROWS_CARDINALITIES
        )

        need_to_recalculate = cluster_recalculation_needed(
            n_of_rows=n_of_rows_for_new_file,
            rows_cardinalities_current_df=rows_cardinalities_current_df,
            recalculate_treshold=RECALCULATE_CLUSTERS_TRESHOLD
        )

        if not need_to_recalculate:

            logger.info(f'Treshold {RECALCULATE_CLUSTERS_TRESHOLD} has not been exceeded, assigning topics based on current_df clusterization')

            new_current_df = get_cluster_labels_for_new_file(
                filename=filename,
                path_to_current_df=PATH_TO_CURRENT_DF,
                path_to_current_df_dir=PATH_TO_CURRENT_DF_DIR,
                path_to_cleared_files_dir=PATH_TO_CLEARED_FILES,
                path_to_faiss_vetors_dir=PATH_TO_FAISS_VECTORS_DIR,
                required_columns=REQUIRED_COLUMNS,
                clusterer_model_name=HDBSCAN_MODEL_NAME,
                umap_model_name=DIM_REDUCER_MODEL_NAME,
                reducer_2d_model_name=REDUCER_2D_MODEL_NAME,
                cleared_files_ext=CLEARED_FILE_EXT
            )

            rows_cardinalities_current_df['only_classified'][filename] = n_of_rows_for_new_file
            set_rows_cardinalities(
                path_to_cardinalities_file=PATH_TO_ROWS_CARDINALITIES,
                updated_cardinalities=rows_cardinalities_current_df
            )

            logger.info('Successfully updated rows cardinalities file for current_df')

            cns_after_clusterization(
                new_current_df=new_current_df,
                path_to_current_df_dir=PATH_TO_CURRENT_DF_DIR,
                path_to_cluster_exec_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR,
                current_df_filename=CURRENT_DF_FILE,
                stop_words=STOP_WORDS,
                labels_column=LABELS_COLUMN,
                cardinalities_column=CARDINALITIES_COLUMN,
                cluster_exec_filename_prefix=CLUSTER_EXEC_FILENAME_PREFIX,
                cluster_exec_filename_ext=CLUSTER_EXEC_FILENAME_EXT,
                topic_df_file_name=TOPICS_DF_FILE,
                only_update=True
            )

        else:

            logger.info(f'Treshold {RECALCULATE_CLUSTERS_TRESHOLD} has been exceeded, recalculating clusters for current_df')

            all_current_df_files = [file_ for in_dict in rows_cardinalities_current_df.values() for file_ in in_dict.keys()]

            new_current_df = get_clusters_for_choosen_files(
                chosen_files= all_current_df_files + [filename],
                path_to_cleared_files=PATH_TO_CLEARED_FILES,
                path_to_embeddings_dir=EMBEDDINGS_DIR,
                path_to_current_df_dir=PATH_TO_CURRENT_DF_DIR,
                rows_cardinalities_file=ROWS_CARDINALITIES_FILE,
                faiss_vectors_dirname=FAISS_VECTORS_DIR,
                embedded_files_filename=EMBEDDED_JSON,
                cleared_files_ext=CLEARED_FILE_EXT,
                random_state=SEED,
                n_neighbors=UMAP.get('n_neighbors'),
                min_dist=UMAP.get('min_dist'),
                n_components=UMAP.get('n_components'),
                min_cluster_size=HDBSCAN_SETTINGS.get('min_cluster_size'),
                min_samples=HDBSCAN_SETTINGS.get('min_samples'),
                metric=HDBSCAN_SETTINGS.get('metric'),                      
                cluster_selection_method=HDBSCAN_SETTINGS.get('cluster_selection_method')
            ).astype({col: str for col in REQUIRED_COLUMNS})
            
            cns_after_clusterization(
                new_current_df=new_current_df,
                path_to_current_df_dir=PATH_TO_CURRENT_DF_DIR,
                path_to_cluster_exec_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR,
                topic_df_file_name=TOPICS_DF_FILE,
                current_df_filename=CURRENT_DF_FILE,
                stop_words=STOP_WORDS,
                labels_column=LABELS_COLUMN,
                cardinalities_column=CARDINALITIES_COLUMN,
                cluster_exec_filename_prefix=CLUSTER_EXEC_FILENAME_PREFIX,
                cluster_exec_filename_ext=CLUSTER_EXEC_FILENAME_EXT
            )

            n_clusters = len(new_current_df[LABELS_COLUMN].unique())

            return redirect(url_for("index", message=f"{n_clusters} clusters has been created successfully."))
        
    else:
        logger.error(f'Can not preprocess file {filename}')
        return redirect(url_for("index", message=f'Can not upload file {filename} for clusters update!'))

    return redirect(url_for("index", message=f"Cluster labels for {filename} have been successfully assigned."))
    
@app.route('/compare_selected_reports', methods=['POST'])
def compare_selected_reports():

    filename1 = request.form.get('field1')
    filename2 = request.form.get('field2')

    comparison_result_df = compare_reports(
        first_report_name=filename1,
        second_report_name=filename2,
        path_to_reports_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR
    )

    logger.debug(comparison_result_df)

    comparison_report_filename = f"{filename1.split('.')[0]}__{filename2.split('.')[0]}_comparison.csv"

    comparison_result_df.to_csv(
        path_or_buf=os.path.join(PATH_TO_COMPARING_REPORTS_DIR, comparison_report_filename),
        index=False
    )

    return 'ok'


if __name__ == '__main__':
    app.run(debug=True)





