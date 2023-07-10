import os
import shutil
import pandas as pd
from flask import Flask, request, render_template, send_file, redirect, url_for, make_response, jsonify
import logging
from utils.module_functions import \
    validate_file, \
    validate_file_extension, \
    read_config, \
    get_report_ext
from utils.data_processing import process_data_from_choosen_files, \
    get_stopwords, \
    read_file, \
    del_file_from_embeded, \
    get_swearwords, \
    get_rows_cardinalities, \
    set_rows_cardinalities, \
    save_df_to_file, \
    get_n_of_rows_df, \
    find_filename_in_dir, \
    get_report_name_with_timestamp, \
    create_response_report, \
    prepare_filters, \
    prepare_reports_to_chose, \
    apply_filters_on_df
import plotly.express as px
import json
import plotly
from utils.cluster import get_clusters_for_choosen_files, \
    get_cluster_labels_for_new_file, \
    cluster_recalculation_needed, \
    cns_after_clusterization, \
    save_cluster_exec_report
from utils.reports import compare_reports, find_latest_two_reports, \
    find_latested_n_exec_report
from utils.comparison_pdf_report import create_pdf_comaprison_report
from copy import copy
import datetime

app = Flask(__name__)

DATE_FILTER_FORMAT = '%Y-%m-%d'
USED_AS_BASE_KEY = "used_as_base"
ONLY_CLASSIFIED_KEY = "only_classified"
TOPICS_CONCAT_FOR_VIZ = 'topics'
MIN_CLUSTER_SAMPLES = 15
ZIP_EXT = '.zip'

ALLOWED_EXTENSIONS = [
    ".csv",
    ".xlsx"
]

DEFAULT_REPORT_FORMAT_SETTINGS = {
    "ext": ".csv", "mimetype": "text/csv"
}

GITKEEP_FILE = '.gitkeep'

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
EMPTY_CONTENT_ARCHIVE_DIR = DIRECTORIES.get('empty_content_archive')
FAISS_VECTORS_DIR = DIRECTORIES.get('faiss_vectors')
CLUSTER_EXEC_REPORTS_DIR = DIRECTORIES.get('cluster_exec_reports')
COMPARING_REPORTS_DIR = DIRECTORIES.get('comparing_reports')
CURRENT_DF_DIR = DIRECTORIES.get('current_df')
FILTERED_DF_DIR = DIRECTORIES.get('filtered_df')
STOPWORDS_DIR = DIRECTORIES.get('stop_words')
SWEARWORDS_DIR = DIRECTORIES.get('swearwords_dir')
STOPWORDS_DIR = DIRECTORIES.get('stop_words')
DETAILED_CLUSTER_EXEC_REPORTS = DIRECTORIES.get('detailed_cluster_exec_reports')
DETAILED_FILTERED_REPORTS = DIRECTORIES.get('detailed_filtered_reports')

ALLOWED_REPORT_EXT_DIR = DIRECTORIES.get('allowed_reports_formats')

EMBEDDED_JSON = FILES.get('embedded_json')
CURRENT_DF_FILE = FILES.get('current_df')
FILTERED_DF_FILE = FILES.get('filtered_df')
TOPICS_DF_FILE = FILES.get('topics_df')
ROWS_CARDINALITIES_FILE = FILES.get('rows_cardinalities')
EXT_MAPPINGS_FILE = FILES.get('ext_mappings')

EMPTY_CONTENTS_EXT = EMPTY_CONTENT_SETTINGS.get('empty_content_ext')
EMPTY_CONTENTS_SUFFIX = EMPTY_CONTENT_SETTINGS.get('empty_content_suffix')

BATCH_SIZE = PIPELINE.get('batch_size')
CLEARED_FILE_EXT = PIPELINE.get('cleared_file_ext')

LOGGING_FORMAT = LOGGER.get('logging_format')
LOGGER_LEVEL = LOGGER.get('logger_level')

ETL_SETTINGS = CONFIGURATION.get('ETL_SETTINGS')
TRANSLATE_CONTENT = ETL_SETTINGS.get('translate')
GET_SENTIMENT = ETL_SETTINGS.get('sentiment')

REQUIRED_COLUMNS = INPUT_FILES_SETTINGS.get('required_columns')

EMBEDDINGS_MODEL = ML.get('embeddings').get('model_name')
SEED = ML.get('seed')
SENTIMENT_MODEL_NAME = ML.get('sentiment').get('model_name')

UMAP = ML.get('UMAP')
DIM_REDUCER_MODEL_NAME = UMAP.get('dim_reducer_model_name')
REDUCER_2D_MODEL_NAME = UMAP.get('reducer_2d_model_name')

HDBSCAN_SETTINGS = ML.get('HDBSCAN_SETTINGS')
HDBSCAN_MODEL_NAME = HDBSCAN_SETTINGS.get('model_name')
RECALCULATE_CLUSTERS_TRESHOLD = ML.get('recalculate_clusters_treshold')
OUTLIER_TRESHOLD = HDBSCAN_SETTINGS.get('outlier_treshold')

LANG_DETECTION_MODEL = ML.get('lang_detection_model')

TRANSLATION_MODELS = ML.get('translation_models')

REPORT_CONFIG = CONFIGURATION.get('REPORT_SETTINGS')

BASE_REPORT_COLUMNS = REPORT_CONFIG.get('base_columns')
ORIGINAL_CONTENT_COLUMN = REPORT_CONFIG.get('original_content_column')
PREPROCESSED_CONTENT_COLUMN = REPORT_CONFIG.get('preprocessed_content_column')
LABELS_COLUMN = REPORT_CONFIG.get('labels_column', 'labels')
CARDINALITIES_COLUMN = REPORT_CONFIG.get('cardinalities_column', 'counts')
SENTIMENT_COLUMN = REPORT_CONFIG.get('sentiment_column', 'sentiment')
FILENAME_COLUMN = REPORT_CONFIG.get('filename_column', 'filename')
ORIGINAL_CONTENT = REPORT_CONFIG.get('original_content')
DATE_COLUMN = REPORT_CONFIG.get('date_column')
CLUSTER_SUMMARY_COLUMN = REPORT_CONFIG.get('cluster_summary_column', 'cluster_summary')

COMPARING_RAPORT_DOWNLOAD_NAME = REPORT_CONFIG.get('download_name')
NO_TOPIC_TOKEN = REPORT_CONFIG.get('no_topic_token')
COMPARING_REPORT_SUFFIX = REPORT_CONFIG.get('comparing_report_suffix')

TOPIC_COLUMN_PREFIX = REPORT_CONFIG.get('topic_column_prefix')

TOPICS_RANGE = range(1, 6)
ALL_DETAILED_REPORT_COLUMNS = [DATE_COLUMN] + BASE_REPORT_COLUMNS + [
    ORIGINAL_CONTENT_COLUMN,
    PREPROCESSED_CONTENT_COLUMN,
    LABELS_COLUMN,  
    FILENAME_COLUMN] + [f"{TOPIC_COLUMN_PREFIX}_{i}" for i in TOPICS_RANGE]

if GET_SENTIMENT:
    ALL_DETAILED_REPORT_COLUMNS += [SENTIMENT_COLUMN]

CLUSTER_EXEC_FILENAME_PREFIX = REPORT_CONFIG.get('cluster_exec_filename_prefix')
CLUSTER_EXEC_FILENAME_EXT = REPORT_CONFIG.get('cluster_exec_filename_ext')

DETAILED_CLUSTER_EXEC_FILENAME_PREFIX = REPORT_CONFIG.get('detailed_cluster_exec_filename_prefix')

FILTERED_REPORT_PREFIX = REPORT_CONFIG.get('filtered_filename_prefix')
FILTERED_FILENAME_EXT = REPORT_CONFIG.get('filtered_filename_ext')

COLS_FOR_LABEL = [f"New_{TOPIC_COLUMN_PREFIX}_{i}" for i in TOPICS_RANGE]
COLS_FOR_OLD_LABEL = [f"Old_{TOPIC_COLUMN_PREFIX}_{i}" for i in TOPICS_RANGE]
OLD_COL_NAME = 'Old_' + CARDINALITIES_COLUMN
NEW_COL_NAME = 'New_' + CARDINALITIES_COLUMN


REPORT_FORMATS_MAPPING = get_report_ext(
    ALLOWED_REPORT_EXT_DIR,
    EXT_MAPPINGS_FILE
)

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

PATH_TO_EMPTY_ARCHIVE = os.path.join(
    DATA_FOLDER,
    EMPTY_CONTENT_ARCHIVE_DIR
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

    cluster_message = request.args.get("cluster_message")
    cluster_failed_message = request.args.get("cluster_failed_message")

    success_upload = request.args.get("success_upload")
    failed_upload = request.args.get("failed_upload")

    delete_success_message = request.args.get("delete_success_message")
    upload_message = request.args.get("upload_message")
    upload_no_file_message = request.args.get("upload_no_file_message")
    delete_failed_message = request.args.get("delete_failed_message")
 
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
        upload_message=upload_message,
        delete_success_message=delete_success_message,
        cluster_message=cluster_message,
        cluster_failed_message=cluster_failed_message,
        success_upload=success_upload, 
        failed_upload=failed_upload,
        upload_no_file_message=upload_no_file_message,
        delete_failed_message=delete_failed_message)

@app.route('/upload_file', methods=['POST'])
def upload_file():
        # Check if files were uploaded
    if len(request.files.getlist('file')) == 1:

        if request.files.getlist('file')[0].filename == '':
            return redirect(url_for("index", upload_no_file_message=f'No file has been selected for uploading!'))
    
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
        return redirect(url_for("index", delete_failed_message=f'No file has been selected for deletion!'))

    base_filename = os.path.splitext(filename)[0]

    logger.debug(f'Filename: {filename}, base name: {base_filename}')

    file_path = os.path.join(
        PATH_TO_VALID_FILES, 
        filename)
    
    if not os.path.exists(file_path):

        logger.debug(f'File {filename} does not exist in File Storage.')
        return redirect(url_for("index", delete_failed_message=f'File {filename} does not exist in File Storage.'))
    
    files_used_for_clustering = get_rows_cardinalities(
        path_to_cardinalities_file=PATH_TO_ROWS_CARDINALITIES
    )

    forbidden_files_to_delete = \
        list(files_used_for_clustering.get(USED_AS_BASE_KEY, {}).keys()) \
        + list(files_used_for_clustering.get(ONLY_CLASSIFIED_KEY, {}).keys())
    
    if filename in forbidden_files_to_delete:
        return redirect(url_for("index", delete_failed_message=f"""
            Deletion of file {filename} is forbidden - it has been used in last clusterization process, 
            cluster again with other files and then delete this file"""))
    
    logger.debug(f'{forbidden_files_to_delete=}')
    logger.debug(f'{filename=}')
    
    try:
        os.remove(file_path)
    except Exception as e:
        return redirect(url_for("index", delete_failed_message=f'Can not delete {filename} from validated files dir!'))

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
                    return redirect(url_for("index", delete_failed_message=f'Can not del {filename} from {data_dir} dir!'))
                else:
                    logger.info(f'Successfully deleted file from {data_dir}')

                    if data_dir == PATH_TO_FAISS_VECTORS_DIR:

                        deleted_successfully_from_json = del_file_from_embeded(
                            filename_to_del=filename,
                            path_to_embeddings_file=os.path.join(EMBEDDINGS_DIR, EMBEDDED_JSON)
                        )

                        if deleted_successfully_from_json:

                            logger.info(f'Successfully deleted {filename} from JSON file')

                        else:

                            logger.error(f'Failed to delete {filename} from JSON file')
                            return redirect(url_for("index", delete_failed_message=f'Failed to delete {filename} from JSON file.'))
    
    return redirect(url_for("index", delete_success_message=f'File {filename} deleted successfully.'))
    
@app.route('/choose_files_for_clusters', methods=['POST', 'GET'])
def choose_files_for_clusters():

    files_for_clustering = request.form.getlist('chosen_files')

    if not files_for_clustering:
        return redirect(url_for("index", cluster_failed_message=f'No file has been selected for clustering.'))
    
    if not all(False for file_ in files_for_clustering if file_ not in os.listdir(PATH_TO_VALID_FILES)):
        return redirect(url_for("index", cluster_failed_message=f'At least one file is not present in File Storage.'))
    
    logger.debug(f'Chosen files: {files_for_clustering}')

    zero_length_after_processing = process_data_from_choosen_files(
        chosen_files=files_for_clustering,
        path_to_valid_files=PATH_TO_VALID_FILES,
        path_to_cleared_files=PATH_TO_CLEARED_FILES,
        path_to_empty_content_dir=PATH_TO_EMPTY_CONTENTS,
        path_to_embeddings_dir=EMBEDDINGS_DIR,
        faiss_vectors_dirname=FAISS_VECTORS_DIR,
        embedded_files_filename=EMBEDDED_JSON,
        embeddings_model_name=EMBEDDINGS_MODEL,
        get_sentiment=GET_SENTIMENT,
        translate_content=TRANSLATE_CONTENT,
        lang_detection_model_name=LANG_DETECTION_MODEL,
        currently_serviced_langs=TRANSLATION_MODELS,
        sentiment_model_name=SENTIMENT_MODEL_NAME,
        swearwords=SWEAR_WORDS,
        original_content_column=ORIGINAL_CONTENT_COLUMN,
        content_column_name=PREPROCESSED_CONTENT_COLUMN,
        cleread_file_ext=CLEARED_FILE_EXT,
        empty_contents_suffix=EMPTY_CONTENTS_SUFFIX,
        empty_content_ext=EMPTY_CONTENTS_EXT,
        batch_size=BATCH_SIZE,
        seed=SEED)
    
    if zero_length_after_processing:

        files_for_clustering = [
            file_ for file_ in files_for_clustering 
            if file_ not in zero_length_after_processing
        ]

    if not files_for_clustering:
        return redirect(url_for("index", cluster_failed_message=f"After cleaning no files are suitable for clusterization."))

    new_current_df = get_clusters_for_choosen_files(
        chosen_files=files_for_clustering,
        path_to_cleared_files=PATH_TO_CLEARED_FILES,
        path_to_embeddings_dir=EMBEDDINGS_DIR,
        path_to_current_df_dir=PATH_TO_CURRENT_DF_DIR,
        rows_cardinalities_file=ROWS_CARDINALITIES_FILE,
        faiss_vectors_dirname=FAISS_VECTORS_DIR,
        embedded_files_filename=EMBEDDED_JSON,
        cleared_files_ext=CLEARED_FILE_EXT,
        outlier_treshold=OUTLIER_TRESHOLD,
        cluster_summary_column=CLUSTER_SUMMARY_COLUMN,
        filename_column=FILENAME_COLUMN,
        random_state=SEED,
        umap_model_name=DIM_REDUCER_MODEL_NAME,
        reducer_2d_model_name=REDUCER_2D_MODEL_NAME,
        n_neighbors=UMAP.get('n_neighbors'),
        min_dist=UMAP.get('min_dist'),
        n_components=UMAP.get('n_components'),
        coverage_with_best=HDBSCAN_SETTINGS.get('coverage_with_best'),
        min_cluster_size=HDBSCAN_SETTINGS.get('min_cluster_size'),
        min_samples=HDBSCAN_SETTINGS.get('min_samples'),
        metric=HDBSCAN_SETTINGS.get('metric'),                      
        cluster_selection_method=HDBSCAN_SETTINGS.get('cluster_selection_method')
    )

    if not isinstance(new_current_df, pd.DataFrame):
        return redirect(url_for("index", cluster_failed_message=new_current_df))
    
    new_current_df = new_current_df.astype({col: str for col in REQUIRED_COLUMNS})
    
    cns_after_clusterization(
        new_current_df=new_current_df,
        path_to_current_df_dir=PATH_TO_CURRENT_DF_DIR,
        path_to_cluster_exec_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR,
        topic_df_file_name=TOPICS_DF_FILE,
        current_df_filename=CURRENT_DF_FILE,
        topic_preffix_name=TOPIC_COLUMN_PREFIX,
        topics_concat_viz_col=TOPICS_CONCAT_FOR_VIZ,
        stop_words=STOP_WORDS,
        labels_column=LABELS_COLUMN,
        cardinalities_column=CARDINALITIES_COLUMN,
        cluster_exec_filename_prefix=CLUSTER_EXEC_FILENAME_PREFIX,
        cluster_exec_filename_ext=CLUSTER_EXEC_FILENAME_EXT,
        content_column_name=PREPROCESSED_CONTENT_COLUMN,
        no_topic_token=NO_TOPIC_TOKEN
    )

    logger.debug(f'Files {files_for_clustering} processed successfully.')

    n_clusters = len(new_current_df[LABELS_COLUMN].unique())

    return redirect(url_for("index", cluster_message=f"{n_clusters} clusters have been created successfully."))

@app.route('/get_empty_contents', methods=['GET'])
def get_empty_contents():

    full_filename = get_report_name_with_timestamp(
        filename_prefix=EMPTY_CONTENTS_SUFFIX.lstrip('_')
    )

    filepath = os.path.join(PATH_TO_EMPTY_ARCHIVE, full_filename)

    if len(set(list(os.listdir(PATH_TO_EMPTY_CONTENTS))).difference(set([GITKEEP_FILE]))) == 0:
        return redirect(url_for("show_clusters", message=f'Currently no empty content files are available!'))
    
    logger.debug(f'Empty content dir: {os.listdir(PATH_TO_EMPTY_CONTENTS)}')
    
    shutil.make_archive(
        filepath, 
        ZIP_EXT.lstrip('.'), 
        PATH_TO_EMPTY_CONTENTS)
    
    # Clear the folder
    for file_ in os.listdir(PATH_TO_EMPTY_CONTENTS):
        if file_.split('.')[0].endswith(EMPTY_CONTENTS_SUFFIX):
            os.remove(os.path.join(PATH_TO_EMPTY_CONTENTS, file_))
    
    # Return the ZIP file to the user
    return send_file(f"{filepath}{ZIP_EXT}", as_attachment=True, download_name=f"{full_filename}{ZIP_EXT}")

@app.route('/show_clusters', methods=['GET'])
def show_clusters():

    try:
        df = read_file(PATH_TO_CURRENT_DF)
    except FileNotFoundError:
        return redirect(url_for("index", no_current_df_message=f"Analysis page will be available after first clusterization process - current_df file is missing")) 

    message_info = request.args.get("message_info")
    message_successful = request.args.get("message_successful")
    message_unsuccessful = request.args.get("message_unsuccessful")

    if isinstance(df, pd.DataFrame):
        
        scatter_plot = px.scatter(
            data_frame=df,
            x='x',
            y='y',
            color=TOPICS_CONCAT_FOR_VIZ
        )

        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

        possible_values_for_filters = prepare_filters(
            df=df,
            date_column=DATE_COLUMN,
            date_filter_format=DATE_FILTER_FORMAT,
            filename_column=FILENAME_COLUMN,
            sentiment_column=SENTIMENT_COLUMN if GET_SENTIMENT else None,
            topic_colum_prefix=TOPIC_COLUMN_PREFIX,
            topics_range=TOPICS_RANGE
        )

        files_for_filtering = possible_values_for_filters.get('files_filter')
        dates_for_filtering = possible_values_for_filters.get('dates_filter')
        topics_for_filtering = possible_values_for_filters.get('topics_filter')
        sentiment_for_filtering = possible_values_for_filters.get('sentiment_filter')

        reports_to_choose = prepare_reports_to_chose(
            path_to_cluster_exec_reports_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR,
            path_to_valid_files=PATH_TO_VALID_FILES,
            files_for_filtering=files_for_filtering,
            gitkeep_file=GITKEEP_FILE,
            exec_report_ext=CLUSTER_EXEC_FILENAME_EXT,
        )

        exec_reports_to_show = reports_to_choose.get('exec_reports_to_show')
        available_for_update = reports_to_choose.get('available_for_update')
        
        fig_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template("cluster_viz_chartjs.html", 
                                figure=fig_json, 
                                files_for_filtering=files_for_filtering,
                                dates_for_filtering=dates_for_filtering,
                                topics_for_filtering=topics_for_filtering,
                                sentiment_for_filtering=sentiment_for_filtering,
                                available_for_update=available_for_update,
                                reports_to_show=exec_reports_to_show,
                                message_info=message_info,
                                message_successful=message_successful,
                                message_unsuccessful=message_unsuccessful,
                                sentiment_column=GET_SENTIMENT)
    
    return 'Nothing to show here'

@app.route('/apply_filter', methods=['POST'])
def apply_filter():

    logger.debug(f'Report columns: {ALL_DETAILED_REPORT_COLUMNS}')

    filtered_files = request.form.getlist('filtered_files')
    logger.debug(f'{filtered_files=}')

    filtered_date_from = request.form.get('filtered_date_from')
    logger.debug(f'{filtered_date_from=}')

    filtered_date_to = request.form.get('filtered_date_to')
    logger.debug(f"{filtered_date_to=}")

    filtered_topics = request.form.getlist('filtered_topics')
    logger.debug(f'{filtered_topics=}')

    filtered_sentiment = request.form.getlist('filtered_sentiment')
    logger.debug(f'{filtered_sentiment=}')

    filtered_df = read_file(
        PATH_TO_CURRENT_DF, 
        columns=ALL_DETAILED_REPORT_COLUMNS)
    
    TOPIC_COLUMNS = tuple(f"{TOPIC_COLUMN_PREFIX}_{i}" for i in TOPICS_RANGE)
    
    filters_dict = {
        FILENAME_COLUMN: filtered_files,
        DATE_COLUMN: [filtered_date_from, filtered_date_to],
        TOPIC_COLUMNS: filtered_topics,
        SENTIMENT_COLUMN: filtered_sentiment
    }

    filtered_df = apply_filters_on_df(
        df=filtered_df,
        filters_dict=filters_dict,
        date_column=DATE_COLUMN,
        date_format=DATE_FILTER_FORMAT
    )

    filtered_df.to_parquet(
        index=False, 
        path=PATH_TO_FILTERED_DF)

    return redirect(url_for("show_clusters", message_successful='Filters applied successfully'))

@app.route('/get_exec_filtered_report', methods=['POST'])
def get_exec_filtered_report():

    report_type = request.form.get('filtered_exec_report_type')

    if not report_type:
        return redirect(url_for("show_clusters", message_unsuccessful="No report type has been selected."))
    
    if report_type not in REPORT_FORMATS_MAPPING.keys():
        return redirect(url_for("show_clusters", message_unsuccessful=f"Selected report type {report_type} is not allowed."))

    ext_settings = REPORT_FORMATS_MAPPING.get(report_type, DEFAULT_REPORT_FORMAT_SETTINGS)
    report_ext = ext_settings.get('ext', '.csv')
    report_mimetype = ext_settings.get('mimetype', 'text/csv')

    try:

        filtered_df = read_file(
            file_path=PATH_TO_FILTERED_DF,
            columns=[LABELS_COLUMN]
        )

    except FileNotFoundError:
        return redirect(url_for("show_clusters", message_unsuccessful=f"DataFrame for filtering is not currently available."))

    filtered_exec_report_name = get_report_name_with_timestamp(
        filename_prefix=FILTERED_REPORT_PREFIX
    )

    try:
        summary_df, _, _ = save_cluster_exec_report(
            df=filtered_df,
            filename=filtered_exec_report_name,
            path_to_cluster_exec_reports_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR,
            clusters_topics=read_file(file_path=os.path.join(PATH_TO_CURRENT_DF_DIR, TOPICS_DF_FILE)),
            filename_ext=FILTERED_FILENAME_EXT,
            labels_column_name=LABELS_COLUMN,
            cardinalities_column_name=CARDINALITIES_COLUMN
        )

    except Exception as e:

        logger.error(f'Error while creating filtered exec report: {e}')
        return redirect(url_for('show_clusters',
                            message_unsuccessful='Can not save filtered report, file may be invalid'))

    try:

        response = create_response_report(
            df=summary_df,
            filename=filtered_exec_report_name,
            ext=report_ext,
            mimetype=report_mimetype,
            file_format=report_type
        )

    except Exception as e:

        logger.error(f'Error while creating response report: {e}')
        response = redirect(url_for('show_clusters',
                            message_unsuccessful='Creation of exec filtered report response failed, file may be invalid'))

    return response

@app.route('/get_detailed_filtered_report', methods=['POST'])
def get_detailed_filtered_report():

    report_type = request.form.get('detailed_filtered_report_type')

    if not report_type:
        return redirect(url_for("show_clusters", message_unsuccessful="No report type has been selected."))
    
    if report_type not in REPORT_FORMATS_MAPPING.keys():
        return redirect(url_for("show_clusters", message_unsuccessful=f"Selected report type {report_type} is not allowed."))

    ext_settings = REPORT_FORMATS_MAPPING.get(report_type, DEFAULT_REPORT_FORMAT_SETTINGS)
    report_ext = ext_settings.get('ext', '.csv')
    report_mimetype = ext_settings.get('mimetype', 'text/csv')

    filtered_df_filename = get_report_name_with_timestamp(
        filename_prefix=FILTERED_REPORT_PREFIX
    )

    try:
        filtered_df = read_file(
            file_path=PATH_TO_FILTERED_DF,
            columns=ALL_DETAILED_REPORT_COLUMNS
        )

    except Exception as e:
        logger.error(f'Error while reading filtered df: {e}')
        return redirect(url_for('show_clusters',
                            message_unsuccessful='Currently DataFrame for filtering is not available'))
        
    try:

        response = create_response_report(
            df=filtered_df,
            filename=filtered_df_filename,
            ext=report_ext,
            mimetype=report_mimetype,
            file_format=report_type
        )

    except Exception as e:

        logger.error(f'Error while creating response report: {e}')
        return redirect(url_for('show_clusters',
                            message_unsuccessful='Creation of detailed filtered report response failed, file may be invalid'))

    return response

@app.route('/get_last_cluster_exec_report', methods=['POST'])
def get_last_cluster_exec_report():

    report_type = request.form.get('last_report_type')

    if not report_type:
        return redirect(url_for("show_clusters", message_unsuccessful="No report type has been chosen."))
    
    if report_type not in REPORT_FORMATS_MAPPING.keys():
        return redirect(url_for("show_clusters", message_unsuccessful=f"Selected report type {report_type} is not allowed."))

    ext_settings = REPORT_FORMATS_MAPPING.get(report_type, DEFAULT_REPORT_FORMAT_SETTINGS)
    report_ext = ext_settings.get('ext', '.csv')
    report_mimetype = ext_settings.get('mimetype', 'text/csv')

    latest_exec_report = find_latested_n_exec_report(
        path_to_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR,
        cluster_exec_prefix=CLUSTER_EXEC_FILENAME_PREFIX,
        n_reports=1
    )

    if latest_exec_report is None:
        return redirect(url_for("show_clusters", message_unsuccessful="No clusterization exec reports are currently available."))

    try:

        cluster_exec_df = read_file(
            file_path=os.path.join(PATH_TO_CLUSTER_EXEC_REPORTS_DIR, latest_exec_report)
        )

    except Exception as e:

        logger.error(f'Can not read clusterizaton report {latest_exec_report} - {e}')
        return redirect(url_for("show_clusters", message_unsuccessful=f"""
            Could not read cluster exec report {latest_exec_report}. File may be invalid"""))

    try :

        response = create_response_report(
            df=cluster_exec_df,
            filename=latest_exec_report.split('.')[0],
            ext=report_ext,
            mimetype=report_mimetype,
            file_format=report_type
        )

    except Exception as e:

        logger.error(f'Creating response report failed. {e}')
        return redirect(url_for("show_clusters", message_unsuccessful=f"""
            Could not create cluster exec report {latest_exec_report}. File may be invalid"""))

    return response

@app.route('/get_chosen_cluster_exec_report', methods=['POST'])
def get_chosen_cluster_exec_report():

    chosen_report_name = request.form.get('chosen_report_name')
    report_type = request.form.get('chosen_report_type')

    if not report_type:
        return redirect(url_for("show_clusters", message_unsuccessful="No report type has been chosen."))
    
    if report_type not in REPORT_FORMATS_MAPPING.keys():
        return redirect(url_for("show_clusters", message_unsuccessful=f"Selected report type {report_type} is not allowed."))
    
    try:

        cluster_exec_df = read_file(
            file_path=os.path.join(PATH_TO_CLUSTER_EXEC_REPORTS_DIR, chosen_report_name)
        )

    except FileNotFoundError:
        return redirect(url_for("show_clusters", message_unsuccessful=f"Selected cluster exec report {chosen_report_name} does not exist in File Storage"))
    
    ext_settings = REPORT_FORMATS_MAPPING.get(report_type, DEFAULT_REPORT_FORMAT_SETTINGS)
    report_ext = ext_settings.get('ext', '.csv')
    report_mimetype = ext_settings.get('mimetype', 'text/csv')

    resp_report = create_response_report(
        df=cluster_exec_df,
        filename=chosen_report_name.split('.')[0],
        ext=report_ext,
        mimetype=report_mimetype,
        file_format=report_type
    )

    return resp_report


@app.route('/get_detailed_cluster_exec_report', methods=['POST'])
def get_detailed_cluster_exec_report():

    report_type = request.form.get('report_type_exec')

    if not report_type:
        return redirect(url_for("show_clusters", message_unsuccessful="No report type has been chosen."))
    
    if report_type not in REPORT_FORMATS_MAPPING.keys():
        return redirect(url_for("show_clusters", message_unsuccessful=f"Selected report type {report_type} is not allowed."))

    ext_settings = REPORT_FORMATS_MAPPING.get(report_type, DEFAULT_REPORT_FORMAT_SETTINGS)
    report_ext = ext_settings.get('ext', '.csv')
    report_mimetype = ext_settings.get('mimetype', 'text/csv')

    detailed_cluster_exec_report_filename = get_report_name_with_timestamp(
        filename_prefix=f"{DETAILED_CLUSTER_EXEC_FILENAME_PREFIX}_{CLUSTER_EXEC_FILENAME_PREFIX}"
    )

    current_df = read_file(
        file_path=PATH_TO_CURRENT_DF,
        columns=ALL_DETAILED_REPORT_COLUMNS
    )

    resp_report = create_response_report(
        df=current_df,
        filename=detailed_cluster_exec_report_filename,
        ext=report_ext,
        mimetype=report_mimetype,
        file_format=report_type
    )

    return resp_report

@app.route('/update_clusters_new_file', methods=['POST'])
def update_clusters_new_file():
    
    uploaded_file = request.files['file']
    filename = uploaded_file.filename

    if filename == '':
        return redirect(url_for("show_clusters", message_unsuccessful=f'No file has been selected for uploading!'))

    logger.debug(f"Uploaded file: {filename}")

    if os.path.exists(os.path.join(PATH_TO_VALID_FILES, filename)):
        return redirect(url_for("show_clusters", message_unsuccessful=f"File {filename} already exists in File Storage, update clusters with this file using 'Cluster with existing file' or choose another file"))
    
    else:

        success_upload, _ = upload_and_validate_files(
            uploaded_files=[uploaded_file]
        )

        logger.debug(f'File {filename} has been validated')

        if success_upload:

            zero_length_after_processing = process_data_from_choosen_files(
                chosen_files=[filename],
                path_to_valid_files=PATH_TO_VALID_FILES,
                path_to_cleared_files=PATH_TO_CLEARED_FILES,
                path_to_empty_content_dir=PATH_TO_EMPTY_CONTENTS,
                path_to_embeddings_dir=EMBEDDINGS_DIR,
                faiss_vectors_dirname=FAISS_VECTORS_DIR,
                embedded_files_filename=EMBEDDED_JSON,
                embeddings_model_name=EMBEDDINGS_MODEL,
                get_sentiment=GET_SENTIMENT,
                translate_content=TRANSLATE_CONTENT,
                lang_detection_model_name=LANG_DETECTION_MODEL,
                currently_serviced_langs=TRANSLATION_MODELS,
                sentiment_model_name=SENTIMENT_MODEL_NAME,
                swearwords=SWEAR_WORDS,
                original_content_column=ORIGINAL_CONTENT_COLUMN,
                content_column_name=PREPROCESSED_CONTENT_COLUMN,
                cleread_file_ext=CLEARED_FILE_EXT,
                empty_contents_suffix=EMPTY_CONTENTS_SUFFIX,
                empty_content_ext=EMPTY_CONTENTS_EXT,
                batch_size=BATCH_SIZE,
                seed=SEED)
            
            if zero_length_after_processing:
                return redirect(url_for("show_clusters", message_unsuccessful=f"After preprocessing the file {filename} no samples have left."))
            
            filename_in_cleared_files = find_filename_in_dir(
                path_to_dir=PATH_TO_CLEARED_FILES) \
                    .get(os.path.splitext(filename)[0])
            
            logger.debug(filename_in_cleared_files)

            n_of_rows_for_new_file = get_n_of_rows_df(
                os.path.join(PATH_TO_CLEARED_FILES, filename_in_cleared_files),
                loaded_column=ORIGINAL_CONTENT_COLUMN,
            )

            if n_of_rows_for_new_file < MIN_CLUSTER_SAMPLES:
                return redirect(url_for("show_clusters", message_unsuccessful=f"Number of samples for clusterization after preprocessing should be at least {MIN_CLUSTER_SAMPLES}."))

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
                    n_of_samples=n_of_rows_for_new_file,
                    topic_df_filename=TOPICS_DF_FILE,
                    outlier_treshold=OUTLIER_TRESHOLD,
                    clusterer_model_name=HDBSCAN_MODEL_NAME,
                    umap_model_name=DIM_REDUCER_MODEL_NAME,
                    reducer_2d_model_name=REDUCER_2D_MODEL_NAME,
                    cleared_files_ext=CLEARED_FILE_EXT
                )

                if not isinstance(new_current_df, pd.DataFrame):
                    return redirect(url_for("show_clusters", message_successful=new_current_df))

                rows_cardinalities_current_df[ONLY_CLASSIFIED_KEY][filename] = n_of_rows_for_new_file
                set_rows_cardinalities(
                    path_to_cardinalities_file=PATH_TO_ROWS_CARDINALITIES,
                    updated_cardinalities=rows_cardinalities_current_df
                )

                logger.info('Successfully updated rows cardinalities file for current_df')

                path_to_exec_report, destination_filename = cns_after_clusterization(
                    new_current_df=new_current_df,
                    path_to_current_df_dir=PATH_TO_CURRENT_DF_DIR,
                    path_to_cluster_exec_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR,
                    current_df_filename=CURRENT_DF_FILE,
                    stop_words=STOP_WORDS,
                    topic_preffix_name=TOPIC_COLUMN_PREFIX,
                    topics_concat_viz_col=TOPICS_CONCAT_FOR_VIZ,
                    labels_column=LABELS_COLUMN,
                    cardinalities_column=CARDINALITIES_COLUMN,
                    cluster_exec_filename_prefix=CLUSTER_EXEC_FILENAME_PREFIX,
                    cluster_exec_filename_ext=CLUSTER_EXEC_FILENAME_EXT,
                    topic_df_file_name=TOPICS_DF_FILE,
                    only_update=True
                )

                # response = make_response(send_file(
                #     path_to_exec_report, 
                #     mimetype="application/octet-stream", 
                #     as_attachment=True, 
                #     download_name=destination_filename))

                # Perform the redirect
                # redirect_url = 
                # response.headers['Location'] = redirect_url
                # response.status_code = 302

                return redirect(url_for("show_clusters", message_successful=f"Cluster labels for {filename} have been successfully assigned."))

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
                    cluster_summary_column=CLUSTER_SUMMARY_COLUMN,
                    outlier_treshold=OUTLIER_TRESHOLD,
                    filename_column=FILENAME_COLUMN,
                    random_state=SEED,
                    umap_model_name=DIM_REDUCER_MODEL_NAME,
                    reducer_2d_model_name=REDUCER_2D_MODEL_NAME,
                    n_neighbors=UMAP.get('n_neighbors'),
                    min_dist=UMAP.get('min_dist'),
                    n_components=UMAP.get('n_components'),
                    coverage_with_best=HDBSCAN_SETTINGS.get('coverage_with_best'),
                    min_cluster_size=HDBSCAN_SETTINGS.get('min_cluster_size'),
                    min_samples=HDBSCAN_SETTINGS.get('min_samples'),
                    metric=HDBSCAN_SETTINGS.get('metric'),                      
                    cluster_selection_method=HDBSCAN_SETTINGS.get('cluster_selection_method')
                )

                if not isinstance(new_current_df, pd.DataFrame):
                    return redirect(url_for("show_clusters", message_successful=new_current_df))
                
                new_current_df = new_current_df.astype({col: str for col in REQUIRED_COLUMNS})
                
                path_to_exec_report, destination_filename = cns_after_clusterization(
                    new_current_df=new_current_df,
                    path_to_current_df_dir=PATH_TO_CURRENT_DF_DIR,
                    path_to_cluster_exec_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR,
                    topic_df_file_name=TOPICS_DF_FILE,
                    topics_concat_viz_col=TOPICS_CONCAT_FOR_VIZ,
                    current_df_filename=CURRENT_DF_FILE,
                    stop_words=STOP_WORDS,
                    topic_preffix_name=TOPIC_COLUMN_PREFIX,
                    labels_column=LABELS_COLUMN,
                    cardinalities_column=CARDINALITIES_COLUMN,
                    cluster_exec_filename_prefix=CLUSTER_EXEC_FILENAME_PREFIX,
                    cluster_exec_filename_ext=CLUSTER_EXEC_FILENAME_EXT,
                    content_column_name=PREPROCESSED_CONTENT_COLUMN,
                    no_topic_token=NO_TOPIC_TOKEN
                )

                # response = make_response(send_file(
                #     path_to_exec_report,
                #     mimetype = "application/octet-stream",
                #     as_attachment = True,
                #     download_name = destination_filename
                # ))

                # return response

                n_clusters = len(new_current_df[LABELS_COLUMN].unique())

                return redirect(url_for("show_clusters", message_successful=f"Clusters successfully updated - {n_clusters} clusters has been created successfully."))
            
        else:
            logger.error(f'Can not preprocess file {filename}')
            return redirect(url_for("show_clusters", message_unsuccessful=f'Can not upload file {filename} for clusters update!'))

@app.route('/update_clusters_existing_file', methods=['POST'])
def update_clusters_existing_file():
    
    existing_file_for_update = request.form.get('ex_file_update')
    logger.debug(existing_file_for_update)

    if not existing_file_for_update:
        return redirect(url_for("show_clusters", message_unsuccessful=f"No file has been selected!"))

    if not os.path.exists(os.path.join(PATH_TO_VALID_FILES, existing_file_for_update)):
        return redirect(url_for("show_clusters", message_unsuccessful=f"Selected file {existing_file_for_update} does not exist in File Storage!"))

    zero_length_after_processing = process_data_from_choosen_files(
        chosen_files=[existing_file_for_update],
        path_to_valid_files=PATH_TO_VALID_FILES,
        path_to_cleared_files=PATH_TO_CLEARED_FILES,
        path_to_empty_content_dir=PATH_TO_EMPTY_CONTENTS,
        path_to_embeddings_dir=EMBEDDINGS_DIR,
        faiss_vectors_dirname=FAISS_VECTORS_DIR,
        embedded_files_filename=EMBEDDED_JSON,
        embeddings_model_name=EMBEDDINGS_MODEL,
        translate_content=TRANSLATE_CONTENT,
        get_sentiment=GET_SENTIMENT,
        lang_detection_model_name=LANG_DETECTION_MODEL,
        currently_serviced_langs=TRANSLATION_MODELS,
        sentiment_model_name=SENTIMENT_MODEL_NAME,
        swearwords=SWEAR_WORDS,
        original_content_column=ORIGINAL_CONTENT_COLUMN,
        content_column_name=PREPROCESSED_CONTENT_COLUMN,
        cleread_file_ext=CLEARED_FILE_EXT,
        empty_contents_suffix=EMPTY_CONTENTS_SUFFIX,
        empty_content_ext=EMPTY_CONTENTS_EXT,
        batch_size=BATCH_SIZE,
        seed=SEED)
    
    if zero_length_after_processing:
        return redirect(url_for("show_clusters", message_unsuccessful=f"After preprocessing the file {existing_file_for_update} no samples have left."))
    
    logger.debug(existing_file_for_update)

    filename_in_cleared_files = find_filename_in_dir(
        path_to_dir=PATH_TO_CLEARED_FILES) \
            .get(os.path.splitext(existing_file_for_update)[0])
    
    logger.debug(filename_in_cleared_files)

    n_of_rows_existing_file = get_n_of_rows_df(
        os.path.join(PATH_TO_CLEARED_FILES, filename_in_cleared_files),
        loaded_column=ORIGINAL_CONTENT_COLUMN,
    )

    if n_of_rows_existing_file < MIN_CLUSTER_SAMPLES:
        return redirect(url_for("show_clusters", message_unsuccessful=f"Number of samples for clusterization after preprocessing should be at least {MIN_CLUSTER_SAMPLES}."))

    rows_cardinalities_current_df = get_rows_cardinalities(
        path_to_cardinalities_file=PATH_TO_ROWS_CARDINALITIES
    )

    need_to_recalculate = cluster_recalculation_needed(
        n_of_rows=n_of_rows_existing_file,
        rows_cardinalities_current_df=rows_cardinalities_current_df,
        recalculate_treshold=RECALCULATE_CLUSTERS_TRESHOLD
    )

    if not need_to_recalculate:

        logger.info(f'Treshold {RECALCULATE_CLUSTERS_TRESHOLD} has not been exceeded, assigning topics based on current_df clusterization')

        new_current_df = get_cluster_labels_for_new_file(
            filename=existing_file_for_update,
            path_to_current_df=PATH_TO_CURRENT_DF,
            path_to_current_df_dir=PATH_TO_CURRENT_DF_DIR,
            path_to_cleared_files_dir=PATH_TO_CLEARED_FILES,
            path_to_faiss_vetors_dir=PATH_TO_FAISS_VECTORS_DIR,
            topic_df_filename=TOPICS_DF_FILE,
            required_columns=REQUIRED_COLUMNS,
            outlier_treshold=OUTLIER_TRESHOLD,
            clusterer_model_name=HDBSCAN_MODEL_NAME,
            umap_model_name=DIM_REDUCER_MODEL_NAME,
            reducer_2d_model_name=REDUCER_2D_MODEL_NAME,
            cleared_files_ext=CLEARED_FILE_EXT
        )

        if not isinstance(new_current_df, pd.DataFrame):
            return redirect(url_for("show_clusters", message_info=new_current_df))

        rows_cardinalities_current_df[ONLY_CLASSIFIED_KEY][existing_file_for_update] = n_of_rows_existing_file
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
            topic_preffix_name=TOPIC_COLUMN_PREFIX,
            topics_concat_viz_col=TOPICS_CONCAT_FOR_VIZ,
            stop_words=STOP_WORDS,
            labels_column=LABELS_COLUMN,
            cardinalities_column=CARDINALITIES_COLUMN,
            cluster_exec_filename_prefix=CLUSTER_EXEC_FILENAME_PREFIX,
            cluster_exec_filename_ext=CLUSTER_EXEC_FILENAME_EXT,
            topic_df_file_name=TOPICS_DF_FILE,
            only_update=True
        )

        return redirect(url_for("show_clusters", message_successful=f"Cluster labels for {existing_file_for_update} have been successfully assigned."))

    else:

        logger.info(f'Treshold {RECALCULATE_CLUSTERS_TRESHOLD} has been exceeded, recalculating clusters for current_df')

        all_current_df_files = [file_ for in_dict in rows_cardinalities_current_df.values() for file_ in in_dict.keys()]

        new_current_df = get_clusters_for_choosen_files(
            chosen_files= all_current_df_files + [existing_file_for_update],
            path_to_cleared_files=PATH_TO_CLEARED_FILES,
            path_to_embeddings_dir=EMBEDDINGS_DIR,
            path_to_current_df_dir=PATH_TO_CURRENT_DF_DIR,
            rows_cardinalities_file=ROWS_CARDINALITIES_FILE,
            faiss_vectors_dirname=FAISS_VECTORS_DIR,
            embedded_files_filename=EMBEDDED_JSON,
            cleared_files_ext=CLEARED_FILE_EXT,
            outlier_treshold=OUTLIER_TRESHOLD,
            cluster_summary_column=CLUSTER_SUMMARY_COLUMN,
            filename_column=FILENAME_COLUMN,
            random_state=SEED,
            umap_model_name=DIM_REDUCER_MODEL_NAME,
            reducer_2d_model_name=REDUCER_2D_MODEL_NAME,
            n_neighbors=UMAP.get('n_neighbors'),
            min_dist=UMAP.get('min_dist'),
            n_components=UMAP.get('n_components'),
            min_cluster_size=HDBSCAN_SETTINGS.get('min_cluster_size'),
            min_samples=HDBSCAN_SETTINGS.get('min_samples'),
            metric=HDBSCAN_SETTINGS.get('metric'),                      
            cluster_selection_method=HDBSCAN_SETTINGS.get('cluster_selection_method')
        )

        if not isinstance(new_current_df, pd.DataFrame):
            return redirect(url_for("show_clusters", message_info=new_current_df))

        new_current_df = new_current_df.astype({col: str for col in REQUIRED_COLUMNS})
        
        cns_after_clusterization(
            new_current_df=new_current_df,
            path_to_current_df_dir=PATH_TO_CURRENT_DF_DIR,
            path_to_cluster_exec_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR,
            topic_df_file_name=TOPICS_DF_FILE,
            current_df_filename=CURRENT_DF_FILE,
            topic_preffix_name=TOPIC_COLUMN_PREFIX,
            topics_concat_viz_col=TOPICS_CONCAT_FOR_VIZ,
            stop_words=STOP_WORDS,
            labels_column=LABELS_COLUMN,
            cardinalities_column=CARDINALITIES_COLUMN,
            cluster_exec_filename_prefix=CLUSTER_EXEC_FILENAME_PREFIX,
            cluster_exec_filename_ext=CLUSTER_EXEC_FILENAME_EXT,
            content_column_name=PREPROCESSED_CONTENT_COLUMN,
            no_topic_token=NO_TOPIC_TOKEN
        )

        n_clusters = len(new_current_df[LABELS_COLUMN].unique())

        return redirect(url_for("show_clusters", message_successful=f"Clusters successfully updated - {n_clusters} clusters have been created successfully."))
    
@app.route('/compare_selected_reports', methods=['POST'])
def compare_selected_reports():

    filename1 = request.form.get('raport-1')
    filename2 = request.form.get('raport-2')
    report_format_form = request.form.get('file-format')

    if any(not file_ for file_ in [filename1, filename2]):
        return redirect(url_for("show_clusters",
                                message_unsuccessful=f"Chosing both files is required"))

    if filename1 == filename2:
        return redirect(url_for("show_clusters",
                                message_unsuccessful=f"Chosen files must be different"))
    
    if not report_format_form:
        return redirect(url_for("show_clusters", message_unsuccessful="No report type has been chosen."))
    
    if report_format_form not in REPORT_FORMATS_MAPPING.keys():
        return redirect(url_for("show_clusters", message_unsuccessful=f"Selected report type {report_format_form} is not allowed."))

    logger.debug(f'{report_format_form=}')

    ext_settings = REPORT_FORMATS_MAPPING.get(report_format_form,
                                              DEFAULT_REPORT_FORMAT_SETTINGS)
    report_ext = ext_settings.get('ext', '.csv')
    report_mimetype = ext_settings.get('mimetype', 'text/csv')

    logger.debug(filename1, filename2)

    try:

        comparison_result_df = compare_reports(
            first_report_name=filename1,
            second_report_name=filename2,
            path_to_reports_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR
        )

    except Exception as e:

        logger.error(f'Comparing reports failed, could not compare report {filename1} and {filename2} - {e}')
        return redirect(url_for("show_clusters",
                                message_unsuccessful=f"""Error while trying to compare reports, files may be invalid."""))

    logger.debug(comparison_result_df)

    comparison_report_filename = f"{filename1.split('.')[0]}__{filename2.split('.')[0]}{COMPARING_REPORT_SUFFIX}"
    
    path_to_new_report = os.path.join(PATH_TO_COMPARING_REPORTS_DIR,
                                      f"{comparison_report_filename}{report_ext}")

    logger.debug(f'{report_ext=}')

    if report_ext in ['.csv', '.xlsx', '.html']:

        save_df_to_file(
            df=comparison_result_df,
            filename=comparison_report_filename,
            path_to_dir=PATH_TO_COMPARING_REPORTS_DIR,
            file_ext=report_ext
        )
       
    elif report_ext == '.pdf':

        filenames = [filename1.split('.')[0], filename2.split('.')[0]]
    
        try:
            create_pdf_comaprison_report(
                df=comparison_result_df,
                old_col_name=OLD_COL_NAME,
                new_col_name=NEW_COL_NAME,
                cols_for_label=COLS_FOR_LABEL,
                cols_for_old_label=COLS_FOR_OLD_LABEL,
                output_file_path=path_to_new_report,
                filenames=filenames
            )

        except Exception as e:

            logger.error(f'Creating pdf report failed for {filenames} - {e}')
            return redirect(url_for("show_clusters",
                                    message_unsuccessful=f"Could not create comparing pdf report"))

        logger.debug(f'Report ext is .pdf')

    else:

        logger.error(f"Report extension {report_ext} is not supported")
        return redirect(url_for("show_clusters", message_unsuccessful=f"""
            Could not create comparing report beacuse report extension {report_ext} is not supported"""))
 
    response = make_response(send_file(
        path_to_new_report,
        mimetype = report_mimetype,
        as_attachment = True,
        download_name = f"{comparison_report_filename}{report_ext}"
    ))

    return response

@app.route('/compare_with_last_report', methods=['POST'])
def compare_with_last_report():

    try:

        filename1, filename2 = find_latested_n_exec_report(
            path_to_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR,
            cluster_exec_prefix=CLUSTER_EXEC_FILENAME_PREFIX,
            n_reports=2)
        
    except (ValueError, TypeError):

        logger.error(f"Could not find latest reports to compare")
        return redirect(url_for("show_clusters", message_unsuccessful=f"Currently there are not enough cluster execution reports to compare"))
    
    report_format_form = request.form.get('file-format')

    if not report_format_form:
        return redirect(url_for("show_clusters", message_unsuccessful="No report type has been chosen."))
    
    if report_format_form not in REPORT_FORMATS_MAPPING.keys():
        return redirect(url_for("show_clusters", message_unsuccessful=f"Selected report type {report_format_form} is not allowed."))

    ext_settings = REPORT_FORMATS_MAPPING.get(report_format_form, DEFAULT_REPORT_FORMAT_SETTINGS)
    report_ext = ext_settings.get('ext', '.csv')
    report_mimetype = ext_settings.get('mimetype', 'text/csv')

    try:

        comparison_result_df = compare_reports(
            first_report_name=filename1,
            second_report_name=filename2,
            path_to_reports_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR
        )

    except Exception as e:

        logger.error(f'Comparing reports failed, could not compare report {filename1} and {filename2} - {e}')
        return redirect(url_for("show_clusters",
                                message_unsuccessful=f"""Error while trying to compare reports, files may be invalid."""))
    logger.debug(comparison_result_df)

    comparison_report_filename = f"{filename1.split('.')[0]}__{filename2.split('.')[0]}{COMPARING_REPORT_SUFFIX}"
    
    path_to_new_report = os.path.join(
        PATH_TO_COMPARING_REPORTS_DIR,
        f"{comparison_report_filename}{report_ext}"
    )

    if report_ext in ['.csv', '.xlsx', '.html']:

        save_df_to_file(
            df=comparison_result_df,
            filename=comparison_report_filename,
            path_to_dir=PATH_TO_COMPARING_REPORTS_DIR,
            file_ext=report_ext
        )

    elif report_ext == '.pdf':

        filenames = [filename1.split('.')[0], filename2.split('.')[0]]

        try:

            create_pdf_comaprison_report(
                df=comparison_result_df,
                old_col_name=OLD_COL_NAME,
                new_col_name=NEW_COL_NAME,
                cols_for_label=COLS_FOR_LABEL,
                cols_for_old_label=COLS_FOR_OLD_LABEL,
                output_file_path=path_to_new_report,
                filenames=filenames
            )

        except Exception as e:

            logger.error(f'Creating pdf report failed for {filenames} - {e}')
            return redirect(url_for("show_clusters",
                                    message_unsuccessful=f"Could not create comparing pdf report"))

    else:

        logger.error(f"Report extension {report_ext} is not supported")
        return redirect(url_for("show_clusters", message_unsuccessful=f"""
            Could not create comparing report beacuse report extension {report_ext} is not supported"""))
 

    response = make_response(send_file(
        path_to_new_report,
        mimetype = report_mimetype,
        as_attachment = True,
        download_name = f"{comparison_report_filename}{report_ext}"
    ))

    return response

if __name__ == '__main__':
    app.run(debug=False)





