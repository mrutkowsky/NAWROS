import os
import shutil
import pandas as pd
from flask import Flask, request, render_template, send_file, redirect, url_for, make_response
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
    save_df_to_file
import plotly.express as px
import json
import plotly
from utils.cluster import get_clusters_for_choosen_files, \
    get_cluster_labels_for_new_file, \
    cluster_recalculation_needed, \
    cns_after_clusterization
from utils.reports import compare_reports, find_latest_two_reports
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
RAPORTS_DIR = DIRECTORIES.get('raports')
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

LANG_DETECTION_MODEL = ML.get('lang_detection_model')

TRANSLATION_MODELS = ML.get('translation_models')
PL_TO_ENG_TRANS = TRANSLATION_MODELS.get('PL_TO_ENG')

REPORT_CONFIG = CONFIGURATION.get('REPORT_SETTINGS')

BASE_REPORT_COLUMNS = REPORT_CONFIG.get('base_columns')
LABELS_COLUMN = REPORT_CONFIG.get('labels_column', 'labels')
CARDINALITIES_COLUMN = REPORT_CONFIG.get('cardinalities_column', 'counts')
SENTIMENT_COLUMN = REPORT_CONFIG.get('sentiment_column', 'sentiment')
FILENAME_COLUMN = REPORT_CONFIG.get('filename_column' 'filename')
COMPARING_RAPORT_DOWNLOAD_NAME = REPORT_CONFIG.get('download_name')
NO_TOPIC_TOKEN = REPORT_CONFIG.get('no_topic_token')
COMPARING_REPORT_SUFFIX = REPORT_CONFIG.get('comparing_report_suffix')

ALL_REPORT_COLUMNS = BASE_REPORT_COLUMNS + [
    LABELS_COLUMN, 
    CARDINALITIES_COLUMN, 
    SENTIMENT_COLUMN, 
    FILENAME_COLUMN
]

CLUSTER_EXEC_FILENAME_PREFIX = REPORT_CONFIG.get('cluster_exec_filename_prefix')
CLUSTER_EXEC_FILENAME_EXT = REPORT_CONFIG.get('cluster_exec_filename_ext')

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

    """
    Uploads and validates a list of files.

    Args:
        uploaded_files (list): A list of uploaded files.

    Returns:
        tuple: A tuple containing two lists. The first list contains the filenames
        of files that were uploaded successfully, and the second list contains
        the filenames and corresponding error messages of files that failed to upload.
    """


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
    """
    Renders the index.html template with the necessary variables.

    Retrieves the message, success_upload, and failed_upload parameters from the request query string
    and converts them from JSON format if available. Retrieves the list of validated files and filters
    them based on their file extensions. Renders the index.html template with the appropriate variables.

    Returns:
        str: The rendered HTML content.
    """

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
    """
    Handles file uploads, performs validation, and redirects to the index page.

    Returns a redirect response to the index page with a success message and the uploaded file details
    if the file upload and validation are successful. If no file is selected for uploading, redirects to
    the index page with an appropriate error message.

    Returns:
        str: A redirect response to the index page.
    """

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
    """
    Handles file deletion and performs additional cleanup tasks.

    Returns a redirect response to the index page with a success message if the file is deleted
    successfully. If no file is selected for deletion or the file does not exist, redirects to
    the index page with an appropriate error message.

    Returns:
        str: A redirect response to the index page.
    """

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

                        if data_dir == PATH_TO_FAISS_VECTORS_DIR:
 
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
    
@app.route('/choose_files_for_clusters', methods=['POST', 'GET'])
def choose_files_for_clusters():
    """
        Processes selected files for clustering, performs data processing and clustering tasks,
        and updates the current dataframe and cluster execution reports.

        Returns a redirect response to the index page with a success message indicating the number
        of clusters created.

        If no files are selected for clustering, redirects to the index page with an error message.

        Returns:
            str: A redirect response to the index page.
        """

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
        lang_detection_model_name=LANG_DETECTION_MODEL,
        translation_model_name=PL_TO_ENG_TRANS,
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
        cluster_exec_filename_ext=CLUSTER_EXEC_FILENAME_EXT,
        content_column_name=CONTENT_COLUMN,
        no_topic_token=NO_TOPIC_TOKEN
    )

    logger.debug(f'Files {files_for_clustering} processed successfully.')

    n_clusters = len(new_current_df[LABELS_COLUMN].unique())

    return redirect(url_for("index", message=f"{n_clusters} clusters has been created successfully."))
    
@app.route('/show_clusters_submit', methods=['POST'])
def show_clusters_submit():
    """
        Handles the submission of the "Show Clusters" form.

        If the form contains the "show_plot" field, redirects to the 'show_clusters' route
        with the value of "show_plot" as a query parameter.

        If the "show_plot" field is not present, redirects to the index page with an error message.

        Returns:
            str: A redirect response to the appropriate page.
        """

    show_plot = request.form.get('show_plot')

    if show_plot:
        return redirect(url_for('show_clusters', show_plot=show_plot))
    else:
        return redirect(url_for("index", message=f"Cannot show the clusters!"))

@app.route('/show_clusters', methods=['GET'])
def show_clusters():
    """
        Renders the cluster visualization page with the necessary variables.

        Retrieves the message parameter from the request query string. Reads the current dataframe from the
        specified path. If the dataframe is valid, creates a scatter plot using Plotly Express and converts it
        to JSON format. Retrieves the list of validated files and filters them based on their file extensions.
        Retrieves the list of cluster execution reports and filters out the '.gitkeep' file. Renders the
        cluster visualization template with the appropriate variables.

        Returns:
            str: The rendered HTML content.
    """

    message = request.args.get("message")

    df = read_file(PATH_TO_CURRENT_DF)

    if isinstance(df, pd.DataFrame):
        
        scatter_plot = px.scatter(
            data_frame=df,
            x='x',
            y='y',
            color=LABELS_COLUMN
        )

        validated_files = os.listdir(
        PATH_TO_VALID_FILES)
        
        validated_files_to_show = [
            v_file for v_file in validated_files 
            if os.path.splitext(v_file)[-1].lower() in ALLOWED_EXTENSIONS
        ]

        raports = os.listdir(
        PATH_TO_CLUSTER_EXEC_REPORTS_DIR)
        
        raports_to_show = [
            raport for raport in raports 
            if raport != '.gitkeep'
        ]
        print(raports_to_show)
        
        fig_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template("cluster_viz_chartjs.html", 
                                figure=fig_json, 
                                columns=ALL_REPORT_COLUMNS, 
                                files=validated_files_to_show,
                                message=message,
                                raports=raports_to_show)
        """
        x_values = df['x'].tolist()
        y_values = df['y'].tolist()

        data = []
        for i in range(len(x_values)):
            point = {'x': x_values[i], 'y': y_values[i]}
            data.append(point)

        print(data)
        return render_template("cluster_viz_chartjs.html", json_data=jsonify(data))
        """

    
    return 'Nothing to show here'

@app.route('/show_filters_submit', methods=['POST'])
def show_filter_submit():
    """
        Handles the submission of the filter form and redirects to the appropriate page.

        Retrieves the value of the 'show_filter' field from the form data. If the value is present,
        redirects to the 'show_filter' route with the 'show_filter' parameter. Otherwise, redirects
        to the index page with an appropriate message.

        Returns:
            str: A redirect response to the designated page.
    """

    show_filter = request.form.get('show_filter')
    if show_filter:
        return redirect(url_for('show_filter', show_filter=show_filter))
    else:
        return redirect(url_for("index", message=f"Cannot show the filtering!"))

@app.route('/show_filter', methods=['GET'])
def show_filter():
    """
        Handles the GET request for showing the filtering page.

        Retrieves the list of report columns from the `ALL_REPORT_COLUMNS` variable. If the variable is
        a list, renders the `filtering.html` template with the `columns` variable set to the list of report columns.
        Otherwise, returns a string indicating that there is nothing to show.

        Returns:
            str: The rendered HTML content or a string indicating that there is nothing to show.
    """

    if request.method == 'GET':
        if isinstance(ALL_REPORT_COLUMNS, list):

            return render_template('filtering.html', columns=ALL_REPORT_COLUMNS)
        
        return 'Nothing to show here'
    
@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    """
        Handles the POST request for applying filters to the DataFrame.

        Retrieves the filter values from the JSON payload of the request. Reads the current DataFrame from the file
        specified by `PATH_TO_CURRENT_DF` and converts it to a DataFrame object. Converts the columns of the DataFrame
        to string type. Constructs a query string based on the filter values. Applies the query string as a filter to
        the DataFrame. Writes the filtered DataFrame to a CSV file specified by `PATH_TO_FILTERED_DF`. Sets a success
        message indicating that the filters have been applied successfully. Renders the `filtering.html` template with
        the updated columns and the success message.

        Returns:
            str: The rendered HTML content.
    """

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
    """
        Handles the POST request for downloading the filtered data report.

        Retrieves the report type from the JSON payload of the request. Reads the filtered DataFrame from the file
        specified by `PATH_TO_FILTERED_DF`. Retrieves the file type for the report. Prepares the CSV file for download
        by writing the DataFrame to the appropriate file format. Creates a response object with the file as the content
        to be downloaded. Sets the appropriate MIME type for the response. Sets the response as an attachment with
        the specified download name. Returns the response.

        Returns:
            Response: The response object for downloading the file.
    """

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
    """
        Handles the POST request for updating clusters with a new file.

        Retrieves the uploaded file from the request. Processes the data from the chosen file.
        Retrieves the rows cardinalities for the current DataFrame. Checks if recalculation of clusters is needed.
        If not, retrieves the cluster labels for the new file based on the current DataFrame. Updates the rows cardinalities file.
        If recalculation is needed, retrieves the clusters for the chosen files. Updates the cluster execution reports and topic DataFrame.
        Redirects to the `show_clusters` route with a success message if recalculation is needed and completed successfully.
        If there was an error during preprocessing or updating the clusters, redirects to the `show_clusters` route with an appropriate error message.

        Returns:
            Response: The response object for redirecting to the `show_clusters` route.
    """

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
            lang_detection_model_name=LANG_DETECTION_MODEL,
            translation_model_name=PL_TO_ENG_TRANS,
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
                cluster_exec_filename_ext=CLUSTER_EXEC_FILENAME_EXT,
                content_column_name=CONTENT_COLUMN,
                no_topic_token=NO_TOPIC_TOKEN
            )

            n_clusters = len(new_current_df[LABELS_COLUMN].unique())

            return redirect(url_for("show_clusters", message=f"{n_clusters} clusters has been created successfully."))
        
    else:
        logger.error(f'Can not preprocess file {filename}')
        return redirect(url_for("show_clusters", message=f'Can not upload file {filename} for clusters update!'))

    return redirect(url_for("show_clusters", message=f"Cluster labels for {filename} have been successfully assigned."))
    
@app.route('/compare_selected_reports', methods=['POST'])
def compare_selected_reports():
    """
        Handles the POST request for comparing selected reports.

        Retrieves the selected filenames and report format from the form data.
        Calls the `compare_reports` function to compare the selected reports.
        Saves the comparison result DataFrame to a file.
        Constructs a response object to download the comparison report.

        Returns:
            Response: The response object for downloading the comparison report.
    """

    filename1 = request.form.get('raport-1')
    filename2 = request.form.get('raport-2')
    report_format_form = request.form.get('file-format')

    ext_settings = REPORT_FORMATS_MAPPING.get(report_format_form, "csv")
    report_ext = ext_settings.get('ext', '.csv')
    report_mimetype = ext_settings.get('mimetype', 'text/csv')

    logger.debug(filename1, filename2)

    comparison_result_df = compare_reports(
        first_report_name=filename1,
        second_report_name=filename2,
        path_to_reports_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR
    )

    logger.debug(comparison_result_df)

    comparison_report_filename = f"{filename1.split('.')[0]}__{filename2.split('.')[0]}{COMPARING_REPORT_SUFFIX}"

    save_df_to_file(
        df=comparison_result_df,
        filename=comparison_report_filename,
        path_to_dir=PATH_TO_COMPARING_REPORTS_DIR,
        file_ext=report_ext
    )

    path_to_new_report = os.path.join(PATH_TO_COMPARING_REPORTS_DIR, f"{comparison_report_filename}{report_ext}")

    response = make_response(send_file(
        path_to_new_report,
        mimetype = report_mimetype,
        as_attachment = True,
        download_name = f"{comparison_report_filename}{report_ext}"
    ))

    return response

@app.route('/compare_with_last_report', methods=['POST'])
def compare_with_last_report():
    """
        Handles the POST request for comparing with the last report.

        Retrieves the filenames of the two latest reports from the specified directory.
        Retrieves the selected report format from the form data.
        Calls the `compare_reports` function to compare the selected reports.
        Saves the comparison result DataFrame to a file.
        Constructs a response object to download the comparison report.

        Returns:
            Response: The response object for downloading the comparison report.
    """

    filename1, filename2 = find_latest_two_reports(PATH_TO_CLUSTER_EXEC_REPORTS_DIR)
    report_format_form = request.form.get('file-format')

    ext_settings = REPORT_FORMATS_MAPPING.get(report_format_form, "csv")
    report_ext = ext_settings.get('ext', '.csv')
    report_mimetype = ext_settings.get('mimetype', 'text/csv')

    comparison_result_df = compare_reports(
        first_report_name=filename1,
        second_report_name=filename2,
        path_to_reports_dir=PATH_TO_CLUSTER_EXEC_REPORTS_DIR
    )

    logger.debug(comparison_result_df)

    comparison_report_filename = f"{filename1.split('.')[0]}__{filename2.split('.')[0]}{COMPARING_REPORT_SUFFIX}"

    save_df_to_file(
        df=comparison_result_df,
        filename=comparison_report_filename,
        path_to_dir=PATH_TO_COMPARING_REPORTS_DIR,
        file_ext=report_ext
    )

    path_to_new_report = os.path.join(PATH_TO_COMPARING_REPORTS_DIR, f"{comparison_report_filename}{report_ext}")

    response = make_response(send_file(
        path_to_new_report,
        mimetype = report_mimetype,
        as_attachment = True,
        download_name = f"{comparison_report_filename}{report_ext}"
    ))

    return response

if __name__ == '__main__':
    app.run(debug=True)





