import os
import shutil
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
import logging
from utils.module_functions import \
    validate_file, \
    validate_file_extension

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
TEMP_FOLDER = 'tmp'
VALIDATED_FILES_FOLDER = 'validated_files'
ALLOWED_EXTENSIONS = {'.csv', '.txt', '.xlsx'}
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOGGER_LEVEL = 'DEBUG'
REQUIRED_COLUMNS = [
    "Questioned_Date", 
    "Model_No", 
    "OS", 
    "SW_Version", 
    "CSC", 
    "Category", 
    "Application_Name", 
    "content"
]

logging.basicConfig(
    level=LOGGER_LEVEL,
    format=LOGGING_FORMAT)

logger = logging.getLogger(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["TEMP_FOLDER"] = TEMP_FOLDER
app.config["VALIDATED_FILES_FOLDER"] = VALIDATED_FILES_FOLDER

@app.route("/")
def index():

    message = request.args.get("message")

    validated_files = os.listdir(
        app.config["VALIDATED_FILES_FOLDER"])

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
                app.config['TEMP_FOLDER'], 
                uploaded_file.filename)
            uploaded_file.save(file_path)

            validated_file_content = validate_file(
                file_path,
                required_columns=REQUIRED_COLUMNS)

            if isinstance(validated_file_content, bool):
                
                shutil.move(
                    os.path.join(
                    app.config["TEMP_FOLDER"], 
                    uploaded_file.filename),
                    os.path.join(
                    app.config["VALIDATED_FILES_FOLDER"], 
                    uploaded_file.filename),
                )
                 
                files_uploading_status['uploaded_succesfully'].append(uploaded_file.filename)
                 
            else:
                os.remove(os.path.join(
                    app.config["TEMP_FOLDER"], 
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
            app.config['VALIDATED_FILES_FOLDER'], 
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

    return redirect(url_for("index"))


if __name__ == '__main__':
    app.run(debug=True)





