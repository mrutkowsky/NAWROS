from flask import Flask, request
import logging


from utils.read_data import read_data

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/load_data', methods=['POST'])
def load_data():
    data_path = request.form.get('data_path')
    logger.info(f"Loading data from {data_path}")
    data = read_data(data_path)

    return 'Data loaded successfully'


if __name__ == '__main__':
    app.run(debug=True)
