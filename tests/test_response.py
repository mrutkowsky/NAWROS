import requests
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


url = "http://127.0.0.1:5000/load_data"

if __name__ == '__main__':
    data_path = 'data/samsung/VOC_for_NLP.xlsx'
    r = requests.post(url, data={'data_path': data_path})
    print(r.text)