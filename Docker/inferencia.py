import joblib
import pandas as pd 

import logging
from sys import stdout

logger = logging.getLogger(__name__)

pipeline= joblib.load('pipeline_model.pkl')

logger.info('Loading pipeline_model')

imput = pd.read_csv('/files/imput.csv')

logger.info('Reading imput.csv')

output = pipeline_model.predict(imput)

logger.info('make Predictions')

pd.DataFrame(output, columns=['Llueve_predicha']).to_csv('/files/ourput.csv', index = False)

logger.info('saved output')