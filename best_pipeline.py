import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file --> Si y_train_clean ya es tu columna de salida y está correctamente definida, no necesitas renombrarla. 
#Solo asegúrate de que esa variable contenga las etiquetas correctas que quieres predecir.
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.8658319485653185
exported_pipeline = GradientBoostingClassifier(learning_rate=0.1, max_depth=6, max_features=0.2, min_samples_leaf=2, min_samples_split=12, n_estimators=100, subsample=0.6500000000000001)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
