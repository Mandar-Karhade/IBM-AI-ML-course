import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=55)

# Average CV score on the training set was: 0.48022040192841586
exported_pipeline = GradientBoostingRegressor(alpha=0.8, learning_rate=0.1, loss="quantile", max_depth=6, max_features=1.0, min_samples_leaf=12, min_samples_split=15, n_estimators=100, subsample=0.7500000000000001)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 55)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
