import xgboost as xgb
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
import numpy as np
import joblib
import pickle

reg = xgb.XGBRegressor(
    tree_method="hist",
    eval_metric='auc',
)

data = pd.read_excel('meta-features dataset.xlsx', header=None)
myData = data
# print(myData)
x_index = np.linspace(0, 60, 60, endpoint=False)
# print(x_index)
y_index = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
# print(y_index)
x = myData.reindex(columns=x_index)

y = myData.reindex(columns=y_index)

x = np.array(x)
y = np.array(y)

X_train = x
y_train = y

reg.fit(X_train, y_train)

joblib.dump(reg, "RPS-AMS_RankingsPredictor.m")
#
# with open('train_model.pkl', 'wb') as f:
#     pickle.dump(reg, f)


