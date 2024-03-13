from sklearn.datasets import load_iris, load_digits, load_diabetes, load_boston, load_wine, load_breast_cancer
import rps_ams_auto_modeling

X, y = load_wine(return_X_y=True)
score = rps_ams_auto_modeling.rps_ams_auto_modeling(X, y)
