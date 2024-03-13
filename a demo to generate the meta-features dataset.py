from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor,\
     HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import load_iris, load_digits, load_diabetes, load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import time


def de_mean(x):
    x_bar = np.mean(x)
    return [x_i - x_bar for x_i in x]


def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    return dot(v, v)


def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n-1)


def correlation(x, y):
    stdev_x = np.std(x)
    stdev_y = np.std(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0


def compute_adjusted_r2(r2, exception):
    if r2 <= exception:
        r2 = exception
    return r2


# import test function TF1 (training set)
# def test_fun_abs(x, d=5):
#     m = 0
#     # n = 0
#     for i in range(d):
#         m = m+np.abs(x[:, i])
#         # n = n+np.cos(x[:, i])
#     # print(rand_f)
#     m = m + 0.4*np.random.random(size)
#     print(m.shape)
#     return m
#
#
# np.random.seed(int(time.time()))
# d = 5
# size = 2000
# x_in = 8*(np.random.random((size, d))-0.5)
# x = x_in
# # print(x_in.shape, '\n')
# y = test_fun_abs(x_in, d)

# import test function TF2 (training set)
# def test_fun_square(x, d=16):
#     m = 0
#     # n = 0
#     for i in range(d):
#         m = m+(x[:, i])**2
#         # n = n+np.cos(x[:, i])
#     # print(rand_f)
#     m = m + 0.4*np.random.random(size)
#     print(m.shape)
#     return m
#
#
# np.random.seed(int(time.time()))
# d = 16
# size = 2000
# x_in = 8*(np.random.random((size, d))-0.5)
# x = x_in
# y = test_fun_square(x_in, d)

# # import test function TF3
# def test_fun_3(x, d=40):
#     m = 0
#     n = 1
#     for i in range(d):
#         m = m+np.abs(x[:, i])
#         n = n*x[:, i]
#     # print(rand_f)
#     result = m + n + 0.4*np.random.random(size)
#     print(result.shape)
#     return result
#
#
# np.random.seed(int(time.time()))
# d = 40
# size = 2000
# x_in = 2*(np.random.random((size, d))-0.5)
# x = x_in
# # print(x_in.shape, '\n')
# y = test_fun_3(x_in, d)

# # import test function TF4
# def test_fun_4(x, d=8):
#     m = 0
#     for i in range(d):
#         m = m+abs(x[:, i]*np.sin(x[:, i])+0.1*x[:, i])
#     return m
#
#
# np.random.seed(int(time.time()))
# d = 8
# size = 2000
# x_in = 10*(np.random.random((size, d))-0.5)
# x = x_in
#
# # print(x_in.shape, '\n')
# y = test_fun_4(x_in, d)

# # import test function TF5
# def test_fun_Michalewicz1(x, d=10):
#     m = 0
#     for i in range(d):
#         m = m+(np.sin(x[:, i]))*(np.sin((i+1)*(x[:, i]**2)/np.pi)) + 0.4*np.random.random(size)
#     return -m
#
#
# np.random.seed(int(time.time()))
# d = 10
# size = 2000
# x_in = 2*(np.random.random((size, d))-0.5)
# x = x_in
# # print(x_in.shape, '\n')
# y = test_fun_Michalewicz1(x_in, d)

# # import test function TF6
# def test_fun6_ackley(x, d=8):
#     m = 0
#     n = 0
#     for i in range(d):
#         m = m+x[:, i]**2
#         n = n+np.cos(x[:, i])
#     return -np.exp(-np.sqrt(m) / d) - np.exp(n/d) + np.exp(1)
#
#
# np.random.seed(int(time.time()))
# d = 8
# size = 2000
# x_in = 8*(np.random.random((size, d))-1)
# x = x_in
# # print(x_in.shape, '\n')
# y = test_fun6_ackley(x_in, d)

# # import test function TF7
# def test_fun7_h1(x, d=12):
#     # space for (-0.5, 0.5)
#     m = 0
#     # n = 0
#     for i in range(d):
#         # print(x[:, i])
#         # m = m+x[:, i]**2
#         m = m + np.abs(np.sin(10*np.pi*x[:, 1])/10*np.pi*x[:, 1])
#         # print(m)
#         # n = n+np.cos(x[:, i])
#     return m
#
#
# np.random.seed(int(time.time()))
# d = 12
# size = 2000
# # print(np.random.random((size, d))-0.5)
# x_in = np.random.random((size, d))-0.5
# # print(x_in)
# x = x_in
# # print(x_in.shape, '\n')
# y = test_fun7_h1(x_in, d)

# import test function TF8
# def test_fun8_griewank(x, d=16):
#     # space for (-0.5, 0.5)
#     m = 0
#     n = 1
#     for i in range(d):
#         m = m + x[:, i]**2/4000
#         n = n*np.cos(x[:, i]/np.sqrt(i+1)) + 0.4*np.random.random(size)
#         # result = m - n + 1
#     return m-n+1
#
#
# np.random.seed(int(time.time()))
# d = 16
# size = 2000
# # print(np.random.random((size, d))-0.5)
# x_in = 5*(np.random.random((size, d))-0.5)
# # print(x_in)
# x = x_in
# # print(x_in.shape, '\n')
# y = test_fun8_griewank(x_in, d)

# import test function TF9
# def test_fun9_Rastrigin(x, d=20):
#     # space for (-0.5, 0.5)
#     m = 0
#     n = 0
#     for i in range(d):
#         m = m + x[:, i]**2
#         n = n - 10*np.cos(2*np.pi*x[:, i])
#         # result = m - n + 1
#     return m+n+10*d
#
#
# np.random.seed(int(time.time()))
# d = 20
# size = 5000
# x_in = 10.24*(np.random.random((size, d))-0.5)
# x = x_in
# y = test_fun9_Rastrigin(x_in, d)

# import test function TF10
# def test_fun10_schwefel(x, d=10):
#     m = 0
#     for i in range(d):
#         m = m+x[:, i]*np.sin(np.sqrt(abs(x[:, i])))
#     return 418*d-m
#
#
# np.random.seed(int(time.time()))
# d = 20
# size = 2000
# x_in = 10*(np.random.random((size, d))-0.5)
# x = x_in
# y = test_fun10_schwefel(x_in, d)

# import test function TF11
# def test_fun11_michalewicz2(x, d=12):
#     m = 0
#     for i in range(d):
#         m = m+np.sin(x[:, i])*(np.sin(i*(x[:, i])**2/np.pi)**4)+0.4*np.random.random(size)
#     return -m
#
#
# np.random.seed(int(time.time()))
# d = 12
# size = 20000
# # print(np.random.random((size, d))-0.5)
# x_in = 1*np.pi*(np.random.random((size, d)))
# x = x_in
# # print(x_in.shape, '\n')
# y = test_fun11_michalewicz2(x_in, d)

# import test function TF12
# def test_fun12_h5(x, d=16):
#     # space for (-0.5, 0.5)
#     m = 0
#     # n = 0
#     for i in range(d):
#         m = m - x[:, i]*np.sin(10*np.pi*x[:, i])
#     return m
#
#
# np.random.seed(int(time.time()))
# d = 16
# size = 10000
# x_in = (np.random.random((size, d))-0.5)*1
# x = x_in
# y = test_fun12_h5(x_in, d)

# # import test function TF13
# def test_fun13_rosenbrock(x, d=18):
#     # space for (-0.5, 0.5)
#     m = 0
#     n = 0
#     for i in range(d):
#         if i == 0:
#             m = (1-x[:, i])**2
#         elif i == 1:
#             n = (x[:, i] - x[:, 0]**2)**2
#         result = m+n
#         # print(x[:, i])
#         # m = m+x[:, i]**2
#         # print(m)
#         # n = n+np.cos(x[:, i])
#     return result
#
#
# np.random.seed(int(time.time()))
# d = 18
# size = 800
# x_in = (np.random.random((size, d))-0.5)*20
# y = test_fun13_rosenbrock(x_in, d)
# x = np.array(x_in)


# # 1.import dataset Bike-sharing
#
# bike_sharing_problem = pd.read_csv('./dataset/Bike-Sharing-Dataset/hour.csv', header=None)
#
# bike_sharing_problem = bike_sharing_problem.dropna()
#
# bike_sharing_problem = bike_sharing_problem[1:]
# x = bike_sharing_problem.reindex(columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# y = bike_sharing_problem.reindex(columns=[13, 14, 15])
# print(x.shape)
# print(y.shape)
# x = np.array(x)
# y = np.array(y)
#
# np.random.seed(40)
# np.random.shuffle(x)
# np.random.seed(40)
# np.random.shuffle(y)
#
# row = x.shape[0]
#
# y = y[:, 2]
#
# x = x[:4000]
# y = y[:4000]

# # 2.import boston dataset
#
# x = load_boston().data
# y = load_boston().target

# 3.import abalone datasets

# abalone = pd.read_csv('./dataset/abalone.data', header=None)
#
# abalone = abalone.dropna()
#
# x = abalone.reindex(columns=[1, 2, 3, 4, 5, 6, 7])
# y = abalone.reindex(columns=[8])
#
# sex = abalone.reindex(columns=[0])
# sex_str = {0: 'F', 1: 'I', 2: 'M'}
#
# for i in range(3):
#     # print(cut_str[i])
#     sex = sex.replace(sex_str[i], i)
#
# x = np.hstack([sex, x])

# # 3.import abalone datasets
# x = load_wine().data
# y = load_wine().target

# # 4.import diamonds datasets
# diamonds = gds.diamonds
# x = diamonds.reindex(columns=['carat', 'depth', 'table', 'x', 'y', 'z'])
# cut = diamonds[['cut']]
# cut_str = {0: 'Fair', 1: 'Good', 2: 'Very Good', 3: 'Premium', 4: 'Ideal'}
#
# for i in range(5):
#     cut = cut.replace(cut_str[i], i)
#
# color = diamonds[['color']]
# color_str = {0: 'J', 1: 'I', 2: 'H', 3: 'G', 4: 'F', 5: 'E', 6: 'D'}
# for j in range(7):
#     color = color.replace(color_str[j], j)
#
# clarity = diamonds[['clarity']]
# clarity_str = {0: 'I1', 1: 'SI2', 2: 'SI1', 3: 'VS2', 4: 'VS1', 5: 'VVS2', 6: 'VVS1', 7: 'IF'}
# for k in range(8):
#     clarity = clarity.replace(clarity_str[k], k)
# cut = np.array(cut)
# color = np.array(color)
# clarity = np.array(clarity)
#
# x = np.hstack([x, cut, color, clarity])
# print(x.shape)
#
# y = diamonds.reindex(columns=['price'])
# y = np.array(y)
# x = x[:3000]
# y = y[:3000]

# 5.import Pima Indian's diabetes datasets
# data = pd.read_csv('./dataset/PimaIndiansDiabetes-main/pima-indians-diabetes.csv')
# y = data['Class']
# x = data.iloc[:, :-2]

# # 6.import MPG datasets
# datasetPath = tf.keras.utils.get_file("auto-mpg.data",
#                                       "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
#
# columnNames = ["MPG","Cylinders","Displacement","Horsepower","Weight","Accleration","Model Year","Origin"]
# rawDataset = pd.read_csv(datasetPath, names=columnNames, na_values="?",comment="\t",sep=" ",skipinitialspace=True)
# dataset = rawDataset.copy()
# # print(dataset)
#
# dataset = dataset.dropna()
# dataset.isna().sum()
#
# x = dataset.reindex(columns=["Cylinders","Displacement","Horsepower","Weight","Accleration","Model Year","Origin"])
# # print(x)
# y = dataset.reindex(columns=['MPG'])

# # 7.import diabetes datasets
# x = load_diabetes().data
# y = load_diabetes().target

# # 7.import breast_cancer datasets
# x = load_breast_cancer().data
# y = load_breast_cancer().target

# # 8.import seeds datasets
# seeds = pd.read_csv('./dataset/seeds_dataset.csv', header=None)
# seeds = seeds.dropna()
#
# x = seeds.reindex(columns=[0, 1, 2, 3, 4, 5, 6])
# y = seeds.reindex(columns=[7])

# # 9.import iris datasets
# x = load_iris().data
# y = load_iris().target

# # 10.import penguins datasets
# penguins = pd.read_csv('./dataset/penguins_size.csv', header=None)
#
# penguins = penguins.dropna()
# # print(diamond)
# x = penguins.reindex(columns=[2, 3, 4, 5])
# x = x[1:]
# # print(x)
# X = np.array(x, dtype=np.float64)
# # print(X.shape)
#
# sex = penguins.reindex(columns=[6])
# sex_str = {0: 'FEMALE', 1: 'MALE'}
# sex = sex[1:]
# for i in range(2):
#     # print(cut_str[i])
#     sex = sex.replace(sex_str[i], i)
# # print(sex)
#
# y = penguins.reindex(columns=[0])
# y = y[1:]
# species_str = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
# for j in range(3):
#     # print(cut_str[i])
#     y = y.replace(species_str[j], j)
# # print(y)
#
# X = np.hstack([X, sex])
# X = X.astype(float)
# x = X

# import Bogie STRESS and MASS fitting problem (engineering application)

data = pd.read_excel('./dataset/x_train_DOE.xlsx', header=None)
myData = data
x = myData.reindex(columns=[0, 1, 2, 3, 4, 5, 6, 7])

# y = myData.reindex(columns=[17])
y = myData.reindex(columns=[8])

# pre-procession

x = np.array(x)
y = np.array(y)
y = np.ravel(y)

np.random.seed(42)
np.random.shuffle(x)
np.random.seed(42)
np.random.shuffle(y)

x = x.astype(float)
y = y.astype(float)

x_average = np.mean(x, axis=0)
y_average = np.mean(y, axis=0)

col0 = x.shape[1]
row0 = x.shape[0]
col = 50
x_average_add = np.zeros([col - col0], dtype=float)
x_average = np.concatenate([x_average, x_average_add])
# print(x.shape, y.shape)
n_split = 1
index = 1
# feature of data: Mean, max, min, variance, num the first column from PCA

# x = np.array_split(x, n_split)
meta_feature_set_with_prediction = np.empty(((int(n_split*index)), int(72)), dtype=float)
batch_size = int(row0/n_split)

for k in range(index):
    y_ = y

    y_ = np.ravel(y_)
    y_ = np.array(y_)

    x_train = x
    y_train = y_

    x_train = np.array_split(x_train, n_split)
    y_train = np.array_split(y_train, n_split)

    for i in range(int(n_split)):
        # define the surrogate model
        rf = RandomForestRegressor()
        dt = tree.DecisionTreeRegressor()
        svr_rbf = svm.SVR()
        gbr = GradientBoostingRegressor()
        et = ExtraTreesRegressor()
        abr = AdaBoostRegressor()
        hgbr = HistGradientBoostingRegressor()

        poly_Feature = PolynomialFeatures()
        linear = linear_model.LinearRegression()

        Nusvr_rbf = svm.NuSVR()

        svr_linear = svm.SVR(kernel='linear')
        svr_poly = svm.SVR(kernel='poly')
        svr_sigmoid = svm.SVR(kernel='sigmoid')

        Nusvr_linear = svm.NuSVR(kernel='linear')
        Nusvr_poly = svm.NuSVR(kernel='poly')
        Nusvr_sigmoid = svm.NuSVR(kernel='sigmoid')

        row = x_train[i].shape[0]

        x_train_add = np.zeros([row, col - col0], dtype=float)
        x_train_reshape = np.concatenate([x_train[i], x_train_add], axis=1)

        x_train_average = np.mean(x_train_reshape, axis=0)
        x_train_std = np.std(x_train_reshape, axis=0)

        y_train_average = np.mean(y_train[i], axis=0)

        y_train_std = np.std(y_train[i], axis=0)

        for j in range(50):
            meta_feature_set_with_prediction[k * (int(n_split)) + i, j] = correlation(x_train_reshape[:, j], y_train[i])

        meta_feature_set_with_prediction[k * (int(n_split)) + i, 50] = y_train_average - y_average
        # r2[k * (int(n_split)) + i, 100] = y_train_average

        meta_feature_set_with_prediction[k * (int(n_split)) + i, 51] = y_train_std

        y_train_skewness = stats.skew(y_train[i])
        y_train_kurtosis = stats.kurtosis(y_train[i])
        meta_feature_set_with_prediction[k * (int(n_split)) + i, 52] = y_train_skewness

        meta_feature_set_with_prediction[k * (int(n_split)) + i, 53] = y_train_kurtosis

        meta_feature_set_with_prediction[k * (int(n_split)) + i, 54] = batch_size

        meta_feature_set_with_prediction[k * (int(n_split)) + i, 55] = col0

        error_value = -1
        cv = 10
        target = np.empty(int(12), dtype=float)

        score_RF = cross_val_score(rf, x_train[i], y_train[i], cv=cv, scoring='r2').mean()
        target[0] = compute_adjusted_r2(score_RF, error_value)
        print("r2 score RF:       ", score_RF)

        score_DT = cross_val_score(dt, x_train[i], y_train[i], cv=cv, scoring='r2').mean()
        target[1] = compute_adjusted_r2(score_DT, error_value)
        print("r2 score DT:       ", score_DT)

        score_ET = cross_val_score(et, x_train[i], y_train[i], cv=cv, scoring='r2').mean()
        target[2] = compute_adjusted_r2(score_ET, error_value)
        # r2 of the representative surrogate model
        meta_feature_set_with_prediction[k * (int(n_split)) + i, 56] = score_ET
        print("r2 score ET:       ", score_ET)

        score_GBR = cross_val_score(gbr, x_train[i], y_train[i], cv=cv, scoring='r2').mean()
        target[3] = compute_adjusted_r2(score_GBR, error_value)
        meta_feature_set_with_prediction[k * (int(n_split)) + i, 57] = score_GBR
        print("r2 score GBR:      ", score_GBR)

        score_ABR = cross_val_score(abr, x_train[i], y_train[i], cv=cv, scoring='r2').mean()
        target[4] = compute_adjusted_r2(score_ABR, error_value)
        print("r2 score ABR:      ", score_ABR)

        score_HGBR = cross_val_score(hgbr, x_train[i], y_train[i], cv=cv, scoring='r2').mean()
        target[5] = compute_adjusted_r2(score_HGBR, error_value)
        print("r2 score HGBR:     ", score_HGBR)

        score_Poly = cross_val_score(linear, poly_Feature.fit_transform(x_train[i]), y_train[i], cv=cv,
                                     scoring='r2').mean()
        meta_feature_set_with_prediction[k * (int(n_split)) + i, 58] = compute_adjusted_r2(score_Poly, error_value)
        print("r2 score Pol:      ", score_Poly)

        score_SVR_rbf = cross_val_score(svr_rbf, x_train[i], y_train[i], cv=cv, scoring='r2').mean()
        target[6] = compute_adjusted_r2(score_SVR_rbf, error_value)
        meta_feature_set_with_prediction[k * (int(n_split)) + i, 59] = score_SVR_rbf
        print("r2 score SVR_rbf:  ", score_SVR_rbf)

        score_SVR_poly = cross_val_score(svr_poly, x_train[i], y_train[i], cv=cv, scoring='r2').mean()
        target[7] = compute_adjusted_r2(score_SVR_poly, error_value)
        print("r2 score SVR_pol:  ", score_SVR_poly)

        score_SVR_sigmoid = cross_val_score(svr_sigmoid, x_train[i], y_train[i], cv=cv, scoring='r2').mean()
        target[8] = compute_adjusted_r2(score_SVR_sigmoid, error_value)
        print("r2 score SVR_sig:  ", score_SVR_sigmoid)

        score_NuSVR_rbf = cross_val_score(Nusvr_rbf, x_train[i], y_train[i], cv=cv, scoring='r2').mean()
        target[9] = compute_adjusted_r2(score_NuSVR_rbf, error_value)
        print("r2 score NuSVR_rbf:", score_NuSVR_rbf)

        score_NuSVR_poly = cross_val_score(Nusvr_poly, x_train[i], y_train[i], cv=cv, scoring='r2').mean()
        target[10] = compute_adjusted_r2(score_NuSVR_poly, error_value)
        print("r2 score NuSVR_pol:", score_NuSVR_poly)

        score_NuSVR_sigmoid = cross_val_score(Nusvr_sigmoid, x_train[i], y_train[i], cv=cv, scoring='r2').mean()
        target[11] = compute_adjusted_r2(score_NuSVR_sigmoid, error_value)
        meta_feature_set_with_prediction[k * (int(n_split)) + i, 60:72] = target
        print("r2 score NuSVR_sig:", score_NuSVR_sigmoid)

meta_feature_datasets = pd.DataFrame(meta_feature_set_with_prediction, columns=None)


writer = pd.ExcelWriter('data_boston'+str(batch_size)+'.xlsx')		# save in Excel
meta_feature_datasets.to_excel(writer, 'sheet1', float_format='%.5f', header=None, index=False)
writer.save()

writer.close()
