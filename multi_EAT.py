import warnings

import numpy as np
from numpy.core.records import fromfile
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.utils import shuffle

warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')
warnings.simplefilter(action='ignore', category=FutureWarning)
from collections import defaultdict
from multiprocessing import Lock, Manager, Process, managers, process

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


class Ensemble_Adaptive_Training():

    def __init__(self, n_iter=10 , learning_rate= 0.14, n_estimators = 200, gama = 0.7,h = "lgbm", order = "random" ) :
        """多目标森林参数设置

        Args:
            n_iter (type): int
            learning_rate (type): float
            n_estimators (type): int
            gama (type): float
            order (type): list or string
        """
        self.bagging_iter = 0
        self.sec_num_tree = 0
        self.sec_num_tree_obj = self.sec_num_tree
        self.numtree = 0
        self.note = []
        self.model = []
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.model_box = Manager().list()
        self.n_estimators = n_estimators
        self.sample_weight = None #采样分布，none默认为均匀分布
        self.gama = gama
        self.h = h
        self.order = order # 链序，来自fcc 方法
        self.suc = Manager().list()
        self.__select_h()

    def __select_h (self):
        """选择回归器"""
        
        if self.h == "lgbm":
            self.regressor = LGBMRegressor(boosting_type='gbdt',  learning_rate=self.learning_rate, n_estimators=self.n_estimators)
        elif self.h == "xgb":
            self.regressor =  XGBRegressor(objective='reg:squarederror',  learning_rate=self.learning_rate, n_estimators=self.n_estimators)
        else:
            pass

    def get_c(self,x_train,y_train,x_test,y_test):
        """
        [预训练，得到评价指标]

        Args:
            x_train ([type]): [numpy.ndarray]
            y_train ([type]): [numpy.ndarray]
            x_test ([type]): [numpy.ndarray]
            y_test ([type]): [numpy.ndarray]

        Returns:
            [type]: [numpy.ndarray]"""

        mt = MultiOutputRegressor(estimator = self.regressor).fit(x_train, y_train)
        pred = mt.predict(x_test)
        c = r2_score(y_test, pred ,multioutput= "raw_values")
    
        return c
    
    def shuffle_dataset(self,y_train,y_test): 
        """打乱训练集和测试集的目标值

        Args:
            y_train ([type]): [numpy.ndarray]
            y_test ([type]): [numpy.ndarray]

        Returns:
            [type]: [numpy.ndarray]
        """
        self.l = np.shape(y_train)[1]
        self.shuffle_index = np.arange(self.l)
        np.random.shuffle(self.shuffle_index)
        self.y_train_shuffled = y_train[:,self.shuffle_index]
        self.y_test_shuffled = y_test[:,self.shuffle_index]
        return self.y_train_shuffled,self.y_test_shuffled, self.shuffle_index

    def new_bagging_train(self,x_train,y_train,x_test,y_test,Limit_shuffled,iter):
        """训练

        Args:
            x_train ([type]): [numpy.ndarray]
            y_train ([type]): [numpy.ndarray]
            x_test ([type]): [numpy.ndarray]
            y_test ([type]): [numpy.ndarray]
            Limit_shuffled ([type]): [numpy.ndarray]
            iter ([type]): [int]
        """
        n=np.shape(x_train)[0]
        inbag_index = np.random.choice(np.arange(n), size=n, replace=False, p= self.sample_weight)
        outbag_index=np.setdiff1d(np.arange(n),inbag_index)
        self.inbag_x_train=x_train[inbag_index,:]
        self.inbag_y_train=y_train[inbag_index,:]
        self.outbag_x_train=x_train[outbag_index,:]
        self.outbag_y_train=y_train[outbag_index,:]
        mt = MultiOutputRegressor(estimator = self.regressor).fit(x_train, y_train)
        y_pred_inbag = mt.predict(x_train)
        y_pred = mt.predict(x_test)
        r2 = r2_score(y_test, y_pred ,multioutput= "raw_values")

        minmax=MinMaxScaler()
        y_train_norm = minmax.fit_transform(y_train)
        y_pred_norm = minmax.fit_transform(y_pred_inbag)
        error_vect = np.abs(y_pred_norm - y_train_norm)
        sample_mask = self.sample_weight > 0
        error_1d = np.sum(error_vect, axis=1)
        masked_sample_weight = self.sample_weight[sample_mask]          #找到需要更新的权重
        masked_error_vector = error_1d[sample_mask]                   #对应的abs误差
        error_max = masked_error_vector.max()                           #找到极值误差，算出一个所谓的误差率的概念，可理解为误差归一化
        if error_max != 0:
            masked_error_vector /= error_max
        masked_error_vector **= 2                                       #对应的平方误差率
        estimator_error = (masked_sample_weight * masked_error_vector).sum()   #平均损失，权重乘误差

        # 系数alpha，本来是每棵树的权重，在这里只为了更新样本采样权重作为计算中间值
        alpha = estimator_error / (1. - estimator_error)         
        zk = (self.sample_weight * np.power(alpha, self.gama * (1. - masked_error_vector))).sum()
        self.sample_weight[sample_mask] = self.sample_weight[sample_mask] * np.power(alpha, self.gama * (1. - masked_error_vector)) / zk

        c=[]
        for i in range (len(Limit_shuffled)):
            if (r2[i] - Limit_shuffled[i])/Limit_shuffled[i] >= -0.05:
                c.append(1)
            else:
                c.append(0)
        d=sum(c)
        self.bagging_iter += 1

        if self.bagging_iter > 100:
            self.note.append(0)
            return
        if d == len(Limit_shuffled) :    #如果所有y的R2值都比相应维度的评价指标高，则采用这个预测值，否则，重新bagging
            pass
        else:
            self.new_bagging_train(x_train,y_train,x_test,y_test,Limit_shuffled,iter)


        if d == len(Limit_shuffled) :
            mt_ok = RegressorChain(base_estimator=self.regressor,order = self.order).fit(x_train, y_train)
            self.sec_num_tree+=1
            self.note.append(1)
            self.numtree+=1
        return mt_ok

    def single_fit(self,model_box,suc,i,x_train,y_train,lock):
        
        """单个进程的训练函数

        Args:
            i ([type]): [int] 进程数索引
            x_train ([type]): [numpy.ndarray]
            y_train ([type]): [numpy.ndarray]
        """
        lock.acquire()
        x_retrain,x_retest,y_retrain,y_retest=train_test_split(x_train,y_train,test_size=0.1)
        limit = self.get_c(x_retrain,y_retrain,x_retest,y_retest)
        n= len(x_retrain)
        self.sample_weight = np.ones((n,))/n
        model = self.new_bagging_train(x_retrain,y_retrain,x_retest,y_retest,limit,i)
        if self.note[0] == 1 :
            suc.append(1)
        else :
            suc.append(0)
        model_box.append(model)
        self.bagging_iter = 0
        lock.release()
    
    def fit(self,x_train,y_train):
        """fit 函数重写，支持多进程

        Args:
            x_train ([type]): [numpy.ndarray]
            y_train ([type]): [numpy.ndarray]
        """
        lock = Lock()
        processes = []
        for i in range (self.n_iter):
            p = Process(target=self.single_fit, args=(self.model_box,self.suc,i,x_train,y_train,lock))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        self.l = np.shape(y_train)[1]
        # print(self.suc)


    def get_params(self,deep = True):
        dict = {
        'n_estimators':self.n_estimators,
        'learning_rate':self.learning_rate,
        'n_iter':self.n_iter,
        'gama':self.gama}
        
        return dict

    def set_params(self, **params) :
        if not params:
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


    def predict(self,x_test) :
        """模型预测

        Args:
            x_test ([type]): [numpy.ndarray]

        Returns:
            [type]: [numpy.ndarray]
        """
        pred = np.zeros(1)
        for model in self.model_box:
            pred = pred + model.predict(x_test)
        return pred/len(self.model_box)

