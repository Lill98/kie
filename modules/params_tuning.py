from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pickle
from config.cfg import Configuration as cfg

import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


class HyperTuning:
    def __init__(self) -> None:
        pass
    
    def preprocess_data(self):
        train_feat_pth = os.path.join(cfg.re_post.features_path, 'train.pkl')
        scaler_pth = os.path.join(cfg.re_post.scaler_path, 'scaler.pkl')
        
        features_train = pickle.load(open(train_feat_pth, 'rb'))
        scaler = pickle.load(open(scaler_pth, 'rb'))
        
        X_train, y_train = features_train.values[:, :-1], features_train.values[:, -1].astype(int)
        X_train = scaler.transform(X_train)
        return X_train, y_train
    
    def bo_params_xgb(self, max_depth, gamma, learning_rate, n_estimators, subsample):    
        X_train, y_train = self.preprocess_data()
        params = {
            'max_depth': int(max_depth),
            'gamma': gamma,
            'learning_rate':learning_rate,
            'subsample': subsample,
            'eval_metric': 'auc',
            'n_estimators':int(n_estimators)
        }
        
        scores = cross_val_score(xgb.XGBClassifier(random_state=1997, **params,use_label_encoder=False),
                                X_train, y_train,cv=5,scoring="f1").mean()
        return scores.mean()
    
    def optimize(self):
        xgb_bo = BayesianOptimization(self.bo_params_xgb, {'max_depth': (3, 40),
                                            'gamma': (0, 1),
                                            'learning_rate':(0,1),
                                            'subsample':(0.5,1),
                                            'n_estimators':(100,400)})
        xgb_bo.maximize(n_iter=5, init_points=10)
        params = xgb_bo.max['params']
        params['max_depth']= int(params['max_depth'])
        params['n_estimators']= int(params['n_estimators'])
        params['eval_metric'] = 'auc'
        print(params)
        
if __name__ == "__main__":
    xgb_ob = HyperTuning()
    xgb_ob.optimize()