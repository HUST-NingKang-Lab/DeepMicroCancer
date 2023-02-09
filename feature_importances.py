import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

def RF_importance(X, y):
    clf = RandomForestClassifier(n_estimators = 100, random_state = 13, n_jobs = -1)
    clf.fit(X, y)
    return clf.feature_importances_

if __name__ == '__main__':
    tissue_X = pd.read_csv('data/tissue/X_train.csv', index_col = 0)
    tissue_y = pd.read_csv('data/tissue/y_train.csv', index_col = 0)['disease_type']
    
    blood_X = pd.read_csv('data/blood/X_train.csv', index_col = 0)
    blood_y = pd.read_csv('data/blood/y_train.csv', index_col = 0)['disease_type']
    
    cancer_type = tissue_y.unique()
    
    for cancer in tqdm(cancer_type):
        # binary prediction on tissue
        y_tissue_tmp = tissue_y.copy()
        y_tissue_tmp = y_tissue_tmp.apply(lambda x: 1 if x == cancer else 0)
        tissue_importance = RF_importance(tissue_X, y_tissue_tmp)
        
        # on blood
        y_blood_tmp = blood_y.copy()
        y_blood_tmp = y_blood_tmp.apply(lambda x:1 if x == cancer else 0)
        blood_importance = RF_importance(blood_X, y_blood_tmp)
        
        # save
        os.makedirs('feature_importances', exist_ok = True)
        result_df = pd.DataFrame({'tissue': tissue_importance,
                                  'blood': blood_importance}, index = tissue_X.columns)
        result_df.to_csv(f'feature_importances/{cancer}.csv')
        
    tissue_importance = RF_importance(tissue_X, tissue_y)
    blood_importance = RF_importance(blood_X, blood_y)
    result_df = pd.DataFrame({'tissue': tissue_importance,
                              'blood': blood_importance}, index = tissue_X.columns)
    result_df.to_csv('feature_importances/overall.csv')
        