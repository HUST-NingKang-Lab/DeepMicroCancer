import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import roc_curve, auc, roc_auc_score
import TransferRandomForest as trf
import matplotlib.pyplot as plt
from joblib import load
from itertools import cycle

def roc_auc_calculate(y_query, y_proba):
    classes = np.unique(y_query)
    Nclasses = len(classes)
    y_test = np.zeros((len(y_query), Nclasses))
    for i in range(len(y_query)):
        y_test[i][y_query[i]] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(Nclasses):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_proba[:, i])
        roc_auc[i] = float(format(auc(fpr[i], tpr[i]), '.5f'))
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(Nclasses)]))#数组拼接得到fpr的矩阵

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)  #构造数字都为0的矩阵，为做平均做准备
    for i in range(Nclasses):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= Nclasses

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = float(format(auc(fpr["macro"], tpr["macro"]), '.5f'))
    

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_proba.ravel())
    roc_auc["micro"] = float(format(auc(fpr["micro"], tpr["micro"]), '.5f'))
    
    roc_auc["ovr"] = float(format(roc_auc_score(y_query, y_proba, multi_class='ovr'), '.5f'))
    roc_auc["ovo"] = float(format(roc_auc_score(y_query, y_proba, multi_class='ovo'), '.5f'))
    return roc_auc, fpr, tpr

def roc_curve_plot(roc_auc, fpr, tpr, filename, output):
    Nclasses = len(roc_auc) - 4 # not plot macro, micro, ovr, ovo
    plt.figure()
    lw = 2
    
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "deeppink", "navy"])
    for i, color in zip(range(Nclasses), colors):
        plt.plot( fpr[i], tpr[i], color=color, lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),)

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(filename + " ROC curve")
    plt.legend(bbox_to_anchor=(2.4, 0.5),loc=5,ncol=2)
    filepath = output + '/' + filename + ".pdf"
    plt.savefig(filepath,bbox_inches = 'tight')
    plt.show()
    
features = 'data/blood/X_test.csv'
model = 'models/blood_model'
labels = 'data/blood/y_test.csv'
fig_name = 'tissue-blood'
output = 'results/tissue-blood'

def independent_predict(features, labels, model, fig_name, output):
    X = pd.read_csv(features, index_col = 0)
    clf = load(f'{model}/model.joblib')
    le = load(f'{model}/label_encoder.joblib')
    
    # match the features
    feature_name = pd.read_csv(f'{model}/features.txt', quotechar = "'", header = None).iloc[:, 0].values
    X = X.loc[:, feature_name]
    
    # predict
    y_pre = clf.predict(X)
    y_pre_proba = clf.predict_proba(X)
    y_pre = le.inverse_transform(y_pre)
    
    result_df = pd.DataFrame(y_pre_proba, index = X.index, columns = le.classes_)
    result_df['prediction'] = y_pre
    
    # save the result
    os.makedirs(output, exist_ok = True)
    result_df.to_csv(f'{output}/prediction.csv')
    
    if labels:
        y = pd.read_csv(labels, index_col = 0).apply(le.transform).values.ravel()
        auc, fpr, tpr = roc_auc_calculate(y, y_pre_proba)
        roc_curve_plot(auc, fpr, tpr, f'{fig_name} RandomForest', output)
        
def transfer_predict(features, labels, model, fig_name, output):
    X = pd.read_csv(features, index_col = 0)
    ser_RF = load(f'{model}/ser_model.joblib')
    strut_RF = load(f'{model}/strut_model.joblib')
    le = load(f'{model}/label_encoder.joblib')
    
    # match the features
    feature_name = pd.read_csv(f'{model}/features.txt', quotechar = "'", header = None).iloc[:, 0].values
    X = X.loc[:, feature_name]
    
    # predict
    y_pre = trf.mix_predict(ser_RF, strut_RF, X.values)
    y_pre = le.inverse_transform(y_pre)
    y_pre_proba = trf.mix_predict_proba(ser_RF, strut_RF, X.values)
    
    result_df = pd.DataFrame(y_pre_proba, index = X.index, columns = le.classes_)
    result_df['prediction'] = y_pre
    
    # save the result
    os.makedirs(output, exist_ok = True)
    result_df.to_csv(f'{output}/prediction.csv')
    
    if labels:
        y = pd.read_csv(labels, index_col = 0).apply(le.transform).values.ravel()
        auc, fpr, tpr = roc_auc_calculate(y, y_pre_proba)
        roc_curve_plot(auc, fpr, tpr, f'{fig_name} transfer RandomForest', output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Predict labels for a new dataset')
    parser.add_argument('-i', '--input', type = str, help = 'The path to a new features file as csv format, each row is a sample, each column is a feature')
    parser.add_argument('-l', '--labels', type = str, default = None, help = 'The path to a labels file as csv format, each row is a sample, the disease_type column is the label, if provided, the auc will be calculated')
    parser.add_argument('-m', '--model', type = str, help = 'The path to the model')
    parser.add_argument('-t', '--type', type = str, help = 'The type of the model, either "independent" or "transfer"')
    parser.add_argument('-f', '--fig_name', type = str, default = None, help = 'The name of the figure to save')
    parser.add_argument('-o', '--output', type = str, help = 'The path to save the predictions')
    args = parser.parse_args()

    if args.type == 'independent':
        independent_predict(args.input, args.labels, args.model, args.fig_name, args.output)
    elif args.type == 'transfer':
        transfer_predict(args.input, args.labels, args.model, args.fig_name, args.output)
        