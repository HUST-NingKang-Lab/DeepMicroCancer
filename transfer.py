import pandas as pd
import os
import argparse
import TransferRandomForest as trf
from joblib import load, dump

source_model = 'models/tissue_model'
source_features = 'data/tissue/X_train.csv'
source_labels = 'data/tissue/y_train.csv'
target_features = 'data/blood/X_train.csv'
target_labels = 'data/blood/y_train.csv'
output_dir = 'models/tissue-blood_model'

def transfer_model(source_model, 
                   source_features, 
                   source_labels, 
                   target_features, 
                   target_labels, 
                   output_dir):
    
    le = load(f'{source_model}/label_encoder.joblib')
    source_model = load(f'{source_model}/model.joblib')
    source_features = pd.read_csv(source_features, index_col = 0).values
    source_labels = pd.read_csv(source_labels, index_col = 0)
    source_labels = source_labels.apply(le.fit_transform).values.ravel()   # Convert labels to integers
    
    target_features = pd.read_csv(target_features, index_col = 0)
    feature_name = target_features.columns.values
    target_features = target_features.values
    target_labels = pd.read_csv(target_labels, index_col = 0)
    target_labels = target_labels.apply(le.fit_transform).values.ravel()   # Convert labels to integers
    
    # Ser
    gRF_list = trf.forest_convert(source_model)
    ser_RF = trf.forest_SER(gRF_list, source_features, source_labels, C = source_model.n_classes_)
    
    # Strut
    strut_RF = trf.STRUT(source_features, source_labels, target_features, target_labels, n_trees=100, verbos=False)
    
    # save the model to a file
    os.makedirs(output_dir, exist_ok = True)
    feature_name.tofile(f'{output_dir}/features.txt', sep = '\n')
    dump(ser_RF, f'{output_dir}/ser_model.joblib')
    dump(strut_RF, f'{output_dir}/strut_model.joblib')
    dump(le, f'{output_dir}/label_encoder.joblib')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Transfer a random forest classifier to target data')
    parser.add_argument('-s', '--source_model', type = str, help = 'The path to the source model')
    parser.add_argument('-sf', '--source_features', type = str, help = 'The path to a features file as csv format, each row is a sample, each column is a feature')
    parser.add_argument('-sl', '--source_labels', type = str, help = 'The path to a labels file as csv format, each row is a sample, the disease_type column is the label')
    parser.add_argument('-tf', '--target_features', type = str, help = 'The path to a features file as csv format, each row is a sample, each column is a feature')
    parser.add_argument('-tl', '--target_labels', type = str, help = 'The path to a labels file as csv format, each row is a sample, the disease_type column is the label')
    parser.add_argument('-o', '--output', type = str, help = 'The path to save the model')
    args = parser.parse_args()
    
    transfer_model(source_model = args.source_model,
                   source_features = args.source_features,
                   source_labels = args.source_labels,
                   target_features = args.target_features,
                   target_labels = args.target_labels,
                   output_dir = args.output)