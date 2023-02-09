import pandas as pd
import os
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump

def build_model(features_file, labels_file, output_dir):
    # Load the features and labels into pandas dataframes
    features = pd.read_csv(features_file, index_col = 0)
    labels = pd.read_csv(labels_file, index_col = 0)
    le = LabelEncoder()
    labels = labels.apply(le.fit_transform)

    # Build a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=13, n_jobs=-1)
    clf.fit(features, labels.values.ravel())

    # Save the model to a file
    os.makedirs(output_dir, exist_ok = True)
    features.columns.values.tofile(f'{output_dir}/features.txt', sep = '\n')
    dump(clf, f'{output_dir}/model.joblib')
    dump(le, f'{output_dir}/label_encoder.joblib')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Build a random forest classifier')
    parser.add_argument('-x', '--features', type = str, help = 'The path to a features file as csv format, each row is a sample, each column is a feature')
    parser.add_argument('-y', '--labels', type = str, help = 'The path to a labels file as csv format, each row is a sample, the disease_type column is the label')
    parser.add_argument('-o', '--output', type = str, help = 'The path to save the model')
    args = parser.parse_args()
    
    build_model(args.features, args.labels, args.output)