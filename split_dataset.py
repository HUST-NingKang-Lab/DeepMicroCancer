import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(features_file, labels_file, test_size, output_dir):
    # Load the data into a pandas dataframe
    X = pd.read_csv(features_file, index_col = 0)
    y = pd.read_csv(labels_file, index_col = 0)['disease_type']

    # Remove the three cancer types with few samples
    y = y[(y['disease_type'] != 'Kidney Chromophobe') &
          (y['disease_type'] != 'Kidney Renal Clear Cell Carcinoma') &
          (y['disease_type'] != 'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma')]
    X = X.loc[metadata.index]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = test_size, 
                                                        random_state = 0)

    # Save the training and testing sets as csv files
    os.makedirs(output_dir, exist_ok = True)
    X_train.to_csv(f'{output_dir}/X_train.csv')
    X_test.to_csv(f'{output_dir}/X_test.csv')
    y_train.to_csv(f'{output_dir}/y_train.csv')
    y_test.to_csv(f'{output_dir}/y_test.csv')

if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description = 'Split the dataset into training and testing sets')
    parser.add_argument('-x', '--features', type = str, help = 'The path to a features file as csv format, each row is a sample, each column is a feature')
    parser.add_argument('-y', '--labels', type = str, help = 'The path to a labels file as csv format, each row is a sample, the disease_type column is the label')
    parser.add_argument('-s', '--test_size', type = float, default = 0.3, help = 'The size of the test set (default = 0.3)')
    parser.add_argument('-o', '--output', type = str, help = 'The path to save the splited dataset')
    args = parser.parse_args()

    split_dataset(args.features, args.labels, args.test_size, args.output)