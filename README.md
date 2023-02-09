This repository contains a diagnosis model using transfer learning techniques for a broad-spectrum of cancer types. The model is built using Random Forest and Random Forest Transfer learning techniques.
# Workflow
The project is divided into four scripts, each with its own function in the overall workflow.

## Split Dataset 
The `split_dataset.py` script is used to split the dataset into training and testing sets. The script takes the path to the features and labels files in csv format as input and generates the training and testing datasets.

### Arguments
```
-x, --features: The path to a features file as csv format, each row is a sample, each column is a feature
-y, --labels: The path to a labels file as csv format, each row is a sample, the disease_type column is the label
-s, --test_size: The size of the test set (default = 0.3)
-o, --output: The path to save the output files

```
### Usage:
Split the tissue dataset and blood dataset into training and testing sets with a test size of 30% and 20% respectively.
```
python split_dataset.py -x data/tissue_snm.csv \
  -s 0.3 \
  -y data/tissue_meta.csv \
  -o data/tissue
  
python split_dataset.py -x data/blood_snm.csv \
  -s 0.2 \
  -y data/blood_meta.csv \
  -o data/blood
```
## Build Model
The `build_model.py` script builds the Random Forest classifier model. The script takes the path to the training features and labels files in csv format as input and outputs a saved model. Saved model contains three files: `model.joblib` contains the model parameters, `features.txt` contains the features used to build the model, and `label_encoder.joblib` contains the label encoder used to encode the labels.

### Arguments
```
-x, --features: The path to a features file as csv format, each row is a sample, each column is a feature
-y, --labels: The path to a labels file as csv format, each row is a sample, the disease_type column is the label
-o, --output: The path to save the model
```

### Usage:
Build the Random Forest classifier model for the tissue and blood datasets.
```
python build_model.py -x data/tissue/X_train.csv \
  -y data/tissue/y_train.csv \
  -o models/tissue_model
  
python build_model.py -x data/blood/X_train.csv \
  -y data/blood/y_train.csv \
  -o models/blood_model
```
## Transfer Model
The `transfer.py` script is used to transfer the model from one dataset to another. The script takes the path to the source model, source features and labels files, target features and labels files in csv format as input and outputs a saved model. The source model should be built using the `build_model.py` script.
### Arguments
```
-s, --source_model: The path to the source model
-sf, --source_features: The path to a source features file as csv format, each row is a sample, each column is a feature
-sl, --source_labels: The path to a source labels file as csv format, each row is a sample, the disease_type column is the label
-tf, --target_features: The path to a target features file as csv format, each row is a sample, each column is a feature
-tl, --target_labels: The path to a target labels file as csv format, each row is a sample, the disease_type column is the label
-o, --output: The path to save the model
```
### Usage:
Transfer the tissue model to the blood dataset.
```
python transfer.py -s models/tissue_model \
  -sf data/tissue/X_train.csv \
  -sl data/tissue/y_train.csv \
  -tf data/blood/X_train.csv \
  -tl data/blood/y_train.csv \
  -o models/tissue-blood_model
```

## Predict
The `predict.py` script is used to predict the labels of the testing dataset. The script takes the path to the testing features and labels files (optional, if the labels file is provided the script will output the predicted labels and the auroc plot) in csv format , the model build using the `build_model.py` or `transfer.py` script, the type of the model, either independent or transfer, the name of the figure to save (optional) the auroc plot and the output path to save the results.

### Arguments
```
-i, --input: The path to a features file as csv format, each row is a sample, each column is a feature
-l, --labels: The path to a labels file as csv format, each row is a sample, the disease_type column is the label
-m, --model: The path to the model
-t, --type: The type of the model, either independent or transfer
-f, --fig_name: The name of the figure to save
-o, --output: The path to save the results
```
### Usage
Predict the labels of the testing dataset for the tissue using the tissue model.
```
python predict.py -i data/tissue/X_test.csv \
  -l data/tissue/y_test.csv \
  -m models/tissue_model \
  -t independent \
  -f tissue-tissue \
  -o results/tissue-tissue
```
Predict the labels of the testing dataset for the blood using the blood model.
```
python predict.py -i data/blood/X_test.csv \
  -l data/blood/y_test.csv \
  -m models/blood_model \
  -t independent \
  -f blood-blood \
  -o results/blood-blood
```
Predict the labels of the testing dataset for the blood using the tissue-blood model.
```
python predict.py -i data/blood/X_test.csv \
  -l data/blood/y_test.csv \
  -m models/tissue-blood_model \
  -t transfer -f tissue-blood \
  -o results/tissue-blood
```
# Feature importances
The feature importances of the tissue model and the blood model for each cancer type are calculate using the `feature_importances.py` script. Each cancer type is considered as a binary classification problem and the output is saved in `feature_importances` folder.

run the script using the following command:
```
python feature_importances.py
```
