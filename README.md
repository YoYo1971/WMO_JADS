# WMO - Predicting the number of WMO clients using aid in householding tasks

JADS Professional Education - Execute track
Group assignment - April 2021
Members of project group:
- Alwin Visser
- Luuk Jans
- Maikel Jonkers
- Mark den Otter
- Nick Neppelenbroek

## Content
1. Business objective of the project
2. Repository
3. Processflow

    3.1. Training a model
    3.2. Predicting clients
    3.3. Run all
4. Functions

## 1. Business objective of the project
This project was developed as a group assignment in the Professional Education Execute track of JADS. The business objective was to develop code that could train a model and predict the number of Wmo-clients that use aid in householding tasks. 

## 2. Repository
In this repository there are an number of folders and files, each with their own goal/function:
* `archive`: Number of notebooks that are used during the development of the code. Most code are changed / not used anymore. Since this is a learning path, the code is saved for the students. 
* `data`: Datafiles, models and logging saved here, to ensure that changes in the source data will not lead to an unusable code. Mainly to ensure reproducibility.
* `img`: Images saved for the README.
* `notebooks`: Notebooks used for development or final report. 
* `src`: Source code of the project
    *`predict`: Utilities/functions for prediction
    * `preprocess`: Utilities/functions for preprocess
    * `run_all`: Scripts where all functions are combined to ensure modular design. In the notebook `run_all.ipynb` all the code is combined to provide a 'user interface'. 
    * `train`: Utilities/functions for train
    * `utilities`: Utilities/functions that are general for all
    * `settings.py`: All the project settings are saved in this file (with docstrings)
    * `mapper_cols.py`: Lists and dictionary used to try different selections to find the optimal feature set. 
* `.gitignore`: Specifies intentionally untracked files that Git should ignore
* `LICENSE`: License of this open source project
* `README.md`: This file ;-)
* `requirements.txt`: Packages used in the scripts and notebooks.

## 3. Procesflow
![Schematic processflow](/img/PROCESSFLOW.jpg?raw=true "Schematic procesflow")

The procesflow makes a distincion between 'Training a model' and 'Predicting clients', respectively: 'Train' and 'Predict'.

### 3.1. Training a model
For training a model, the table below provides a brief overview of the steps:

|Step|Input|Process|Output|
|:---|:---|:---|:---|
|**Get data**|Settings (tablenames historical data CBS, rename columns, index columns, etc.)|Collect and combine data to analytical base table|Combined dataset with index on region and period| 
|**Preprocess**|<ul><li>Settings</li><li>Output of 'Get data'</li></ul>|Select, clean, impute and scale|Compressed dataset with index on region and period|
|**Train**|<ul><li>Settings</li><li>Output of 'Preprocess'</li></ul>|Fit multiple models|Grid search object|

### 3.2. Predicting clients
|Step|Input|Process|Output|
|:---|:---|:---|:---|
|**Get data**|Settings (tablenames historical data CBS, rename columns, index columns, etc.)|Collect and combine data to analytical base table|Combined dataset with index on region and period| 
|**Get data predict**|Settings (tablenames prognose data CBS, rename columns, index columns, etc.)|Collect and combine data to analytical base table|Combined dataset of prognoses with index on region and period| 
|**Preprocess predict**|<ul><li>Settings</li><li>Output of 'Get data'</li><li>Output of 'Get data predict'</li></ul>|Select, clean, impute and scale|Combined dataset for future with index on region and period|
|**Preprocess**|<ul><li>Settings</li><li>Output of 'Get data'</li></ul>|Select, clean, impute and scale|Compressed dataset with index on region and period|
|**Predict**|<ul><li>Settings</li><li>Output of 'Preprocess'</li><li>Best model of 'Train'</li></ul>|Apply model|Prediction of the number of Wmo-clients|

### 3.3. Run all: `src.run_all.run_all.ipynb`
In the folder `src/run_all/` the notebook `run_all.ipynb` is the notebook where all the functionality comes together.
The user can use this notebook to train a model and/or predict the number of Wmo-clients. There are a number of base
settings a user can change accordingly to his/her preference(s). The base settings are:

`PROCES` : str

    String value of the proces to be runned. Options for string are one of the following list: 
    `['train', 'predict', 'train_and_predict']`. Default: 'train_and_predict'.
    
`SOURCE` : str

    String value of the source of the data. As a user you can choose to get the newest data from CBS Statline 
    by running the total script from scratch (will take a number of minutes) or use the collected data as saved 
    in this project. 
    IMPORTANT NOTE: By collecting new data, there may be a possibility that CBS has changed 
    columnnames, deleted tables, etc. In that case the code may run into an error!
    
    Options:
    
    1. Get (new) data from CBS or apply gridsearch on preprocessed data: 'new'
    2. From hardcoded files of get data: 'hardcoded'
    
    Default: 'hardcoded'
    
`PERSONAL_NOTE` : str

    String value to add to the different filenames to make sure the user recognizes the files they have generated. 
    Default: 'PROCESS+'_'+SOURCE'

`PREDICT_PERIODS` : list(int)

    List with integer values for the years to predict, i.e.: [2020, 2021, 2022] 
    IMPORTANT NOTE: During developing historical data was available up and including 2019. The start of this list 
    always needs to be 2020. Default: '[2020, 2021, 2022, 2023]'

`SAVE_ALL` : bool

    Boolean value `[True, False]` if results and logging need to be saved. Default: 'False'.

## 4. Functions
Overview of all functions in the different modules, summed up in order of appearance:

* `src.run_all.get_data.get_data(save_all=False, personal_note="")`: Custom function to get the right dataset for the WMO use case. In this script all the necessary data is loaded for training and/or predicting the WMO clients. Note: Most parameters are loaded by settings.get_data.
    * `src.preprocess.preprocess_utils.get_and_combine_cbs_tables((dict_tables, double_trouble_colnames=None, url='opendata.cbs.nl')`: Method to get multiple similar tables in the CBS database.
    * `src.preprocess.preprocess_utils.rename_and_subset_cols(df, dict_rename, list_cols, include=True)`: Method to rename and subset certain columns from a DataFrame.
    * `src.preprocess.preprocess_utils.get_region_period_spec_val_subtable(df, region=None, period=None, col='typemaatwerkarrangement', spec_value=None)`: Method to subset the dataframe based on a certain region, period and specific value of a column.
    * `src.preprocess.preprocess_utils.downcast_variables_dataframe(df)`: Method to downcast the variables in a DataFrame.
* `src.run_all.get_data.get_data_predict(periods=settings.get_data_predict['LIST_PERIODS'], save_all=True, personal_note="")`: Custom function to get the right future dataset for the WMO use case. In this script a few prognoses are loaded from CBS Statline for predicting the WMO clients. Note: Most parameters are loaded by settings.get_data.
    * `src.preprocess.preprocess_utils.get_and_combine_cbs_tables((dict_tables, double_trouble_colnames=None, url='opendata.cbs.nl')`: Method to get multiple similar tables in the CBS database.
    * `src.preprocess.preprocess_utils.rename_and_subset_cols(df, dict_rename, list_cols, include=True)`: Method to rename and subset certain columns from a DataFrame.
* `src.run_all.preprocess.preprocess_data(df, save_all=False, personal_note="")`: Method to preprocess the data (select columns, impute, scale).
    * `src.preprocess.preprocess_utils.make_df_missing(df)`: Method to calculate the number and percentages of missing values.
    * `src.utilities.transformers.ColumnSelector(cols=None)`: This is a transformer class to select columns/features from a DataFrame.
    * `src.utilities.transformers.GroupInterpolateImputer(groupcols, interpolate_method='linear', cols=None, **kwargs)`: This is a transformer class to impute the missing values with the pd.interpolate method.
    * `src.utilities.transformers.CustomImputer(imputer, cols=None)`: This is a transformer class to impute the selected columns using a defined imputer.
    * `src.utilities.transformers.RelativeColumnScaler(dict_relatively_cols=None)`: This is a transformer class to scale a (number of) column(s) based on another column.
    * `src.utilities.transformers.CustomScaler(cols, scaler)`: This is a transformer class to scale the selected columns using a defined scaler.
* `src.run_all.preprocess.preprocess_data_predict(df_get_data=pd.DataFrame(), df_get_data_predict=pd.DataFrame(), save_all=False, personal_note="")`: Method to preprocess the historical and prognosed data (select columns, impute, scale). 
    * `src.utilities.transformers.GroupInterpolateImputer(groupcols, interpolate_method='linear', cols=None, **kwargs)`: This is a transformer class to impute the missing values with the pd.interpolate method.
* `src.run_all.train.train_and_fit_models(df_preprocessed, filename_input, param_grid=[], save_all=False, personal_note="")`: Method to train model(s) on a preprocessed dataset and return the gridsearch object.
    * `src.train.train_utils.drop_nan_from_specific_columns (df, columns_to_check)`: Drops all rows with nan values in specific columns in a dataframe
    * `src.train.train_utils.rmse_from_neg_mean_squared_error(neg_mean_squared_error)`: Calculates RMSE from the neq mean squared error.
    * `src.train.train_utils.rmse_from_gridsearch_best_estimator(grid_search, X_test, y_test)`: Calculates RMSE from the grid search best estimator.
    * `src.train.train_utils.split_clf_and_params(best_estimator_clf)`: Takes best estimator[clf] and outputs a list with clf and parameters
* `src.run_all.predict.predict_data(trained_model, periods=settings.get_data_predict['LIST_PERIODS'], df_get_data=pd.DataFrame(), df_get_data_predict=pd.DataFrame(), save_all=False, personal_note="")`: Method to predict values in the future with a trained model.
    * `src.run_all.preprocess.preprocess_data_predict(df_get_data=pd.DataFrame(), df_get_data_predict=pd.DataFrame(), save_all=False, personal_note="")`: Method to preprocess the historical and prognosed data (select columns, impute, scale).
    * `src.run_all.preprocess.preprocess_data(df, save_all=False, personal_note="")`: Method to preprocess the data (select columns, impute, scale).
* `src.utilities.utilities.get_latest_file(filename_str_contains, datapath='../data/', filetype='parquet')`: Method to get the latest file to given a certain string value.
* `src.utilities.utilities.list_filenames(path, filename_str_contains)`: Method to get a list of filenames with a certain string value in the name.

