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
1. Repository
2. Processflow
   2.1 Training a model
   2.2 Predicting clients
   2.3 Run all
3. Functions
4. 

## 1. Repository
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

## 2. Procesflow
![Schematic processflow](/img/PROCESSFLOW.jpg?raw=true "Schematic procesflow")

The procesflow makes a distincion between 'Training a model' and 'Predicting clients', respectively: 'Train' and 'Predict'.

### 2.1 Training a model
For training a model, the table below provides a brief overview of the steps:

|Step|Input|Process|Output|
|:---|:---|:---|:---|
|**Get data**|Settings (tablenames historical data CBS, rename columns, index columns, etc.)|Collect and combine data to analytical base table|Combined dataset with index on region and period| 
|**Preprocess**|<ul><li>Settings</li><li>Output of 'Get data'</li></ul>|Select, clean, impute and scale|Compressed dataset with index on region and period|
|**Train**|<ul><li>Settings</li><li>Output of 'Preprocess'</li></ul>|Fit multiple models|Grid search object|

### 2.2 Predicting clients
|Step|Input|Process|Output|
|:---|:---|:---|:---|
|**Get data**|Settings (tablenames historical data CBS, rename columns, index columns, etc.)|Collect and combine data to analytical base table|Combined dataset with index on region and period| 
|**Get data predict**|Settings (tablenames prognose data CBS, rename columns, index columns, etc.)|Collect and combine data to analytical base table|Combined dataset of prognoses with index on region and period| 
|**Preprocess predict**|<ul><li>Settings</li><li>Output of 'Get data'</li><li>Output of 'Get data predict'</li></ul>|Select, clean, impute and scale|Combined dataset for future with index on region and period|
|**Preprocess**|<ul><li>Settings</li><li>Output of 'Get data'</li></ul>|Select, clean, impute and scale|Compressed dataset with index on region and period|
|**Predict**|<ul><li>Settings</li><li>Output of 'Preprocess'</li><li>Best model of 'Train'</li></ul>|Apply model|Prediction of the number of Wmo-clients|

### 2.3 Run all
Description of the combined scripts and notebook (added later)

## Functions
Overview of all functions added later, like:
* Preprocess utilities:
    * `get_and_combine_cbs_tables`: Method to get multiple similar tables in the CBS database.

