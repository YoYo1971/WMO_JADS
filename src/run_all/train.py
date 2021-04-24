# Imports
import sys
sys.path.append('../../')
from datetime import datetime
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# Custom functions
import src.settings as settings
from src.train.train_utils import drop_nan_from_specific_columns, split_clf_and_params, \
    rmse_from_neg_mean_squared_error, rmse_from_gridsearch_best_estimator


def train_and_fit_models(df_preprocessed, filename_input, param_grid=[], save_all=False, personal_note=""):
    """
    Method to train model(s) on a preprocessed dataset.

    Parameters
    ----------
    df_preprocessed : pd.DataFrame()

    filename_input : str
    param_grid : list(dict(str:list()))
    save_all: Bool
        Boolean value to save DataFrame and settings
    personal_note : str
        String value to add to the filename to make recognition of the saved files easier.

    Returns
    -------
    Complete gridsearch object, where `gridsearchobject.best_estimator_` will give the best model.
    """

    # Delete rows where target value is NaN
    df = df_preprocessed.copy()
    df = drop_nan_from_specific_columns(df, settings.train['Y_VALUE'])

    # Make X and y dataset
    X = df.drop(settings.Y_TARGET_COLS, axis=1)
    y = df[settings.train['Y_VALUE']]
    # Split X and y into train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings.train['TEST_SIZE'], random_state=settings.train['RANDOM_STATE'])

    # Construct basic pipeline for gridsearch
    pl_gs_total = Pipeline([('clf', LinearRegression())])  # Placeholder Estimator

    # Make/use param_grid for all classifiers and hyper parameters
    if len(param_grid) == 0:
        param_grid_total = [{'clf': [LinearRegression()],
                             'clf__normalize': settings.train['GRIDSEARCH_NORMALIZE'], },

                            {'clf': [Ridge()],
                             'clf__alpha': settings.train['GRIDSEARCH_ALPHA']},

                            {'clf': [Lasso()],
                             'clf__alpha': settings.train['GRIDSEARCH_ALPHA']},

                            {'clf': [KNeighborsRegressor()],
                             'clf__n_neighbors': settings.train['GRIDSEARCH_NEIGHBORS']},

                            {'clf': [XGBRegressor()],
                             'clf__gamma': settings.train['GRIDSEARCH_GAMMA'],
                             'clf__n_estimators': settings.train['GRIDSEARCH_N_ESTIMATORS']},
                            ]
    else:
        param_grid_total = param_grid

    # Initiate gridsearch object
    grid_search_total = GridSearchCV(pl_gs_total, param_grid_total, cv=settings.train['CROSS_VALIDATE'],
                                     scoring=settings.train['MODEL_SCORING'],
                                     return_train_score=True)

    # Fit gridsearch object on to the data
    grid_search_total.fit(X_train, y_train)
    # Get the best estimator (based on best train score)
    print(f"The model with the best train score is:\n{grid_search_total.best_estimator_['clf']}")
    # Calculate the RMSE for best estimator
    print(f"This model has a train score (RMSE) of: {rmse_from_neg_mean_squared_error(grid_search_total.best_score_)}")
    print(
        f"This model has a test score (RMSE) of: {rmse_from_gridsearch_best_estimator(grid_search_total, X_test, y_test)}")

    # Save
    if save_all:
        # Save the best estimator of the gridsearch in a Pickle file
        suffix_datetime = datetime.strftime(datetime.now(), format='%Y%m%d%H%M')
        filename_output = f'best_model_{suffix_datetime}_{personal_note}'
        pickle.dump(grid_search_total.best_estimator_, open(f'{settings.DATAPATH}{filename_output}.pickle', 'wb'))

        # Save log of train step
        df_log = pd.DataFrame({"Model": [split_clf_and_params(grid_search_total.best_estimator_['clf'])[0]],
                              "Gridsearch_Params": [
                                  split_clf_and_params(grid_search_total.best_estimator_['clf'])[1]],
                              "Train_RMSE": [rmse_from_neg_mean_squared_error(grid_search_total.best_score_)],
                              "Test_RMSE": [rmse_from_gridsearch_best_estimator(grid_search_total, X_test, y_test)],
                              "Number_of_features": [len(X.columns)],
                              "Y_value": settings.train['Y_VALUE'],
                              "Input_filename": [filename_input],
                              "Output_filename": [filename_output],
                              })
        df_log.to_csv(settings.train['LOG_PATH'] + filename_output + '.csv', index=False,
                                     header=True)

    return grid_search_total