import numpy as np
from sklearn.metrics import mean_squared_error

def drop_nan_from_specific_columns (df, columns_to_check):
    """
    Drops all rows with nan values in specific columns in a dataframe
    """
    return df.dropna(axis=0,
                    how='any',
                    thresh=None,
                    subset=columns_to_check,
                    inplace=False)

def split_clf_and_params(best_estimator_clf):
    """
    Takes best estimator[clf] and outputs a list with clf and parameters
    """
    clf_and_params = str(best_estimator_clf)
    clf_and_params = clf_and_params.replace(")", "")
    clf_and_params_split = clf_and_params.split("(")
    return clf_and_params_split

def rmse_from_neg_mean_squared_error(neg_mean_squared_error):
    """
    Calculates RMSE from the neq mean squared error
    """
    rmse = np.sqrt(-(neg_mean_squared_error))
    return rmse

# functie maken om op basis van de grid search best estimator, het beste RMSE model te selecteren
def rmse_from_gridsearch_best_estimator(grid_search, X_test, y_test):
    """
    Calculates RMSE from the grid search best estimator
    """
    rmse = np.sqrt(mean_squared_error(y_test, grid_search.best_estimator_.predict(X_test)))
    return rmse