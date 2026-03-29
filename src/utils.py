import os
import sys

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import  GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            joblib.dump(obj, file_obj)
            logging.info("saving completed")
    except Exception as e:
        raise CustomException (e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return joblib.load(file_obj)
            logging.info("loading completed")
    except Exception as e:
        raise CustomException (e,sys)


def evaluate_model(model, x_test, y_test):
    """
    Evaluate a trained model using MSE and R2 Score.
    """
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2


def train_and_evaluate(models, x_train, y_train, x_test, y_test):
                results = {}

                for model_name, model_param in models.items():
                    print(f"Training {model_name}...")

                    grid = GridSearchCV(
                        estimator=model_param['model'],
                        param_grid=model_param['param'],
                        cv=5,
                        n_jobs=-1,  # Faster if many models
                        verbose=1
                    )
                    grid.fit(x_train, y_train)

                    best_model = grid.best_estimator_
                    mse, r2 = evaluate_model(best_model, x_test, y_test)

                
                    results[model_name] = {
                        'Model': best_model,
                        'Best Parameters': grid.best_params_,
                        'MSE': mse,
                        'R2': r2
                    }


                    print(f"{model_name} training completed ✓")

                    return results

                




def print_results(results):
        """
        Nicely format and print model performance.
        """
        print("\n====== Model Evaluation Results ======\n")
        for model_name, metrics in results.items():
            print(f"Model: {model_name}")
            print(f"  Best Parameters: {metrics['Best Parameters']}")
            print(f"  MSE: {metrics['MSE']:.3f}")
            print(f"  R2 Score: {metrics['R2']:.3f}\n")