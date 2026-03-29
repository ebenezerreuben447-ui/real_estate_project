import pandas as pd
import numpy as np

import sys 
from dataclasses import dataclass
 
import joblib

from src.exception import CustomException
from src.utils import save_object
from src.logger import logging
import os
from sklearn.model_selection import  GridSearchCV, cross_val_score
from src.utils import evaluate_model, train_and_evaluate, save_object

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifact','model.pkl')
   


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    

    def initiate_modeltrainer(self,x_train,y_train,x_text, y_test):
        try:
            logging.info("loading models with hyperparameter tunning") 
            models = {
                   'Linear Regression': {'model' :LinearRegression(),
                          'param' : {# LinearRegression doesn't have many hyperparameters to tune
                                   'copy_X' : [True,False], 
                                    'fit_intercept' : [True, False], 
                                    'n_jobs': [-1, 1, 2], 
                                    'positive': [True, False], 
                                    'tol': [1e-4, 1e-3, 1e-5]
                                    } 
                               },
                    'RandomForest': {'model':RandomForestRegressor(random_state=42),
                       'param': {'n_estimators': [50, 100],
                                  'max_depth': [None, 10, 20],
                                  'min_samples_split': [2, 5]}
                      },
            'XGBoost': {'model':XGBRegressor(objective='reg:squarederror', random_state=42),
              'param' : {'n_estimators' : [50, 100],
                         'max_depth': [3, 5, 7],
                         'learning_rate': [0.01, 0.1, 0.2]
                        }
                }}
            
            
            
            def get_best_model(results):
                """
                Select the best model based on highest R2 score.
                """
                best_model_name = None
                best_r2 = float('-inf')
                best_model_info = None

                for model_name, metrics in results.items():
                    if metrics['R2'] > best_r2:
                        best_r2 = metrics['R2']
                        best_model_name = model_name
                        best_model_info = metrics

                return best_model_name, best_model_info

            results = train_and_evaluate(models, x_train, y_train, x_text, y_test)
            best_model_name, best_info = get_best_model(results)
            
            best_model = best_info['Model']
            

            logging.info("saving best model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            predicted=best_model.predict(x_text)

            r2_square= r2_score(y_test,predicted)

            return r2_square
        except Exception as e:
            raise CustomException (e, sys)


 

  



        