import sys 
from dataclasses import dataclass

import numpy as np
import pandas as pd 
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.utils import save_object
from src.logger import logging
import os


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformers(self):
        try:
            categorical_columns = ["Neighborhood"]
            numerical_columns = [
                "Gr Liv Area", "Bedroom AbvGr", "Year Built",
                "Year Remod/Add", "Yr Sold", "Total Bsmt SF"
            ]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ],
                remainder='passthrough'
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        


    def initiate_datatransformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            target_column = "SalePrice"
            x_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]
            x_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            preprocessor = self.get_transformers()

            logging.info("Fitting preprocessor on training data")
            x_train_arr = preprocessor.fit_transform(x_train)
            x_test_arr = preprocessor.transform(x_test)
            
            x_train_arr = x_train_arr.toarray()
            x_test_arr = x_test_arr.toarray()

            y_train_arr = np.array(y_train).reshape(-1, 1)
            y_test_arr = np.array(y_test).reshape(-1, 1)

            logging.info(f"X_train_arr shape: {x_train_arr.shape}, y_train_arr shape: {y_train_arr.shape}")
            logging.info(f"X_test_arr shape: {x_test_arr.shape}, y_test_arr shape: {y_test_arr.shape}")

            

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor
            )

            logging.info("Preprocessor saved successfully")

            return (x_train_arr,
                    y_train_arr,
                   x_test_arr,  
                   y_test_arr, 
                   self.data_transformation_config.preprocessor_file_path)

        except Exception as e:
            raise CustomException(e, sys)



    
   