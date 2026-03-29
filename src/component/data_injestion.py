import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException 
from src.component.database_connection import load_data_from_sql
from src.component.data_transformation import DataTransformation
from src.component.data_transformation import DataTransformationConfig
from src.component.model_trainer import ModelTrainer
from src.component.model_trainer import ModelTrainerConfig
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass  
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    raw_data_path: str=os.path.join('artifact','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # ✅ Get dataframe from SQL
            df = load_data_from_sql()
            logging.info("Data loaded from SQL into Data Ingestion component")

            # selecting the important features
            select_colunms = ["Gr Liv Area", # Above ground living area
                "Bedroom AbvGr", # Numbers of the bedrooms above the grade
                "Neighborhood", # Location
                "Year Built",
                "Year Remod/Add", #Year remodeled
                "Yr Sold",
                "Total Bsmt SF", #basement size
                "SalePrice" ]# targert
            df = df[select_colunms]
            logging.info("selecting important features completed")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) 

            logging.info("train test split initiated")
            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info("ingestion is completed")
            
            return (self.ingestion_config.train_data_path, 
                    self.ingestion_config.test_data_path
                    )
        except Exception as e:
            raise CustomException  (e,sys) 


if __name__=="__main__":
    df = load_data_from_sql()
    
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    x_train_arr,y_train_arr,x_test_arr,y_test_arr,_=data_transformation.initiate_datatransformation(train_data_path, test_data_path)

    model_train=ModelTrainer()
    print(model_train.initiate_modeltrainer( x_train_arr,y_train_arr,x_test_arr,y_test_arr))
    
    
