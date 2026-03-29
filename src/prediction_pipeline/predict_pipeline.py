import numpy as np
import pandas as pd
import joblib
import sys
from src.exception import CustomException
import os
from src.utils import load_object

class HouseData:
    def __init__(self, GrLivArea, BedroomAbvGr, Neighborhood, YearBuilt,
                 YearRemodAdd, YrSold, TotalBsmtSF):
        self.GrLivArea = GrLivArea
        self.BedroomAbvGr = BedroomAbvGr
        self.Neighborhood = Neighborhood
        self.YearBuilt = YearBuilt
        self.YearRemodAdd = YearRemodAdd
        self.YrSold = YrSold
        self.TotalBsmtSF = TotalBsmtSF

    def to_dataframe(self):
        return pd.DataFrame({
            "Gr Liv Area": [self.GrLivArea],
            "Bedroom AbvGr": [self.BedroomAbvGr],
            "Neighborhood": [self.Neighborhood],
            "Year Built": [self.YearBuilt],
            "Year Remod/Add": [self.YearRemodAdd],
            "Yr Sold": [self.YrSold],
            "Total Bsmt SF": [self.TotalBsmtSF]
        })
    



class PredictPipeline:
    def __init__(self):
        self.model_path = "artifact/model.pkl"
        self.preprocessor_path = "artifact/preprocessor.pkl"

        self.model = load_object(file_path= self.model_path)
        self.preprocessor = load_object(file_path= self.preprocessor_path)

    def predict(self, data: pd.DataFrame):
        try:
            transformed = self.preprocessor.transform(data)
            transformed = transformed.toarray()
            preds = self.model.predict(transformed)
            return preds

        except Exception as e:
            raise CustomException(e, sys)



