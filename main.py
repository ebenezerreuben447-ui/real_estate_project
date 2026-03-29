from fastapi import FastAPI
from pydantic import BaseModel
from src.prediction_pipeline.predict_pipeline import PredictPipeline, HouseData

app = FastAPI(title="House Price Prediction API")


class HouseRequest(BaseModel):
    GrLivArea: float
    BedroomAbvGr: int
    Neighborhood: str
    YearBuilt: int
    YearRemodAdd: int
    YrSold: int
    TotalBsmtSF: float


@app.get("/")
def home():
    return {"message": "Welcome to House Price Prediction API"}


@app.post("/predict")
def predict_price(data: HouseRequest):
    try:
        # ✅ Use HouseData (not HouseRequest)
        input_data = HouseData(
            GrLivArea=data.GrLivArea,
            BedroomAbvGr=data.BedroomAbvGr,
            Neighborhood=data.Neighborhood,
            YearBuilt=data.YearBuilt,
            YearRemodAdd=data.YearRemodAdd,
            YrSold=data.YrSold,
            TotalBsmtSF=data.TotalBsmtSF
        )

        df = input_data.to_dataframe()

        pipeline = PredictPipeline()
        result = pipeline.predict(df)

        return {"Predicted House Price": float(result[0])}

    except Exception as e:
        return {"error": str(e)}