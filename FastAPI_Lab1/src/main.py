from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data

app = FastAPI()

class HouseData(BaseModel):
    MedInc: float       # Median income in block
    HouseAge: float     # Median house age in block
    AveRooms: float     # Average number of rooms
    AveBedrms: float    # Average number of bedrooms
    Population: float   # Block population
    AveOccup: float     # Average house occupancy
    Latitude: float     # House block latitude
    Longitude: float    # House block longitude

class HouseResponse(BaseModel):
    predicted_price: float

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=HouseResponse)
async def predict_house_price(house_features: HouseData):
    try:
        features = [[
            house_features.MedInc,
            house_features.HouseAge,
            house_features.AveRooms,
            house_features.AveBedrms,
            house_features.Population,
            house_features.AveOccup,
            house_features.Latitude,
            house_features.Longitude
        ]]
        prediction = predict_data(features)
        return HouseResponse(predicted_price=float(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
