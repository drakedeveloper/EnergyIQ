from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from model import load_model, get_forecast, get_anomalies, get_evaluation, get_realtime
from recommendations import get_recommendations


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting EnergyIQ — loading GGCN model...")
    load_model()
    yield
    print("👋 Shutting down")


app = FastAPI(title="EnergyIQ API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "EnergyIQ API v2 — GGCN Model ⚡", "status": "ok"}


@app.get("/api/realtime")
def realtime():
    return get_realtime()


@app.get("/api/forecast")
def forecast(hours: int = 48):
    return {"data": get_forecast(hours)}


@app.get("/api/anomalies")
def anomalies(limit: int = 50):
    data = get_anomalies(limit)
    return {"count": len(data), "data": data}


@app.get("/api/evaluation")
def evaluation():
    return get_evaluation()


@app.get("/api/recommendations")
def recommendations():
    data = get_recommendations()
    total_saving = sum(r['saving_mw'] for r in data)
    total_co2    = sum(r['co2_kg_day'] for r in data)
    return {"count": len(data), "total_saving_mw": round(total_saving, 1),
            "total_co2_kg_day": round(total_co2, 0), "data": data}


@app.get("/api/features")
def features():
    data = get_feature_importance()
    return {"count": len(data), "data": data}