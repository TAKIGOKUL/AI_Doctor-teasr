import os
from fastapi import FastAPI;
from fastapi.middleware.cors import CORSMiddleware
from flask_cors import CORS

# from routes.login_routes import login_api_router
from routes.disease_model import disease_api_router
from routes.medbot import medbot_api_router
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origins=['*']
)

# app.include_router(login_api_router)
app.include_router(disease_api_router)
app.include_router(medbot_api_router)