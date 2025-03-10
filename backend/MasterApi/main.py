# import the required modules
import os
# Importing the fastapi for the API
from fastapi import HTTPException
from fastapi.responses import JSONResponse

# import logging
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set
from tornado.websocket import WebSocketClosedError  # Correct import
from fastapi.responses import FileResponse
from fastapi import File, UploadFile
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the directory when the server is initiated
if not os.path.exists("api_response"):
    os.makedirs("api_response")
    logger.info('Directory created successfully')


#Importing all the static files and CORS middleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Importing the FastAPI


from fastapi import File, UploadFile

from fastapi import FastAPI
from backend.MasterApi.routes import router

app = FastAPI()
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

@app.get("/page1")
async def page1(request: Request):
    return templates.TemplateResponse("page1.html", {"request": request})

#make an api end point for uploading event log






