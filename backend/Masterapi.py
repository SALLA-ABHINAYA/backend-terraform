# import the required modules
import os
# Importing the fastapi for the API
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# import logging
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set
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

app = FastAPI()

import json
import pandas as pd
from fastapi import File, UploadFile


app = FastAPI()

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

@app.get("/")
async def root():
    return {"message": "Welcome to the FMEA Analysis API"}

@app.post("/output_ocel_threshold")
async def post_output_ocel_threshold(data: Dict[str, Any]):
    try:
        with open("api_response/output_ocel_threshold.json", 'w') as f:
            json.dump(data, f)
        return JSONResponse(content={"message": "output_ocel_threshold.json saved successfully"})
    except Exception as e:
        logger.error(f"Error saving output_ocel_threshold.json: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving output_ocel_threshold.json: {str(e)}")

@app.post("/process_data")
async def post_process_data(data: Dict[str, Any]):
    try:
        with open("api_response/process_data.json", 'w') as f:
            json.dump(data, f)
        return JSONResponse(content={"message": "process_data.json saved successfully"})
    except Exception as e:
        logger.error(f"Error saving process_data.json: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving process_data.json: {str(e)}")

@app.post("/output_ocel")
async def post_output_ocel(data: Dict[str, Any]):
    try:
        with open("api_response/output_ocel.json", 'w') as f:
            json.dump(data, f)
        return JSONResponse(content={"message": "output_ocel.json saved successfully"})
    except Exception as e:
        logger.error(f"Error saving output_ocel.json: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving output_ocel.json: {str(e)}")

@app.post("/enhanced_prompt")
async def post_enhanced_prompt(data: Dict[str, Any]):
    try:
        with open("api_response/enhanced_prompt.json", 'w') as f:
            json.dump(data, f)
        return JSONResponse(content={"message": "enhanced_prompt.json saved successfully"})
    except Exception as e:
        logger.error(f"Error saving enhanced_prompt.json: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving enhanced_prompt.json: {str(e)}")

@app.post("/fmea_settings")
async def post_fmea_settings(data: Dict[str, Any]):
    try:
        with open("api_response/fmea_settings.json", 'w') as f:
            json.dump(data, f)
        return JSONResponse(content={"message": "fmea_settings.json saved successfully"})
    except Exception as e:
        logger.error(f"Error saving fmea_settings.json: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving fmea_settings.json: {str(e)}")

@app.post("/event_log")
async def post_event_log(file: UploadFile = File(...)):
    try:
        file_path = os.path.join("api_response", "event_log.csv")
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        return JSONResponse(content={"message": "event_log.csv saved successfully"})
    except Exception as e:
        logger.error(f"Error saving event_log.csv: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving event_log.csv: {str(e)}")



@app.get("/analyze")
async def analyze():
    try:
        # First verify OCEL model file exists
        if not os.path.exists('ocpm_output/output_ocel.json'):
            raise HTTPException(status_code=404, detail="OCEL model file (output_ocel.json) not found. Please run Outlier Analysis first to analyze event log")

        # Load process data
        with open("ocpm_output/process_data.json", 'r') as f:
            ocel_data = json.load(f)

        # Validate OCEL data structure
        if 'ocel:events' not in ocel_data:
            logger.error("Invalid OCEL data structure")
            raise HTTPException(status_code=400, detail="Invalid OCEL data structure - missing events")

        # Initialize analyzer with validated data
        analyzer = OCELEnhancedFMEA(ocel_data)

        # Perform FMEA analysis
        fmea_results = analyzer.identify_failure_modes()

        # Log analysis summary
        logger.info(f"Analysis complete. Found {len(fmea_results)} failure modes")
        logger.info(f"Using relationships from OCEL model with {len(analyzer.data_manager.object_relationships)} object types")

        # Get AI insights
        ai_insights = get_fmea_insights(fmea_results)

        # Prepare response
        response = {
            "fmea_results": fmea_results,
            "ai_insights": ai_insights
        }

        return JSONResponse(content=json.dumps(response, cls=CustomJSONEncoder))

    except WebSocketClosedError:
        raise HTTPException(status_code=500, detail="Connection lost. Please refresh the page.")
    except FileNotFoundError as e:
        logger.error(f"File error in FMEA analysis: {str(e)}")
        raise HTTPException(status_code=404, detail=f"File error: {str(e)}")
    except ValueError as e:
        logger.error(f"Validation error in FMEA analysis: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {str(e)}")
    except Exception as e:
        logger.error(f"FMEA analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


