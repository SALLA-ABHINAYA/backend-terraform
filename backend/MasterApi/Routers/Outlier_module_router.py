## import all fasapi related libraries i want to make this a router forporcess-discoevery
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from fastapi import FastAPI
from fastapi import Request
from fastapi.templating import Jinja2Templates
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set
from tornado.websocket import WebSocketClosedError
from fastapi.responses import FileResponse
from fastapi import File, UploadFile


#  import pm4py
import pm4py
from fastapi import APIRouter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import tempfile
import os
from fastapi import HTTPException
import pm4py

import logging
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
import json
import plotly
from fastapi import Depends
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from computation.Outlier_module.Outlier_Analysis_utils import get_object_interactions, get_object_metrics, get_object_lifecycle_graph

import logging
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Dict, List, Any

from computation.Outlier_module.Outlier_Analysis_utils import get_object_interactions, get_object_metrics, get_object_lifecycle_graph


from computation.Outlier_module.Outlier_Analysis_utils import initialize_unfair_ocel_analyzer_with_failure_patterns
from computation.Outlier_module.Outlier_Analysis_utils import initialize_unfair_ocel_analyzer_with_time_analysis
from computation.Outlier_module.Outlier_Analysis_utils import initialize_unfair_ocel_analyzer_with_resource_analysis
from computation.Outlier_module.Outlier_Analysis_utils import initialize_unfair_ocel_analyzer_with_case_analysis_patterns

from backend.models.pydantic_models import  InteractionsResponse
from backend.models.pydantic_models import MetricModel
from backend.models.pydantic_models import LifecycleModel
from backend.models.pydantic_models import AIAnalysisResponse
from backend.models.pydantic_models import DataModel
from backend.models.pydantic_models import ResourceAnalysis
from backend.models.pydantic_models import FailureLogic
from backend.models.pydantic_models import CaseAnalysisDocument


from backend.utils.helpers import extract_json_schema
from backend.utils.helpers import convert_timestamps


out_router = APIRouter(prefix="/outlier-analysis", tags=["Outlier Analysis"])


# Create the directory when the server is initiated
if not os.path.exists("api_response"):
    os.makedirs("api_response")
    logger.info('Directory created successfully')

# implement the root route
@out_router.get("/")
async def read_root(request: Request):
    return {"message": "Welcome to the Outlier Analysis Module"}


## routing for ocpm ui
@out_router.get("/interactions",response_model=InteractionsResponse)
async def object_interactions():
    """API endpoint for object type interactions."""
    try:
        interactions_data = get_object_interactions()
        print(f"Type of interactions: {type(interactions_data)}")
        print(f"Content of interactions: {interactions_data}")
        
        # Convert to a list format which is always JSON serializable
        interactions_list = []
        
        # Handle the case where interactions_data is already wrapped in a dict
        
        data_to_process = interactions_data["interactions"]
            
        # Process the data into a list of objects
        if isinstance(data_to_process, dict):
            for key, value in data_to_process.items():
                # Convert any type of key to a serializable format
                if hasattr(key, '__iter__') and not isinstance(key, str):
                    # Create a dictionary with elements as separate fields
                    interaction = {"elements": list(key), "count": value}
                else:
                    interaction = {"key": str(key), "count": value}
                interactions_list.append(interaction)

        # print(f"Interactions list: {interactions_list}")
        print('starting')
        print(extract_json_schema(interactions_list))
        
        return {"interactions": interactions_list}
    except Exception as e:
        print(f"Error in object_interactions: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@out_router.get("/metrics",response_model=MetricModel)
def object_metrics():
            """API endpoint for object type metrics."""
            try:
                metrics = get_object_metrics()
                return {"metrics": metrics}  # Return as JSON
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))






@out_router.get("/lifecycle/{object_type}",response_model=LifecycleModel)
def object_lifecycle(object_type: str):
            """API endpoint for object lifecycle graph."""
            try:
                lifecycle_graph = get_object_lifecycle_graph(object_type)
                return {"lifecycle_graph": lifecycle_graph}  # Return as JSON
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


## OUTLIER ANALYSIS MODULE AI INSIGHTS

from computation.Outlier_module.IntegratedAPAAnalyzer import IntegratedAPAAnalyzer


@out_router.get('/run_ai_analysis',response_model=AIAnalysisResponse)
def run_ai_analysis():
    ocel_path = os.path.join("api_response", "process_data.json")
    if not os.path.exists(ocel_path):
        raise HTTPException(status_code=404, detail="process_data.json file not found")

    try:
        analyzer = IntegratedAPAAnalyzer()
        analyzer.load_ocel(ocel_path)

        stats = analyzer.stats

        # Default question for AI analysis
        default_question = "What are the main process patterns?"
        analysis_result = analyzer.analyze_with_ai(default_question)

        response = {
            "total_events": stats['general']['total_events'],
            "total_cases": stats['general']['total_cases'],
            "total_resources": stats['general']['total_resources'],
            "ai_analysis": analysis_result
        }

        return JSONResponse(response)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="OCEL file not found")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Value error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during AI analysis: {str(e)}")




@out_router.get('/get_visualization_data',response_model=DataModel)
def get_visualization_data():
    """Retrieve data for process visualizations."""
    ocel_path = os.path.join("api_response", "process_data.json")
    if not os.path.exists(ocel_path):
        raise HTTPException(status_code=404, detail="process_data.json file not found")

    try:
        analyzer = IntegratedAPAAnalyzer()
        analyzer.load_ocel(ocel_path)

        # Generate visualizations
        figures = analyzer.create_visualizations()
        print(f"Type of figures: {type(figures)}")
        
        # Convert Plotly figures to JSON using plotly's built-in serialization

        
        # Convert Plotly figures to JSON-serializable format
        visualization_data = {
            "activity_distribution": json.loads(plotly.io.to_json(figures["activity_distribution"])),
            "resource_distribution": json.loads(plotly.io.to_json(figures["resource_distribution"]))
        }

        return JSONResponse(content=visualization_data)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        return JSONResponse(content={"error": f"Error generating visualization data: {str(e)}"}, status_code=500)


#craete an end point router for _display_failure_patterns_markdown



import numpy as np

def convert_numpy_types(data):
    if isinstance(data, np.bool_):  # Convert NumPy boolean to Python boolean
        return bool(data)
    elif isinstance(data, np.integer):  # Convert NumPy integer to Python int
        return int(data)
    elif isinstance(data, np.floating):  # Convert NumPy float to Python float
        return float(data)
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]  # Recursively convert lists
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}  # Recursively convert dicts
    return data  # Return original if no conversion needed







@out_router.get('/display_failure_patterns',response_model=FailureLogic)
async def display_failure_patterns():
    """Display failure patterns."""
    try:
        markdown_logic = initialize_unfair_ocel_analyzer_with_failure_patterns()
        return (markdown_logic)
      #  return {"markdown": markdown_logic}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#create an endpoint for time_analysis of the failure patterns





@out_router.get('/resource_analysis',response_model=ResourceAnalysis)
def resource_analysis():
    """Resource analysis of failure patterns."""
    try:
        resource_analysis_data = initialize_unfair_ocel_analyzer_with_resource_analysis()
        return resource_analysis_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TimeDataTimeGap(BaseModel):
    time_data:Dict
    time_gaps:List[Dict]
    markdown:str

class TimeLogic(BaseModel):
    time_logic:TimeDataTimeGap

@out_router.get('/time_analysis',response_model=TimeLogic)
def time_analysis():
    """Time analysis of failure patterns."""
    try:
        time_analysis_data = initialize_unfair_ocel_analyzer_with_time_analysis()
        return (time_analysis_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






@out_router.get('/case_analysis_patterns',response_model=CaseAnalysisDocument)
async def case_analysis_patterns():
    """Case analysis of failure patterns."""
    try:
        case_analysis_data = initialize_unfair_ocel_analyzer_with_case_analysis_patterns()
        return case_analysis_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

