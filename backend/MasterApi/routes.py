from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io
import os
from computation.Process_discovery_module.Process_discovery_utils import create_directories
from computation.Process_discovery_module.Process_discovery_utils import save_uploaded_file
from computation.Process_discovery_module.Process_discovery_utils import analyze_risks
from computation.Process_discovery_module.Process_discovery_utils import visualize_risk_distribution
from computation.Process_discovery_module.Process_discovery_utils import show_loader
from computation.Process_discovery_module.Process_discovery_utils import hide_loader
from computation.Process_discovery_module.Process_discovery_utils import process_mining_analysis

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


router = APIRouter()

@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    # Save the file to the api_response folder
    save_path = os.path.join("api_response", "data_event_log.csv")
    with open(save_path, "wb") as f:
        f.write(content)

    # Example: Print CSV content (replace with processing logic)
    print(df.head())

    return {"message": f"Successfully uploaded {file.filename}", "data": df.head().to_dict()}


import tempfile
import os
import xml.dom.minidom
from fastapi import HTTPException, Response
import pm4py

@router.get("/calculate")
async def calculate():
    try:
        try:
            file_path = os.path.join("api_response", "data_event_log.csv")
            bpmn_graph, event_log = process_mining_analysis(file_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Data file not found")
        
        # Create a temporary file to write the BPMN XML
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".bpmn")
        temp_file.close()
        
        # Write the BPMN graph to the temporary file
        pm4py.write_bpmn(bpmn_graph, temp_file.name)
        
        # Read the XML content
        with open(temp_file.name, 'r') as f:
            bpmn_xml_raw = f.read()
        
        # Pretty print the XML
        dom = xml.dom.minidom.parseString(bpmn_xml_raw)
        pretty_xml = dom.toprettyxml(indent="  ")

        # Remove the temporary file
        os.unlink(temp_file.name)
        
        # Return either as JSON with the XML as a string
        return {
            "success": True,
            "status_code": 200,
            "message": "Successfully generated BPMN XML",
            "bpmn_xml": pretty_xml,
        }
        
        # Alternatively, return as a proper XML response:
        # return Response(
        #     content=pretty_xml,
        #     media_type="application/xml"
        # )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during process mining analysis: {str(e)}")


@router.get("/download_csv")
async def download_csv():
    file_path = os.path.join("data","fx_trade_log_small.csv")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    with open(file_path, "rb") as f:
        content = f.read()
    
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=data.csv"}
    )


## for the OUTLIER MODULE
from computation.Outlier_module.Outlier_Analysis_utils import get_object_interactions, get_object_metrics, get_object_lifecycle_graph



import logging
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

## routing for ocpm ui
@router.get("/interactions")
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

        print(f"Interactions list: {interactions_list}")
        
        return {"interactions": interactions_list}
    except Exception as e:
        print(f"Error in object_interactions: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
def object_metrics():
            """API endpoint for object type metrics."""
            try:
                metrics = get_object_metrics()
                return {"metrics": metrics}  # Return as JSON
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

@router.get("/lifecycle/{object_type}")
def object_lifecycle(object_type: str):
            """API endpoint for object lifecycle graph."""
            try:
                lifecycle_graph = get_object_lifecycle_graph(object_type)
                return {"lifecycle_graph": lifecycle_graph}  # Return as JSON
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


## OUTLIAER ANALYSIS MODULE AI INSIGHTS

from computation.Outlier_module.IntegratedAPAAnalyzer import IntegratedAPAAnalyzer
from fastapi.responses import JSONResponse

@router.get('/run_ai_analysis')
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



@router.get('/get_visualization_data')
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
        import json
        import plotly.utils
        
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

# How This Works:
# ✅ Extracts visualization data instead of rendering plots.
# ✅ Converts Plotly figures to JSON format (to_dict()).
# ✅ Frontend team can reconstruct plots using Plotly in JavaScript.

# Frontend Usage Example (JavaScript with Plotly.js):
# Once the frontend receives the JSON response, they can use it like this:

# javascript
# Copy
# Edit
# fetch("http://localhost:5000/get_visualization_data", {
#     method: "POST",
#     headers: { "Content-Type": "application/json" },
#     body: JSON.stringify({ ocel_path: "path/to/ocel.json" }),
# })
# .then(response => response.json())
# .then(data => {
#     Plotly.newPlot('activityChart', data.activity_distribution.data, data.activity_distribution.layout);
#     Plotly.newPlot('resourceChart', data.resource_distribution.data, data.resource_distribution.layout);
# })
# .catch(error => console.error("Error fetching visualization data:", error));






# FX ANALYTICS MODULE

## DIGITAL TWIN CODE WITH GAP ANALYSIS

def generate_trading_volume_chart():
    """Creates a currency trading volume visualization and returns it as Base64 image."""

    # Mock Data for Trading Volume
    dates = pd.date_range(start='2024-01-01', periods=6, freq='M')
    eur_volume = [4000, 3000, 2000, 2780, 1890, 2390]
    usd_volume = [2400, 1398, 9800, 3908, 4800, 3800]
    gbp_volume = [2400, 2210, 2290, 2000, 2181, 2500]

    # Convert to NumPy for serialization safety
    dates = dates.strftime('%Y-%m-%d').tolist()  # Convert to string format
    eur_volume = np.array(eur_volume).tolist()
    usd_volume = np.array(usd_volume).tolist()
    gbp_volume = np.array(gbp_volume).tolist()

    # Create Plot
    plt.figure(figsize=(8, 5))
    plt.plot(dates, eur_volume, marker='o', label="EUR")
    plt.plot(dates, usd_volume, marker='s', label="USD")
    plt.plot(dates, gbp_volume, marker='^', label="GBP")
    plt.xlabel("Date")
    plt.ylabel("Trading Volume")
    plt.title("Currency Trading Volume Over Time")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    # Convert plot to Base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    return base64_image

@router.get("/fx-analytics")
async def handle_graph_analytics():
    """Handle FX Trading Analytics (without visualization)"""
    
    # Mock Metrics Data
    trading_volume_chart = generate_trading_volume_chart()
    metrics = {
        "ml_model_accuracy": "94.5%",
        "anomalies_detected": 27,
        "pattern_confidence": "87.2%",
        "trading_volume": "$1.2B"
    }

    # Mock DataFrames (Converted to Dicts for JSON response)
    mock_volume = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=6, freq='M'),
        'EUR': [4000, 3000, 2000, 2780, 1890, 2390],
        'USD': [2400, 1398, 9800, 3908, 4800, 3800],
        'GBP': [2400, 2210, 2290, 2000, 2181, 2500]
    }).to_dict(orient="records")

    mock_patterns = pd.DataFrame({
        'time': ['09:00', '10:00', '11:00', '12:00', '13:00', '14:00'],
        'volume': [30, 45, 35, 50, 25, 40],
        'volatility': [65, 55, 85, 45, 60, 75]
    }).to_dict(orient="records")

    mock_anomalies = pd.DataFrame({
        'interval': list(range(1, 7)),
        'normal': [40, 30, 20, 27, 18, 23],
        'anomaly': [24, 13, 38, 15, 42, 19]
    }).to_dict(orient="records")

    # Return JSON Response
    return {
        "metrics": metrics,
        "trading_volume_chart": trading_volume_chart,
    }



# FMEA ANALYSIS MODULE
# pages/5_FMEA_Analysis.py
import os
import traceback
from collections import defaultdict

import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any, Set
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import tornado   # Required for Streamlit sharing

# import the OCELFailureMode class from the OCELFailure.py file
from computation.FMEA_module.OCELFailure import OCELFailureMode

# import the fmea_utils.py file
from computation.FMEA_module.fmea_utils import get_fmea_insights, display_rpn_distribution, display_fmea_analysis

# import the OCELDataManager class from the OCELDataManager.py file
from computation.FMEA_module.OCELDataManager import OCELDataManager

# import the OCELEnhancedFMEA class from the OCELEnhancedFMEA.py file
from computation.FMEA_module.OCELEnhancedFMEA import OCELEnhancedFMEA

from utils import get_azure_openai_client


@router.get("/fmea-analysis")
def perform_fmea_analysis():
    try:
        # Verify OCEL model file exists
        if not os.path.exists("api_response/output_ocel.json"):
            raise HTTPException(status_code=400, detail="OCEL model file not found. Run Outlier Analysis first.")
        
        # Load process data
        with open("api_response/process_data.json", "r") as f:
            ocel_data = json.load(f)
        
        # Validate OCEL data structure
        if "ocel:events" not in ocel_data:
            logger.error("Invalid OCEL data structure")
            raise HTTPException(status_code=400, detail="Invalid OCEL data structure - missing events")
        
        # Initialize analyzer with validated data
        analyzer = OCELEnhancedFMEA(ocel_data)
        
        # Perform FMEA analysis
        fmea_results = analyzer.identify_failure_modes()
        logger.info(f"Analysis complete. Found {len(fmea_results)} failure modes")
        
        # Get AI insights
        ai_insights = get_fmea_insights(fmea_results)
        
        return JSONResponse(content={
            "failure_modes": len(fmea_results),
            "findings": ai_insights.get("findings", "No findings available"),
            "insights": ai_insights.get("insights", "No insights available"),
            "recommendations": ai_insights.get("recommendations", "No recommendations available"),
        })
    
    except FileNotFoundError as e:
        logger.error(f"File error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File error: {str(e)}")
    except ValueError as e:
        logger.error(f"Data validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


