from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io
import os
from Process_discovery_module.Process_discovery_utils import create_directories
from Process_discovery_module.Process_discovery_utils import save_uploaded_file
from Process_discovery_module.Process_discovery_utils import analyze_risks
from Process_discovery_module.Process_discovery_utils import visualize_risk_distribution
from Process_discovery_module.Process_discovery_utils import show_loader
from Process_discovery_module.Process_discovery_utils import hide_loader
from Process_discovery_module.Process_discovery_utils import process_mining_analysis

#  import pm4py
import pm4py

import tempfile
import os
from fastapi import HTTPException
import pm4py


router = APIRouter()

@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    # Save the file to the api_response folder
    save_path = os.path.join("api_response", "data.csv")
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
        bpmn_graph, event_log = process_mining_analysis(os.path.join("api_response", "data.csv"))
        
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