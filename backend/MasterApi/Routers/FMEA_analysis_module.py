from fastapi import APIRouter, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
import os
import logging
import pandas as pd
import json
from datetime import datetime
from computation.FMEA_module.OCELFailure import OCELFailureMode
from computation.FMEA_module.fmea_utils import get_fmea_insights
from computation.FMEA_module.OCELEnhancedFMEA import OCELEnhancedFMEA
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

fmea_router = APIRouter(prefix="/fmea-analysis", tags=["FMEA Analysis"])

# Create the directory when the server is initiated
if not os.path.exists("api_response"):
    os.makedirs("api_response")
    logger.info('Directory created successfully')

# Implement the root route
@fmea_router.get("/")
async def read_root(request: Request):
    return {"message": "Welcome to the FMEA Analysis Module"}

def convert_timestamps(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # Convert to ISO 8601 format
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

@fmea_router.get("/fmea-analysis")
def perform_fmea_analysis():
    try:
        if not os.path.exists("api_response/process_data.json"):
            raise HTTPException(status_code=400, detail="OCEL model file not found. Run Outlier Analysis first.")
        
        with open("api_response/process_data.json", "r") as f:
            ocel_data = json.load(f)
        
        if "ocel:events" not in ocel_data:
            logger.error("Invalid OCEL data structure")
            raise HTTPException(status_code=400, detail="Invalid OCEL data structure - missing events")
        
        analyzer = OCELEnhancedFMEA(ocel_data)
        fmea_results = analyzer.identify_failure_modes()
        
        fmea_results_path = os.path.join("api_response", "fmea_results.json")
        with open(fmea_results_path, "w") as f:
            json.dump(fmea_results, f, indent=4, default=convert_timestamps)
        
        logger.info(f"Analysis complete. Found {len(fmea_results)} failure modes")
        
        ai_insights = get_fmea_insights(fmea_results)
        ai_insights_path = os.path.join("api_response", "ai_insights.json")
        with open(ai_insights_path, "w") as f:
            json.dump(ai_insights, f, indent=4)
        
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

@fmea_router.get("/rpn_distribution")
async def display_rpn_distribution():
    try:
        fmea_results_path = os.path.join("api_response", "fmea_results.json")
        if not os.path.exists(fmea_results_path):
            raise HTTPException(status_code=404, detail="FMEA results file not found")

        with open(fmea_results_path, "r") as f:
            fmea_results = json.load(f)

        df = pd.DataFrame(fmea_results)
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=df['rpn'],
            nbinsx=20,
            name='RPN Distribution',
            marker_color='blue',
            opacity=0.7
        ))

        fig.add_vline(
            x=200,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical Threshold (RPN=200)",
            annotation_position="top right"
        )

        fig.add_vrect(
            x0=0, x1=100,
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Low Risk",
            annotation_position="bottom"
        )
        fig.add_vrect(
            x0=100, x1=200,
            fillcolor="yellow", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Medium Risk",
            annotation_position="bottom"
        )
        fig.add_vrect(
            x0=200, x1=1000,
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="High Risk",
            annotation_position="bottom"
        )

        fig.update_layout(
            title={
                'text': 'Risk Priority Number (RPN) Distribution',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='RPN Value',
            yaxis_title='Number of Failure Modes',
            showlegend=True,
            height=500,
            annotations=[
                dict(
                    x=50, y=1.05,
                    text="Low Risk Zone",
                    showarrow=False,
                    xref='x', yref='paper',
                    font=dict(color="green")
                ),
                dict(
                    x=150, y=1.05,
                    text="Medium Risk Zone",
                    showarrow=False,
                    xref='x', yref='paper',
                    font=dict(color="orange")
                ),
                dict(
                    x=250, y=1.05,
                    text="High Risk Zone",
                    showarrow=False,
                    xref='x', yref='paper',
                    font=dict(color="red")
                )
            ]
        )

        stats = {
            "average_rpn": df['rpn'].mean(),
            "median_rpn": df['rpn'].median(),
            "std_deviation": df['rpn'].std(),
            "90th_percentile": df['rpn'].quantile(0.9)
        }

        risk_zones = {
            'Low Risk (RPN ≤ 100)': len(df[df['rpn'] <= 100]),
            'Medium Risk (100 < RPN ≤ 200)': len(df[(df['rpn'] > 100) & (df['rpn'] <= 200)]),
            'High Risk (RPN > 200)': len(df[df['rpn'] > 200])
        }

        risk_fig = go.Figure(data=[
            go.Bar(
                x=list(risk_zones.keys()),
                y=list(risk_zones.values()),
                marker_color=['green', 'yellow', 'red']
            )
        ])
        risk_fig.update_layout(
            title="Distribution by Risk Zone",
            xaxis_title="Risk Zone",
            yaxis_title="Number of Failure Modes",
            height=400
        )

        return JSONResponse(content={
            "rpn_distribution_plot": json.loads(fig.to_json()),
            "risk_zone_plot": json.loads(risk_fig.to_json()),
            "statistics": stats,
            "risk_zones": risk_zones
        })

    except Exception as e:
        logger.error(f"Error in display_rpn_distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error displaying RPN distribution: {str(e)}")

