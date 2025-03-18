import shutil
import streamlit as st
import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
import os, json
import plotly.graph_objects as go
# from pages.Process_Discovery_module.process_mining_enhanced import FXProcessMining
from Process_discovery_module.process_mining_enhanced import FXProcessMining
from Process_discovery_module.risk_analysis import ProcessRiskAnalyzer, EnhancedFMEA
from io import StringIO
import pandas as pd
import requests
import logging
from fastapi.responses import HTMLResponse

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'output', 'staging']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Create directories
    os.makedirs("ocpm_data", exist_ok=True)
    os.makedirs("ocpm_output", exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to data directory"""
    create_directories()
    file_path = os.path.join('data', uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def analyze_risks(event_log, bpmn_graph):
    """Perform risk analysis on process model"""
    try:
        # Initialize ProcessRiskAnalyzer
        risk_analyzer = ProcessRiskAnalyzer(event_log, bpmn_graph)
        risk_analyzer.analyze_bpmn_graph()

        # Initialize EnhancedFMEA
        fmea = EnhancedFMEA(
            failure_modes=risk_analyzer.failure_modes,
            activity_stats=risk_analyzer.activity_stats
        )

        # Get risk assessment results
        risk_assessment = fmea.assess_risk()

        # Calculate process metrics
        process_metrics = {
            'total_activities': len(risk_analyzer.activity_stats),
            'high_risk_activities': len([r for r in risk_assessment if r['rpn'] > 200]),
            'medium_risk_activities': len([r for r in risk_assessment if 100 < r['rpn'] <= 200]),
            'low_risk_activities': len([r for r in risk_assessment if r['rpn'] <= 100])
        }

        return risk_assessment, process_metrics

    except Exception as e:
        logger.error(f"Error in risk assessment: {str(e)}")
        raise

def visualize_risk_distribution(risk_assessment_results):
    """Create visualization of risk distribution"""
    activities = [r['failure_mode'] for r in risk_assessment_results]
    rpn_values = [r['rpn'] for r in risk_assessment_results]
    severities = [r['severity'] for r in risk_assessment_results]
    likelihoods = [r['likelihood'] for r in risk_assessment_results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=severities,
        y=likelihoods,
        mode='markers',
        marker=dict(
            size=[r['rpn'] * 5 for r in risk_assessment_results],
            color=rpn_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="RPN")
        ),
        text=activities,
        hovertemplate="<b>Activity:</b> %{text}<br>" +
                      "<b>Severity:</b> %{x:.2f}<br>" +
                      "<b>Likelihood:</b> %{y:.2f}<br>" +
                      "<b>RPN:</b> %{marker.color:.2f}<br>"
    ))

    fig.update_layout(
        title="Risk Distribution Matrix",
        xaxis_title="Severity",
        yaxis_title="Likelihood",
        showlegend=False
    )

    return fig

def show_loader():
    """Show full-screen loader including sidebar"""
    loader_html = """
    <style>
        /* Full-screen overlay */
        .overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.3);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
        }
        
        /* Loader animation */
        .loader {
            border: 8px solid #f3f3f3;
            border-top: 8px solid #3498db;
            border-radius: 50%;
            width: 80px;
            height: 80px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Ensure Streamlit sidebar is also covered */
        [data-testid="stSidebar"] {
            z-index: 9999 !important;
        }
    </style>

    <div class="overlay" id="loader">
        <div class="loader"></div>
    </div>

    <script>
        function hideLoader() {
            var loader = document.getElementById("loader");
            if (loader) {
                loader.style.display = "none";
            }
        }
    </script>
    """
    return HTMLResponse(content=loader_html)

def hide_loader():
    """Hide loader"""
    return HTMLResponse(content="""
        <style>
            .overlay { display: none; }
        </style>
    """)

def process_mining_analysis(csv_path):
    """Perform process mining analysis, save CSV for later use, and send it via API"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs("ocpm_output", exist_ok=True)

        # Copy uploaded file to output directory for use by Outlier Analysis
        # output_csv_path = os.path.join("ocpm_output", "event_log.csv")
        # shutil.copy2(csv_path, output_csv_path)

        # Send the file via API POST request
        # try:  
        #     api_url = "http://127.0.0.1:8000/event_log"
        #     with open(csv_path, 'rb') as f:
        #         response = requests.post(api_url, files={'file': f})
    
        #     if response.status_code == 200:
        #         st.success("File successfully sent to the API.")
        #     else:
        #         st.error(f"Failed to send file to the API. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while sending the file to the API: {str(e)}")

    try:
        # Initialize FX Process Mining
        fx_miner = FXProcessMining(csv_path)
        fx_miner.preprocess_data()
        fx_miner.discover_process()

        # Get process model components
        process_tree = fx_miner.process_tree
        petri_net = fx_miner.process_model
        initial_marking = fx_miner.initial_marking
        final_marking = fx_miner.final_marking
        event_log = fx_miner.event_log

        # Generate visualizations
        pn_gviz = pn_visualizer.apply(petri_net, initial_marking, final_marking)
        pn_visualizer.save(pn_gviz, "output/fx_trade_petri_net.png")
        pn_visualizer.save(pn_gviz, "api_response/fx_trade_petri_net.png")

        pt_gviz = pt_visualizer.apply(process_tree)
        pt_visualizer.save(pt_gviz, "output/fx_trade_process_tree.png")
        pt_visualizer.save(pt_gviz, "api_response/fx_trade_process_tree.png")

        bpmn_graph = pm4py.convert_to_bpmn(process_tree)
        bpmn_gviz = bpmn_visualizer.apply(bpmn_graph)
        bpmn_visualizer.save(bpmn_gviz, "output/fx_trade_bpmn.png")
        bpmn_visualizer.save(bpmn_gviz, "api_response/fx_trade_bpmn.png")

        return bpmn_graph, event_log

    except Exception as e:
        logger.error(f"Error in process analytics analysis: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
