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

from Process_discovery_module.Process_discovery_utils import create_directories
from Process_discovery_module.Process_discovery_utils import save_uploaded_file
from Process_discovery_module.Process_discovery_utils import analyze_risks
from Process_discovery_module.Process_discovery_utils import visualize_risk_distribution
from Process_discovery_module.Process_discovery_utils import show_loader
from Process_discovery_module.Process_discovery_utils import hide_loader
from Process_discovery_module.Process_discovery_utils import process_mining_analysis

# Set up Graphviz path based on environment
azure_file_path = os.getenv("AZURE_FILE_PATH")
if azure_file_path:
    # We're in Azure
    os.environ["PATH"] = os.environ["PATH"] + ":" + azure_file_path
else:
    # We're in Windows local environment
    graphviz_path = "C:\\Program Files\\Graphviz\\bin"
    os.environ["PATH"] = os.environ["PATH"] + ";" + graphviz_path


# In 1_Process_Discovery.py

def main():
    st.set_page_config(page_title="IRMAI Process Discovery", page_icon="📊", layout="wide")

    # Header and Instructions
    st.title("📊 Process Discovery")
    st.info("Upload your Event Log to perform Process Discovery.")

    # Initialize session state
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'bpmn_graph' not in st.session_state:
        st.session_state.bpmn_graph = None
    if 'event_log' not in st.session_state:
        st.session_state.event_log = None

    # File Upload
    # uploaded_file = st.file_uploader("Upload Event Log (CSV)", type=['csv'])
    


    # API endpoint
    api_url = "http://127.0.0.1:8000/download_csv"

    if st.button("Fetch CSV File"):
        response = requests.get(api_url)

        if response.status_code == 200:
            st.write("✅ Successfully fetched the CSV file.")

            # ✅ Save CSV file immediately
            csv_file_path = "api_response/downloaded_event_log.csv"
            with open(csv_file_path, "wb") as f:
                f.write(response.content)

            # ✅ Read from saved file
            df = pd.read_csv(csv_file_path)
            
            st.write("📄 CSV File Received:")
            st.dataframe(df)

            # ✅ Store file path in session state for further processing
            st.session_state.uploaded_file = csv_file_path  

            # ✅ Processing
            show_loader()
            try:
                bpmn_graph, event_log = process_mining_analysis(csv_file_path)
                st.session_state.bpmn_graph = bpmn_graph
                st.session_state.event_log = event_log
                hide_loader()
                st.success('✅ Analysis completed successfully!')

            except Exception as e:
                hide_loader()
                st.error(f"❌ An error occurred: {str(e)}")
                st.write("🔹 Please ensure:")
                st.write("1. Your CSV file has the required columns: `case_id`, `activity`, `timestamp`")
                st.write("2. The data format is correct")
                st.write("3. Graphviz is installed and in your system PATH")

        else:
            st.error("❌ Failed to fetch CSV file from API.")



    if st.session_state.uploaded_file is not None:
        # Create tabs for different analysis sections
        tabs = st.tabs(["Process Discovery"])

        # Process Mining Tab
        with tabs[0]:
            st.subheader("Process Discovery")

            # Display process visualizations
            st.subheader("Process Visualizations")
            viz_tabs = st.tabs(["BPMN", "Petri Net", "Process Tree"])
            with viz_tabs[0]:
                st.image("output/fx_trade_bpmn.png")
            with viz_tabs[1]:
                st.image("output/fx_trade_petri_net.png")
            with viz_tabs[2]:
                st.image("output/fx_trade_process_tree.png")

import requests



main()