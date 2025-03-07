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
from FMEA_module.OCELFailure import OCELFailureMode

# import the fmea_utils.py file
from FMEA_module.fmea_utils import get_fmea_insights, display_rpn_distribution, display_fmea_analysis

# import the OCELDataManager class from the OCELDataManager.py file
from FMEA_module.OCELDataManager import OCELDataManager

# import the OCELEnhancedFMEA class from the OCELEnhancedFMEA.py file
from FMEA_module.OCELEnhancedFMEA import OCELEnhancedFMEA

from utils import get_azure_openai_client


# Importing the fastapi for the API
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)





# the pd option not reuqired here
# # Increase pandas display limit
# pd.set_option("styler.render.max_elements", 600000)














# Page execution
st.set_page_config(page_title="FMEA Analysis", layout="wide")
st.title("FMEA Analysis")


try:
    # First verify OCEL model file exists
    if not os.path.exists('ocpm_output/output_ocel.json'):
        st.error("OCEL model file (output_ocel.json) not found. Please run Outlier Analysis first to analyze event log")
        st.stop()

    # Load process data
    with open("ocpm_output/process_data.json", 'r') as f:
        ocel_data = json.load(f)

    # Validate OCEL data structure
    if 'ocel:events' not in ocel_data:
        st.error("Invalid OCEL data structure - missing events")
        logger.error("Invalid OCEL data structure")
        raise ValueError("Invalid OCEL data structure")

    # Initialize analyzer with validated data
    analyzer = OCELEnhancedFMEA(ocel_data)

    # Track analysis progress
    with st.spinner('Performing FMEA analysis...'):
        fmea_results = analyzer.identify_failure_modes()

        # Log analysis summary
        logger.info(f"Analysis complete. Found {len(fmea_results)} failure modes")
        logger.info(
            f"Using relationships from OCEL model with {len(analyzer.data_manager.object_relationships)} object types")

    # Get AI insights
    with st.spinner('Generating AI insights...'):
        ai_insights = get_fmea_insights(fmea_results)

    # Display AI insights in a collapsible section
    with st.expander("🤖 AI-Powered FMEA Insights", expanded=True):
        if ai_insights['findings']:
            st.markdown("### Key Findings")
            st.markdown(ai_insights['findings'])

        if ai_insights['insights']:
            st.markdown("### Critical Insights")
            st.markdown(ai_insights['insights'])

        if ai_insights['recommendations']:
            st.markdown("### Recommendations")
            st.markdown(ai_insights['recommendations'])


    # Display results with additional context
    st.success(f"Analysis complete - identified {len(fmea_results)} failure modes")
    display_fmea_analysis(fmea_results)

    if st.session_state.get('websocket_error'):
        st.warning("Previous session ended unexpectedly. Data has been refreshed.")
        st.session_state.websocket_error = False

except tornado.websocket.WebSocketClosedError:
    st.session_state.websocket_error = True
    st.error("Connection lost. Please refresh the page.")
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
    st.error(f"An unexpected error occurred: {str(e)}")

except FileNotFoundError as e:
    st.error(f"File error: {str(e)}")
    logger.error(f"File error in FMEA analysis: {str(e)}")
except ValueError as e:
    st.error(f"Data validation error: {str(e)}")
    logger.error(f"Validation error in FMEA analysis: {str(e)}")
except Exception as e:
    st.error(f"Error in FMEA analysis: {str(e)}")
    logger.error(f"FMEA analysis error: {str(e)}", exc_info=True)



