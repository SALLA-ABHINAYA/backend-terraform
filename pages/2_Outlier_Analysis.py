import shutil
import traceback
import streamlit as st
from Outlier_module.Unfair_Advanced_Process_Logs_Analytics import UnfairOCELAnalyzer
from Outlier_module.ocpm_analysis import create_ocpm_ui
import os
from pathlib import Path
from openai import OpenAI
import pandas as pd
import plotly.express as px
import json
from neo4j import GraphDatabase

from utils import get_azure_openai_client

# import the IntegratedAPAAnalyzer class from the IntegratedAPAAnalyzer.py file
from Outlier_module.IntegratedAPAAnalyzer import IntegratedAPAAnalyzer
# 


def run_unfairness_analysis():
    """Run unfairness analysis"""
    st.subheader("Outlier Analysis")
    ocel_path = st.session_state.get('ocel_path') or find_ocel_file()

    if not ocel_path or not os.path.exists(ocel_path):
        st.warning("⚠️ Please process data in the Process Analysis tab first.")
        return
    try:
        analyzer = UnfairOCELAnalyzer(ocel_path)
        analyzer.display_enhanced_analysis()
    except Exception as e:
        st.error(f"Error in unfairness analysis: {str(e)}")
        st.error(f"Detailed error:\n{traceback.format_exc()}")


def find_ocel_file():
    """Find the OCEL file in expected locations"""
    possible_paths = [
        "ocpm_output/process_data.json",
        "ocpm_data/process_data.json",
        os.path.join("ocpm_output", "process_data.json"),
        os.path.join("ocpm_data", "process_data.json")
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def run_ai_analysis():
    """Run AI-powered analysis"""
    st.subheader("🤖 AI-Powered Process Analysis")

    ocel_path = st.session_state.get('ocel_path')
    if not ocel_path or not os.path.exists(ocel_path):
        st.warning("⚠️ Please process data in the Process Analysis tab first.")
        return

    try:
        analyzer = IntegratedAPAAnalyzer()
        analyzer.load_ocel(ocel_path)

        # Save statistics in session state
        st.session_state['stats'] = analyzer.stats

        # Display Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Events", analyzer.stats['general']['total_events'])
        with col2:
            st.metric("Total Cases", analyzer.stats['general']['total_cases'])
        with col3:
            st.metric("Total Resources", analyzer.stats['general']['total_resources'])

        # AI Analysis Section
        st.subheader("Ask Questions About Your Process")

        st.write("**Example Questions:**")
        st.write("- What are the most common activity sequences?")
        st.write("- How are different resources utilized across cases?")
        st.write("- What is the typical process flow for trades?")

        question = st.text_input(
            "Ask a question about the process:",
            placeholder="e.g., What are the main process patterns?"
        )

        if question:
            with st.spinner("Analyzing..."):
                analysis = analyzer.analyze_with_ai(question)
                st.write(analysis)

        # Show visualizations
        st.subheader("Process Visualizations")
        figures = analyzer.create_visualizations()

        viz_tabs = st.tabs(["Activities", "Resources"])
        with viz_tabs[0]:
            st.plotly_chart(figures['activity_distribution'], use_container_width=True)
        with viz_tabs[1]:
            st.plotly_chart(figures['resource_distribution'], use_container_width=True)

    except Exception as e:
        st.error(f"Error in AI analysis: {str(e)}")


def setup_apa_page():
    """Set up the integrated APA Analytics page"""
    st.title("📊 IRMAI APA Analytics")

    st.info("APA provides comprehensive process analysis including AI-powered insights.")

    # Create tabs for different analyses
    main_tabs = st.tabs(["Process Analysis", "Outlier Analysis", "AI Insights"])

    with main_tabs[0]:
        create_ocpm_ui()

        if 'ocpm_df' in st.session_state:
            ocel_path = Path("ocpm_output/process_data.json")
            st.session_state['ocel_path'] = str(ocel_path)

    with main_tabs[1]:
        run_unfairness_analysis()

    with main_tabs[2]:
        run_ai_analysis()


if __name__ == "__main__":
    st.set_page_config(
        page_title="IRMAI APA Analysis",
        page_icon="📊",
        layout="wide"
    )
    setup_apa_page()