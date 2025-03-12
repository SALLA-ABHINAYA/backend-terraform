import os
import pandas as pd
import networkx as nx
import pydot
import json
from computation.Outlier_module.ocpm_analysis import OCPMAnalyzer  # Assuming you have this module
# from ocpm_visualizer import OCPMVisualizer  # Assuming this exists

EVENT_LOG_PATH = "api_response/data_event_log.csv"

def load_event_log():
    """Load event log from file."""
    if not os.path.exists(EVENT_LOG_PATH):
        raise FileNotFoundError("Event log not found. Run Process Discovery first.")

    try:
        df = pd.read_csv(EVENT_LOG_PATH, sep=";")
    except:
        df = pd.read_csv(EVENT_LOG_PATH)

    return df

def perform_analysis():
    """Perform OCPM analysis and return processed data."""
    df = load_event_log()
    analyzer = OCPMAnalyzer(df)

    ocel_path = analyzer.save_ocel()

    return {
        "ocel_path": ocel_path,
        "analyzer": analyzer
    }

def get_object_interactions():
    """Return object type interactions as a JSON response."""
    analysis = perform_analysis()
    interactions = analysis["analyzer"].analyze_object_interactions()
    return {"interactions": interactions}

def get_object_metrics():
    """Return object type metrics as a JSON response."""
    analysis = perform_analysis()
    metrics = analysis["analyzer"].calculate_object_metrics()
    return {"metrics": metrics}

def get_object_lifecycle_graph(object_type: str):
    """Generate object lifecycle graph and return as DOT format."""
    analysis = perform_analysis()
    lifecycle_graph = analysis["analyzer"].generate_object_lifecycle_graph(object_type)
    dot_graph = nx.nx_pydot.to_pydot(lifecycle_graph)
    return {"graph_dot": dot_graph.to_string()}


