import os
import shutil

import streamlit as st
import json
import pandas as pd

import plotly.graph_objects as go
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import logging
import traceback
from datetime import datetime
from typing import Dict, List
from neo4j import GraphDatabase

from utils import get_azure_openai_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)








def delete_all_files(directory="ocpm_output"):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Deletes subdirectories as well
        print(f"All files and subdirectories in '{directory}' have been deleted.")
    else:
        print(f"Directory '{directory}' does not exist.")

def main():
    st.set_page_config(
        page_title="IRMAI Process Analytics",
        page_icon="👋",
        layout="wide"
    )

    delete_all_files()


if __name__ == "__main__":
    main()

