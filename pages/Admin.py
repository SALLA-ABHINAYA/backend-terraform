import streamlit as st
import json
import os
from typing import Dict, List
import time

def load_ocel_json(file_path: str, data_folder_path: str = None) -> Dict:
    """
    Load OCEL JSON file with priority logic
    Args:
        file_path: Original file path
        data_folder_path: Data folder path
    """
    try:
        original_path = file_path
        data_folder_file = os.path.join(data_folder_path, os.path.basename(file_path)) if data_folder_path else None
        
        # Check if both files exist
        original_exists = os.path.exists(original_path)
        data_folder_exists = data_folder_file and os.path.exists(data_folder_file)
        
        if original_exists and data_folder_exists:
            # Compare modification times
            original_mtime = os.path.getmtime(original_path)
            data_folder_mtime = os.path.getmtime(data_folder_file)
            
            # If original is newer, delete data folder file and use original
            if original_mtime > data_folder_mtime:
                os.remove(data_folder_file)
                with open(original_path, 'r') as f:
                    return json.load(f)
        
        # Try loading from data folder first if it exists
        if data_folder_exists:
            with open(data_folder_file, 'r') as f:
                return json.load(f)
        
        # Fall back to original path
        if original_exists:
            with open(original_path, 'r') as f:
                return json.load(f)
                
        return {}
    except json.JSONDecodeError:
        st.error("Invalid JSON file")
        return {}
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return {}

def save_ocel_json(data: Dict, file_path: str, data_folder_path: str = None) -> None:
    """
    Save OCEL JSON file to both original location and data folder
    Args:
        data: Data to save
        file_path: Original file path
        data_folder_path: Optional data folder path to save copy
    """
    # # Save to original location
    # with open(file_path, 'w') as f:
    #     json.dump(data, f, indent=2)
    
    # If data folder specified, save there too
    if data_folder_path:
        # Create data folder if it doesn't exist
        os.makedirs(data_folder_path, exist_ok=True)
        data_file = os.path.join(data_folder_path, os.path.basename(file_path))
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)


def handle_add_threshold_activity(selected_type: str):
    """Handle adding a new activity"""
    if f"new_activity_{selected_type}" not in st.session_state:
        st.session_state[f"new_activity_{selected_type}"] = True
        st.session_state[f"activity_thresholds_{selected_type}"]["New Activity"] = {
            "max_duration_hours": 1.0,
            "max_gap_after_hours": 0.5
        }
        st.rerun()

def handle_remove_threshold_activity(selected_type: str, activity: str):
    """Handle removing an activity"""
    if activity in st.session_state[f"activity_thresholds_{selected_type}"]:
        del st.session_state[f"activity_thresholds_{selected_type}"][activity]
        st.rerun()


def edit_threshold_values(object_type_data: Dict, selected_type: str) -> Dict:
    """Edit threshold values for an object type"""
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            width: 100% !important;
        }
        .stSelectbox > div > div > select {
            width: 300px;
        }
        .stNumberInput > div > div > input {
            width: 100% !important;
        }
        div[data-testid="column"] {
            padding: 0px 5px;
            display: flex;
            align-items: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # General settings
    st.subheader("General Settings")
    col1, col2 = st.columns(2)
    with col1:
        total_duration = st.number_input(
            "Total Duration Hours",
            value=float(object_type_data.get("total_duration_hours", 0)),
            min_value=0.0,
            step=0.5,
            key=f"total_duration_{selected_type}"
        )
    with col2:
        default_gap = st.number_input(
            "Default Gap Hours",
            value=float(object_type_data.get("default_gap_hours", 0)),
            min_value=0.0,
            step=0.25,
            key=f"default_gap_{selected_type}"
        )

    # Activity thresholds
    st.write("---")
    st.subheader("Activity Thresholds")
    
    # Header row
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    with col1:
        st.write("**Activity**")
    with col2:
        st.write("**Max Duration Hours**")
    with col3:
        st.write("**Max Gap After Hours**")
    with col4:
        st.write("**Action**")

    # Initialize activity thresholds if not in session state
    if f"activity_thresholds_{selected_type}" not in st.session_state:
        st.session_state[f"activity_thresholds_{selected_type}"] = object_type_data.get("activity_thresholds", {})

    # Add new activity button
    if st.button("Add New Activity", key=f"add_activity_thresholds_{selected_type}"):
        # Find a unique name for the new activity
        base_name = "New Activity"
        new_name = base_name
        counter = 1
        while new_name in st.session_state[f"activity_thresholds_{selected_type}"]:
            new_name = f"{base_name} {counter}"
            counter += 1
        
        # Add new activity to dictionary with default values
        st.session_state[f"activity_thresholds_{selected_type}"][new_name] = {
            "max_duration_hours": 1.0,
            "max_gap_after_hours": 0.5
        }
        st.rerun()

    # Display and edit activities
    activities_to_remove = []
    for activity, thresholds in sorted(st.session_state[f"activity_thresholds_{selected_type}"].items()):
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            new_activity = st.text_input(
                "Activity",
                value=activity,
                key=f"activity_name_{selected_type}_{activity}"
            )
        
        with col2:
            max_duration = st.number_input(
                "Max Duration",
                value=float(thresholds.get("max_duration_hours", 0)),
                min_value=0.0,
                step=0.25,
                key=f"max_duration_{selected_type}_{activity}"
            )
        
        with col3:
            max_gap = st.number_input(
                "Max Gap",
                value=float(thresholds.get("max_gap_after_hours", 0)),
                min_value=0.0,
                step=0.25,
                key=f"max_gap_{selected_type}_{activity}"
            )
        
        with col4:
            st.markdown('<div style="height: 30px; display: flex; align-items: center;">', unsafe_allow_html=True)
            if st.button("Remove", key=f"remove_threshold_{selected_type}_{activity}"):
                 handle_remove_threshold_activity(selected_type, activity)

        # Update activity name if changed
        if new_activity != activity:
            st.session_state[f"activity_thresholds_{selected_type}"][new_activity] = st.session_state[f"activity_thresholds_{selected_type}"].pop(activity)
        
        # Update threshold values
        st.session_state[f"activity_thresholds_{selected_type}"][new_activity] = {
            "max_duration_hours": max_duration,
            "max_gap_after_hours": max_gap
        }

    # Remove marked activities
    for activity in activities_to_remove:
        del st.session_state[f"activity_thresholds_{selected_type}"][activity]

    return {
        "total_duration_hours": total_duration,
        "default_gap_hours": default_gap,
        "activity_thresholds": st.session_state[f"activity_thresholds_{selected_type}"]
    }

def handle_remove_item(selected_type: str, item_type: str, index: int):
    """Handle removing an item"""
    if index < len(st.session_state[f"{item_type}_{selected_type}"]):
        st.session_state[f"{item_type}_{selected_type}"].pop(index)
        st.rerun()

def edit_object_type(object_type: Dict, selected_type: str) -> Dict:
    """Edit individual object type properties"""
    st.markdown("""
        <style>
        .stSelectbox > div > div > select {
            width: 300px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for this object type if not exists
    if f"activities_{selected_type}" not in st.session_state:
        st.session_state[f"activities_{selected_type}"] = object_type.get("activities", [])
    if f"attributes_{selected_type}" not in st.session_state:
        st.session_state[f"attributes_{selected_type}"] = object_type.get("attributes", [])
    if f"relationships_{selected_type}" not in st.session_state:
        st.session_state[f"relationships_{selected_type}"] = object_type.get("relationships", [])
    
    name = st.text_input("Name", object_type.get("name", ""), key=f"name_{selected_type}")
    
    # Activities Section
    st.write("---")
    st.subheader("Activities")
    
    # Handle add activity
    if st.button("Add Activity", key=f"add_activity_{selected_type}"):
        st.session_state[f"activities_{selected_type}"].append("")
    
    # Display activities
    activities_to_remove = []
    for i, activity in enumerate(st.session_state[f"activities_{selected_type}"]):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.session_state[f"activities_{selected_type}"][i] = st.text_input(
                "Activity",
                activity,
                key=f"activity_{selected_type}_{i}",
                label_visibility="collapsed"
            )
        with col2:
            if st.button("Remove", key=f"remove_activity_{selected_type}_{i}"):
                handle_remove_item(selected_type, "activities", i)
    
    # Remove marked activities
    for idx in reversed(activities_to_remove):
        st.session_state[f"activities_{selected_type}"].pop(idx)
    
    # Attributes Section
    st.write("---")
    st.subheader("Attributes")
    
    # Handle add attribute
    if st.button("Add Attribute", key=f"add_attribute_{selected_type}"):
        st.session_state[f"attributes_{selected_type}"].append("")
    
    # Display attributes
    attributes_to_remove = []
    for i, attribute in enumerate(st.session_state[f"attributes_{selected_type}"]):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.session_state[f"attributes_{selected_type}"][i] = st.text_input(
                "Attribute",
                attribute,
                key=f"attribute_{selected_type}_{i}",
                label_visibility="collapsed"
            )
        with col2:
            if st.button("Remove", key=f"remove_attribute_{selected_type}_{i}"):
                handle_remove_item(selected_type, "attributes", i)
    
    # Remove marked attributes
    for idx in reversed(attributes_to_remove):
        st.session_state[f"attributes_{selected_type}"].pop(idx)
    
    # Relationships Section
    st.write("---")
    st.subheader("Relationships")
    
    # Handle add relationship
    if st.button("Add Relationship", key=f"add_relationship_{selected_type}"):
        st.session_state[f"relationships_{selected_type}"].append("")
    
    # Display relationships
    relationships_to_remove = []
    for i, relationship in enumerate(st.session_state[f"relationships_{selected_type}"]):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.session_state[f"relationships_{selected_type}"][i] = st.text_input(
                "Relationship",
                relationship,
                key=f"relationship_{selected_type}_{i}",
                label_visibility="collapsed"
            )
        with col2:
            if st.button("Remove", key=f"remove_relationship_{selected_type}_{i}"):
                handle_remove_item(selected_type, "relationships", i)
    
    # Remove marked relationships
    for idx in reversed(relationships_to_remove):
        st.session_state[f"relationships_{selected_type}"].pop(idx)
    
    return {
        "name": name,
        "activities": [a for a in st.session_state[f"activities_{selected_type}"] if a],
        "attributes": [a for a in st.session_state[f"attributes_{selected_type}"] if a],
        "relationships": [r for r in st.session_state[f"relationships_{selected_type}"] if r]
    }

def setup_ocel_editor_page():
    """Main OCEL editor page setup"""
    st.title("🔧 OCEL Editor")

    # # Reset session state on each load
    # for key in list(st.session_state.keys()):
    #     del st.session_state[key]
    
    # File path
    ocel_path = "ocpm_output/output_ocel.json"
    threshold_file = "ocpm_output/output_ocel_threshold.json"

    data_folder = "ocpm_data"
    

    if not os.path.exists(ocel_path) or not os.path.exists(threshold_file):
        st.error("Required files not found! Please run APA Analytics first to generate the necessary files.")
        st.info("Go to APA Analytics page and process your data to continue.")
        return
    
   
    
    if 'file_data' not in st.session_state:
        st.session_state.file_data = load_ocel_json(ocel_path, data_folder)
        
    if 'threshold_data' not in st.session_state:
        st.session_state.threshold_data = load_ocel_json(threshold_file, data_folder)
        
    # Track file modification times
    if 'last_file_check' not in st.session_state:
        st.session_state.last_file_check = {
            'ocel': os.path.getmtime(ocel_path) if os.path.exists(ocel_path) else 0,
            'threshold': os.path.getmtime(threshold_file) if os.path.exists(threshold_file) else 0
        }
    
    # Check if original files have been modified
    current_ocel_mtime = os.path.getmtime(ocel_path) if os.path.exists(ocel_path) else 0
    current_threshold_mtime = os.path.getmtime(threshold_file) if os.path.exists(threshold_file) else 0
    
    # Reload data if original files have changed
    if current_ocel_mtime > st.session_state.last_file_check['ocel']:
        st.session_state.file_data = load_ocel_json(ocel_path, data_folder)
        st.session_state.last_file_check['ocel'] = current_ocel_mtime
        
    if current_threshold_mtime > st.session_state.last_file_check['threshold']:
        st.session_state.threshold_data = load_ocel_json(threshold_file, data_folder)
        st.session_state.last_file_check['threshold'] = current_threshold_mtime
    
    # Create tabs
    editor_tab, json_tab = st.tabs(["OCEL Editor", "OCEL Threshold Editor"])
    
    with editor_tab:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Object Types")
        with col2:
            if st.button("Add New Object Type", use_container_width=True):
                new_name = "New_Object_Type"
                counter = 1
                while new_name in st.session_state.file_data:
                    new_name = f"New_Object_Type_{counter}"
                    counter += 1
                
                st.session_state.file_data[new_name] = {
                    "name": new_name,
                    "activities": [],
                    "attributes": [],
                    "relationships": []
                }
        
        # Object type selector
        object_types = list(st.session_state.file_data.keys())
        selected_type = st.selectbox(
            "Select Object Type",
            object_types,
            label_visibility="collapsed" if object_types else "visible"
        ) if object_types else None
        
        if selected_type:
            st.write("---")
            col1, col2 = st.columns([4, 1])
            with col1:
                st.subheader(f"Editing: {selected_type}")
            with col2:
                if st.button("Delete Object Type", type="secondary", use_container_width=True):
                    del st.session_state.file_data[selected_type]
                    st.rerun()
            
            # Edit object type properties
            updated_type = edit_object_type(st.session_state.file_data[selected_type], selected_type)
            st.session_state.file_data[selected_type] = updated_type
            
            st.write("---")
            if st.button("Save Changes", key=f"save_ocel", type="primary", use_container_width=True):
                save_ocel_json(st.session_state.file_data, ocel_path, data_folder)
                st.success("Changes saved successfully!")
    
    with json_tab:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Object Types")
        with col2:
            if st.button("Add New Object Type", key="add_new_threshold_type", use_container_width=True):
                new_name = "New_Object_Type"
                counter = 1
                while new_name in st.session_state.threshold_data:
                    new_name = f"New_Object_Type_{counter}"
                    counter += 1
                
                # Initialize new object type with default threshold values
                st.session_state.threshold_data[new_name] = {
                    "total_duration_hours": 0.0,
                    "default_gap_hours": 0.0,
                    "activity_thresholds": {}
                }
                st.rerun()
        object_types = list(st.session_state.threshold_data.keys())
        selected_type = st.selectbox(
            "Select Object Type",
            object_types,
            key="threshold_object_type_selector",
            label_visibility="visible"
        ) if object_types else None

        if selected_type:
            st.write("---")
            col1, col2 = st.columns([4, 1])
            with col1:
                st.subheader(f"Editing: {selected_type}")
            with col2:
                if st.button("Delete Object Type", key="delete_threshold_type", type="secondary", use_container_width=True):
                    del st.session_state.threshold_data[selected_type]
                    st.rerun()
        
        if selected_type:
            st.write("---")
            # Edit threshold values
            updated_data = edit_threshold_values(st.session_state.threshold_data[selected_type], selected_type)
            st.session_state.threshold_data[selected_type] = updated_data
            
            # Save changes button
            st.write("---")
            if st.button("Save Changes",key=f"save_threshold", type="primary", use_container_width=True):
                save_ocel_json(st.session_state.threshold_data, threshold_file, data_folder)
                st.success("Changes saved successfully!")

if __name__ == "__main__":
    st.set_page_config(
        page_title="OCEL Editor",
        page_icon="🔧",
        layout="wide"
    )
    setup_ocel_editor_page()