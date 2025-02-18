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
        # Form paths
        data_folder_file = os.path.join(data_folder_path, os.path.basename(file_path)) if data_folder_path else None
        
        # If data folder file exists, use it without any checks
        if data_folder_file and os.path.exists(data_folder_file):
            with open(data_folder_file, 'r') as f:
                return json.load(f)
        
        # If no data folder file, try original path
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
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
        try:
            # Create data folder if it doesn't exist
            os.makedirs(data_folder_path, exist_ok=True)
            data_file = os.path.join(data_folder_path, os.path.basename(file_path))
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2)

            timestamp_file = os.path.join(data_folder_path, "last_update.txt")
            current_time = str(time.time())
            
            if os.path.exists(timestamp_file):
                # Read existing timestamps
                with open(timestamp_file, 'r') as f:
                    lines = f.readlines()
                    timestamps = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            timestamps[key] = value
                
                # Update current timestamp
                timestamps['CURRENT_UPD_TIME'] = current_time
                
            else:
                # Create new timestamp file
                timestamps = {
                    'CURRENT_UPD_TIME': current_time,
                    'PREV_UPD_TIME': ''
                }
            
            # Write updated timestamps
            with open(timestamp_file, 'w') as f:
                for key, value in timestamps.items():
                    f.write(f"{key}:{value}\n")
                
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")


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

def edit_fmea_settings(fmea_data: Dict, selected_type: str) -> Dict:
    """Edit FMEA settings for an object type"""
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
        }
        /* Compact buttons */
        .stButton button {
            padding: 0.25rem 1rem;
            min-height: 38px;
        }
        .element-container:has(button) {
            margin-top: 25px;  /* Adjust this value to match your input box position */
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state for visibility and criticality if not exists
    if f"visibility_{selected_type}" not in st.session_state:
        st.session_state[f"visibility_{selected_type}"] = fmea_data.get("object_visibility", {}).get(selected_type, -1)

    if f"criticality_{selected_type}" not in st.session_state:
        st.session_state[f"criticality_{selected_type}"] = fmea_data.get("object_criticality", {}).get(selected_type, 1)

    if f"temporal_deps_{selected_type}" not in st.session_state:
        st.session_state[f"temporal_deps_{selected_type}"] = fmea_data.get("temporal_dependencies", {}).get(selected_type, {
            "sequences": [],
            "constraints": {},
            "business_logic": [],
            "validation_requirements": []
        })

    if f"critical_activities_{selected_type}" not in st.session_state:
        st.session_state[f"critical_activities_{selected_type}"] = fmea_data.get("critical_activities", {}).get(selected_type, [])

    # Visibility and Criticality Settings
    st.subheader("Object Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        visibility = st.number_input(
            "Visibility Level (-3 to -1)",
            value=int(st.session_state[f"visibility_{selected_type}"]),
            min_value=-3,
            max_value=-1,
            step=1,
            key=f"visibility_input_{selected_type}"
        )
    
    with col2:
        criticality = st.number_input(
            "Criticality Level (1 to 5)",
            value=int(st.session_state[f"criticality_{selected_type}"]),
            min_value=1,
            max_value=5,
            step=1,
            key=f"criticality_input_{selected_type}"
        )

    # Temporal Dependencies Section
    st.write("---")
    st.subheader("Temporal Dependencies")

    # Initialize temporal dependencies in session state
    if f"temporal_deps_{selected_type}" not in st.session_state:
        st.session_state[f"temporal_deps_{selected_type}"] = fmea_data.get("temporal_dependencies", {}).get(selected_type, {
            "sequences": [],
            "constraints": {},
            "business_logic": [],
            "validation_requirements": []
        })

    # Sequences
    st.write("**Activity Sequences**")
    if st.button("Add New Sequence", key=f"add_sequence_{selected_type}"):
        st.session_state[f"temporal_deps_{selected_type}"]["sequences"].append([])
        st.rerun()

    for seq_idx, sequence in enumerate(st.session_state[f"temporal_deps_{selected_type}"]["sequences"]):
        st.write(f"Sequence {seq_idx + 1}")
        
        # Add activity to sequence
        if st.button("Add Activity", key=f"add_activity_seq_{selected_type}_{seq_idx}"):
            sequence.append("")
            st.rerun()

        for act_idx, activity in enumerate(sequence):
            col1, col2 = st.columns([4, 1])
            with col1:
                sequence[act_idx] = st.text_input(
                    f"Activity {act_idx + 1}",
                    activity,
                    key=f"sequence_activity_{selected_type}_{seq_idx}_{act_idx}"
                )
            with col2:
                if st.button("Remove", key=f"remove_activity_{selected_type}_{seq_idx}_{act_idx}"):
                    sequence.pop(act_idx)
                    st.rerun()

        if st.button("Remove Sequence", key=f"remove_sequence_{selected_type}_{seq_idx}"):
            st.session_state[f"temporal_deps_{selected_type}"]["sequences"].pop(seq_idx)
            st.rerun()

    # Constraints
    st.write("---")
    st.write("**Constraints**")
    if st.button("Add Constraint", key=f"add_constraint_{selected_type}"):
        st.session_state[f"temporal_deps_{selected_type}"]["constraints"]["New Constraint"] = ""
        st.rerun()

    constraints_to_remove = []
    for activity, constraint in st.session_state[f"temporal_deps_{selected_type}"]["constraints"].items():
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            new_activity = st.text_input(
                "Activity",
                activity,
                key=f"constraint_activity_{selected_type}_{activity}"
            )
        with col2:
            new_constraint = st.text_input(
                "Constraint",
                constraint,
                key=f"constraint_value_{selected_type}_{activity}"
            )
        with col3:
            if st.button("Remove", key=f"remove_constraint_{selected_type}_{activity}"):
                constraints_to_remove.append(activity)

    # Update constraints
    updated_constraints = {}
    for activity, constraint in st.session_state[f"temporal_deps_{selected_type}"]["constraints"].items():
        if activity not in constraints_to_remove:
            new_activity = st.session_state[f"constraint_activity_{selected_type}_{activity}"]
            new_constraint = st.session_state[f"constraint_value_{selected_type}_{activity}"]
            updated_constraints[new_activity] = new_constraint
    st.session_state[f"temporal_deps_{selected_type}"]["constraints"] = updated_constraints

    # Business Logic
    st.write("---")
    st.write("**Business Logic**")
    if st.button("Add Business Logic", key=f"add_logic_{selected_type}"):
        st.session_state[f"temporal_deps_{selected_type}"]["business_logic"].append("")
        st.rerun()

    for idx, logic in enumerate(st.session_state[f"temporal_deps_{selected_type}"]["business_logic"]):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.session_state[f"temporal_deps_{selected_type}"]["business_logic"][idx] = st.text_input(
                "Logic",
                logic,
                key=f"business_logic_{selected_type}_{idx}"
            )
        with col2:
            if st.button("Remove", key=f"remove_logic_{selected_type}_{idx}"):
                st.session_state[f"temporal_deps_{selected_type}"]["business_logic"].pop(idx)
                st.rerun()

    # Validation Requirements
    st.write("---")
    st.write("**Validation Requirements**")
    if st.button("Add Validation Requirement", key=f"add_validation_{selected_type}"):
        st.session_state[f"temporal_deps_{selected_type}"]["validation_requirements"].append("")
        st.rerun()

    for idx, req in enumerate(st.session_state[f"temporal_deps_{selected_type}"]["validation_requirements"]):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.session_state[f"temporal_deps_{selected_type}"]["validation_requirements"][idx] = st.text_input(
                "Requirement",
                req,
                key=f"validation_req_{selected_type}_{idx}"
            )
        with col2:
            if st.button("Remove", key=f"remove_validation_{selected_type}_{idx}"):
                st.session_state[f"temporal_deps_{selected_type}"]["validation_requirements"].pop(idx)
                st.rerun()

    # Critical Activities
    st.write("---")
    st.subheader("Critical Activities")
    if st.button("Add Critical Activity", key=f"add_critical_{selected_type}"):
        st.session_state[f"critical_activities_{selected_type}"].append("")
        st.rerun()

    for idx, activity in enumerate(st.session_state[f"critical_activities_{selected_type}"]):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.session_state[f"critical_activities_{selected_type}"][idx] = st.text_input(
                "Activity",
                activity,
                key=f"critical_activity_{selected_type}_{idx}"
            )
        with col2:
            if st.button("Remove", key=f"remove_critical_{selected_type}_{idx}"):
                st.session_state[f"critical_activities_{selected_type}"].pop(idx)
                st.rerun()

    # Return updated data
    return {
        "object_visibility": {**fmea_data.get("object_visibility", {}), **{selected_type: visibility}},
        "object_criticality": {**fmea_data.get("object_criticality", {}), **{selected_type: criticality}},
        "temporal_dependencies": {**fmea_data.get("temporal_dependencies", {}), **{selected_type: st.session_state[f"temporal_deps_{selected_type}"]}},
        "critical_activities": {**fmea_data.get("critical_activities", {}), **{selected_type: st.session_state[f"critical_activities_{selected_type}"]}},
        "regulatory_keywords": fmea_data.get("regulatory_keywords", [])
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
    fmea_file = "ocpm_output/fmea_settings.json"

    data_folder = "ocpm_data"
    

    if not os.path.exists(ocel_path) or not os.path.exists(threshold_file):
        st.error("Required files not found! Please run APA Analytics first to generate the necessary files.")
        st.info("Go to APA Analytics page and process your data to continue.")
        return
    
    # Initialize base data
    if 'file_data' not in st.session_state:
        st.session_state.file_data = load_ocel_json(ocel_path, data_folder)
        
    if 'threshold_data' not in st.session_state:
        st.session_state.threshold_data = load_ocel_json(threshold_file, data_folder)
    
    if 'fmea_data' not in st.session_state:  # Add FMEA data initialization
        st.session_state.fmea_data = load_ocel_json(fmea_file, data_folder)
        
    # Track file modification times only for output files
    if 'last_file_check' not in st.session_state:
        st.session_state.last_file_check = {
            'ocel': os.path.getmtime(ocel_path) if os.path.exists(ocel_path) else 0,
            'threshold': os.path.getmtime(threshold_file) if os.path.exists(threshold_file) else 0,
            'fmea': os.path.getmtime(fmea_file) if os.path.exists(fmea_file) else 0
        }
    
    data_ocel_file = os.path.join(data_folder, os.path.basename(ocel_path))
    data_threshold_file = os.path.join(data_folder, os.path.basename(threshold_file))
    
    # Only check output files if data folder files don't exist
    if not os.path.exists(data_ocel_file):
        current_ocel_mtime = os.path.getmtime(ocel_path) if os.path.exists(ocel_path) else 0
        if current_ocel_mtime > st.session_state.last_file_check['ocel']:
            st.session_state.file_data = load_ocel_json(ocel_path, data_folder)
            st.session_state.last_file_check['ocel'] = current_ocel_mtime
            
    if not os.path.exists(data_threshold_file):
        current_threshold_mtime = os.path.getmtime(threshold_file) if os.path.exists(threshold_file) else 0
        if current_threshold_mtime > st.session_state.last_file_check['threshold']:
            st.session_state.threshold_data = load_ocel_json(threshold_file, data_folder)
            st.session_state.last_file_check['threshold'] = current_threshold_mtime
    
    # Create tabs
    editor_tab, json_tab, fmea_tab = st.tabs(["OCEL Editor", "OCEL Threshold Editor", "FMEA Settings"])
    
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

    with fmea_tab:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Object Types")
        with col2:
            if st.button("Add New Object Type", key="add_new_fmea_type", use_container_width=True):
                new_name = "New_Object_Type"
                counter = 1
                while new_name in st.session_state.fmea_data.get("object_visibility", {}):
                    new_name = f"New_Object_Type_{counter}"
                    counter += 1
                
                # Initialize new object type with default FMEA values
                if "object_visibility" not in st.session_state.fmea_data:
                    st.session_state.fmea_data["object_visibility"] = {}
                if "object_criticality" not in st.session_state.fmea_data:
                    st.session_state.fmea_data["object_criticality"] = {}
                if "temporal_dependencies" not in st.session_state.fmea_data:
                    st.session_state.fmea_data["temporal_dependencies"] = {}
                if "critical_activities" not in st.session_state.fmea_data:
                    st.session_state.fmea_data["critical_activities"] = {}
                
                st.session_state.fmea_data["object_visibility"][new_name] = -1
                st.session_state.fmea_data["object_criticality"][new_name] = 1
                st.session_state.fmea_data["temporal_dependencies"][new_name] = {
                    "sequences": [],
                    "constraints": {},
                    "business_logic": [],
                    "validation_requirements": []
                }
                st.session_state.fmea_data["critical_activities"][new_name] = []
                st.rerun()

        # Object type selector
        object_types = list(st.session_state.fmea_data.get("object_visibility", {}).keys())
        selected_type = st.selectbox(
            "Select Object Type",
            object_types,
            key="fmea_object_type_selector",
            label_visibility="visible"
        ) if object_types else None

        if selected_type:
            st.write("---")
            col1, col2 = st.columns([4, 1])
            with col1:
                st.subheader(f"Editing: {selected_type}")
            with col2:
                if st.button("Delete Object Type", key="delete_fmea_type", type="secondary", use_container_width=True):
                    # Remove object type from all FMEA data structures
                    st.session_state.fmea_data["object_visibility"].pop(selected_type, None)
                    st.session_state.fmea_data["object_criticality"].pop(selected_type, None)
                    st.session_state.fmea_data["temporal_dependencies"].pop(selected_type, None)
                    st.session_state.fmea_data["critical_activities"].pop(selected_type, None)
                    st.rerun()

            # Edit FMEA settings
            updated_data = edit_fmea_settings(st.session_state.fmea_data, selected_type)
            st.session_state.fmea_data = updated_data

            # Regulatory Keywords Section (Global settings)
            st.write("---")
            st.subheader("Regulatory Keywords (Global)")
            
            if "regulatory_keywords" not in st.session_state:
                st.session_state["regulatory_keywords"] = st.session_state.fmea_data.get("regulatory_keywords", [])

            if st.button("Add Regulatory Keyword", key="add_regulatory_keyword"):
                st.session_state["regulatory_keywords"].append("")
                st.rerun()

            for idx, keyword in enumerate(st.session_state["regulatory_keywords"]):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.session_state["regulatory_keywords"][idx] = st.text_input(
                        "Keyword",
                        keyword,
                        key=f"regulatory_keyword_{idx}"
                    )
                with col2:
                    if st.button("Remove", key=f"remove_keyword_{idx}"):
                        st.session_state["regulatory_keywords"].pop(idx)
                        st.rerun()

            # Update regulatory keywords in main data
            st.session_state.fmea_data["regulatory_keywords"] = [
                kw for kw in st.session_state["regulatory_keywords"] if kw
            ]

            # Save changes button
            st.write("---")
            if st.button("Save Changes", key="save_fmea", type="primary", use_container_width=True):
                save_ocel_json(st.session_state.fmea_data, fmea_file, data_folder)
                st.success("Changes saved successfully!")

if __name__ == "__main__":
    st.set_page_config(
        page_title="OCEL Editor",
        page_icon="🔧",
        layout="wide"
    )
    setup_ocel_editor_page()