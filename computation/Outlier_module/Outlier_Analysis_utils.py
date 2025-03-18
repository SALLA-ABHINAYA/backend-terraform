import os
import pandas as pd
import networkx as nx
import pydot
import json
from computation.Outlier_module.ocpm_analysis import OCPMAnalyzer  # Assuming you have this module
# from ocpm_visualizer import OCPMVisualizer  # Assuming this exists
from typing import Dict
import logging
import numpy as np

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

def _count_resource_interactions(self) -> int:
        """Count number of resource interactions in process"""
        interactions = 0
        for case_id, case_events in self.case_events.items():
            case_resources = set()
            for event_id in case_events:
                event = next((e for e in self.ocel_data['ocel:events'] if e['ocel:id'] == event_id), None)
                if event:
                    resource = event.get('ocel:attributes', {}).get('resource', 'Unknown')
                    case_resources.add(resource)
            if len(case_resources) > 1:
                interactions += len(case_resources) - 1
        return interactions

def _count_major_violations(self, duration_data: Dict) -> int:
        """Count number of major timing violations"""
        major_violations = 0
        for metrics in duration_data.values():
            if hasattr(metrics, 'details'):
                violations = metrics.details.get('outlier_events', {}).get('timing_gap', [])
                major_violations += sum(1 for v in violations
                                        if v['details'].get('time_gap_minutes', 0) >
                                        v['details'].get('threshold_minutes', 0) * 2)
        return major_violations

def _identify_bottlenecks(self, duration_data: Dict) -> int:
        """Identify number of process bottlenecks"""
        return sum(1 for metrics in duration_data.values()
                   if hasattr(metrics, 'is_outlier') and metrics.is_outlier)

def _count_object_interactions(self, case_data: Dict) -> int:
        """Count cases with multiple object interactions"""
        multi_object_cases = 0
        for metrics in case_data.values():
            if hasattr(metrics, 'details'):
                events = metrics.details.get('events', {})
                if len(set(events.get('object_types', []))) > 1:
                    multi_object_cases += 1
        return multi_object_cases

def _count_case_variants(self, case_data: Dict) -> int:
        """Count unique case variants based on activity sequences"""
        variants = set()
        for metrics in case_data.values():
            if hasattr(metrics, 'details'):
                activity_sequence = tuple(metrics.details.get('events', {}).get('activity_sequence', []))
                if activity_sequence:
                    variants.add(activity_sequence)
        return len(variants)

def get_explanation(self, tab_type: str, metrics: Dict) -> Dict:
        """Generate AI explanation for OCEL analysis results"""
        logger.info(f"Generating OCEL explanation for: {tab_type}")

        try:
            context = self._build_analysis_context(tab_type, metrics)

            prompt = f"""
            Analyze this Object-Centric Process Mining data for {context['title']}:

            Process Context:
            {self._format_process_context()}

            Analysis Metrics:
            {self._format_metrics(context['metrics'])}

            Specific Patterns:
            {self._format_patterns(context['patterns'])}

            Object Type Interactions:
            {self._format_object_interactions()}

            Provide detailed analysis focusing on:
            1. Object-centric interactions and their impact on process performance
            2. Process conformance, deviations, and their root causes
            3. Resource and time utilization patterns affecting multiple object types
            4. Multi-perspective performance insights across object lifecycles
            5. Specific recommendations for process improvement

            Format response as:
            1. Key findings about object interactions and process patterns
            2. Critical insights about performance and conformance
            3. Specific, actionable recommendations for process improvement
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are an expert in object-centric process mining analysis, focusing on multi-object interactions and process patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            return self._format_ai_response(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {
                "error": "Unable to generate explanation",
                "details": str(e)
            }

from computation.Outlier_module.Unfair_Advanced_Process_Logs_Analytics import UnfairOCELAnalyzer 

OCEL_PATH = "api_response/process_data.json"
# Create a function to initialize the UnfairOCELAnalyzer

def convert_numpy_types(data):
    if isinstance(data, np.bool_):  # Convert NumPy boolean to Python boolean
        return bool(data)
    elif isinstance(data, np.integer):  # Convert NumPy integer to Python int
        return int(data)
    elif isinstance(data, np.floating):  # Convert NumPy float to Python float
        return float(data)
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]  # Recursively convert lists
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}  # Recursively convert dicts
    return data  # Return original if no conversion needed

def convert_timestamps(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # Convert to ISO 8601 format
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()  # Convert to string
        return super().default(obj)



def initialize_unfair_ocel_analyzer_with_failure_patterns():
    """Initialize UnfairOCELAnalyzer with OCEL data and process failure patterns."""

    unfair_analyzer = UnfairOCELAnalyzer(OCEL_PATH)
    markdown_failure_patterns = unfair_analyzer._display_failure_patterns_markdown()
    failure_patterns = unfair_analyzer.process_failure_patterns()
    resouce_outlier_data=unfair_analyzer.get_resource_outlier_data()
    resouce_outlier_data=convert_numpy_types(resouce_outlier_data)
    markdown_resource_outlier=unfair_analyzer._display_resource_outlier_markdown()
    
    #return (resouce_outlier_data)
    return {"failure_logic": failure_patterns,"failure_markdown": markdown_failure_patterns}


def initialize_unfair_ocel_analyzer_with_resource_analysis():
    unfair_analyzer = UnfairOCELAnalyzer(OCEL_PATH)
    resouce_outlier_data=unfair_analyzer.get_resource_outlier_data()
    resouce_outlier_data=convert_numpy_types(resouce_outlier_data)
    markdown_resource_outlier=unfair_analyzer._display_resource_outlier_markdown()
    return {"resource_outlier_data":resouce_outlier_data,"resource_outlier_markdown":markdown_resource_outlier}

def initialize_unfair_ocel_analyzer_with_time_analysis():
    """Initialize UnfairOCELAnalyzer with OCEL data and process time analysis."""
    try:
        unfair_analyzer = UnfairOCELAnalyzer(OCEL_PATH)
        time_data_and_time_gaps = unfair_analyzer.analyze_time_outliers()

        return time_data_and_time_gaps
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Or handle the error as needed

def initialize_unfair_ocel_analyzer_with_case_analysis_patterns():
    """Initialize UnfairOCELAnalyzer with OCEL data and process case analysis."""
    try:
        unfair_analyzer = UnfairOCELAnalyzer(OCEL_PATH)
        case_analysis = unfair_analyzer.prepare_case_analysis()
        markdown=unfair_analyzer.display_case_outlier_markdown()

        case_analysis= convert_numpy_types(case_analysis)
        return {"case_analysis":case_analysis,"case_analysis_markdown":markdown}
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Or handle the error as needed
































# def _display_failure_patterns_logic(self):
#     """Display the logic content for Failure Patterns tab"""
#     # Validate failures data exists
#         failures_data = self.outliers.get('failures', {})
#         if not failures_data:
#             st.warning("No failure pattern data available")
#         else:
#             col1, col2 = st.columns([2, 1])

#             with col1:
#                 failure_counts = {
#                     'Sequence Violations': len(failures_data.get('sequence_violations', [])),
#                     'Incomplete Cases': len(failures_data.get('incomplete_cases', [])),
#                     'Long Running': len(failures_data.get('long_running', [])),
#                     'Resource Switches': len(failures_data.get('resource_switches', [])),
#                     'Rework Activities': len(failures_data.get('rework_activities', []))
#                 }

#                 if any(failure_counts.values()):
#                     fig = go.Figure(data=[
#                         go.Bar(
#                             x=list(failure_counts.keys()),
#                             y=list(failure_counts.values()),
#                             text=list(failure_counts.values()),
#                             textposition='auto',
#                         )
#                     ])

#                     fig.update_layout(
#                         title='Process Failure Patterns Distribution',
#                         xaxis_title='Failure Pattern Type',
#                         yaxis_title='Count',
#                         showlegend=False
#                     )
#                     st.plotly_chart(fig)

#                     # Show failure details in expandable sections
#                     for pattern, failures in failures_data.items():
#                         if failures:
#                             with st.expander(f"{pattern.replace('_', ' ').title()} ({len(failures)})"):
#                                 try:
#                                     failure_df = pd.DataFrame(failures)
#                                     st.dataframe(failure_df)
#                                 except Exception as e:
#                                     logger.error(f"Error creating failure DataFrame: {str(e)}")
#                                     st.error(f"Error displaying failure details: {str(e)}")

#             with col2:
#                 failure_metrics = {
#                     'sequence_violations': len(failures_data.get('sequence_violations', [])),
#                     'incomplete_cases': len(failures_data.get('incomplete_cases', [])),
#                     'long_running': len(failures_data.get('long_running', []))
#                 }

#                 st.markdown("### Understanding Failure Patterns")
#                 try:
#                     explanation = self.get_explanation('failure', failure_metrics)
#                     st.markdown(explanation.get('summary', 'No summary available'))

#                     if explanation.get('insights'):
#                         st.markdown("#### Key Insights")
#                         for insight in explanation['insights']:
#                             st.markdown(f"• {insight}")

#                     if explanation.get('recommendations'):
#                         st.markdown("#### Recommendations")
#                         for rec in explanation['recommendations']:
#                             st.markdown(f"• {rec}")
#                 except Exception as e:
#                     logger.error(f"Error getting explanation: {str(e)}")
#                     st.error("Unable to generate insights at this time")


# def _display_resource_outlier_logic(self):
#     """Display the logic content for Resource Outlier tab"""
#     # Validate resource data exists
#         resource_data = self.outliers.get('resource_load', {})
#         if not resource_data:
#             st.warning("No resource analysis data available")
#         else:
#             col1, col2 = st.columns([2, 1])

#             with col1:
#                 # Create resource workload visualization
#                 workload_data = []
#                 resource_details = {}

#                 for resource, metrics in resource_data.items():
#                     if isinstance(metrics, (dict, object)):  # Validate metrics object
#                         try:
#                             workload_data.append({
#                                 'Resource': resource,
#                                 'Z-Score': getattr(metrics, 'z_score', 0),
#                                 'Is Outlier': getattr(metrics, 'is_outlier', False),
#                                 'Total Events': metrics.details['metrics'].get('total_events', 0) if hasattr(
#                                     metrics, 'details') else 0,
#                                 'Cases': len(metrics.details['events'].get('cases', [])) if hasattr(metrics,
#                                                                                                     'details') else 0,
#                                 'Activities': len(metrics.details['events'].get('activities', [])) if hasattr(
#                                     metrics, 'details') else 0
#                             })

#                             # Store details for traceability
#                             if hasattr(metrics, 'details'):
#                                 resource_details[resource] = {
#                                     'metrics': metrics.details.get('metrics', {}),
#                                     'events': metrics.details.get('events', {}),
#                                     'patterns': metrics.details.get('outlier_patterns', {})
#                                 }
#                         except Exception as e:
#                             logger.error(f"Error processing resource metrics: {str(e)}")

#                 if workload_data:
#                     # Create resource plot
#                     try:
#                         fig = px.scatter(
#                             pd.DataFrame(workload_data),
#                             x='Resource',
#                             y='Z-Score',
#                             size='Total Events',
#                             color='Is Outlier',
#                             title='Resource Workload Distribution',
#                             custom_data=['Cases', 'Activities']
#                         )

#                         fig.update_traces(
#                             hovertemplate=(
#                                     "<b>%{x}</b><br>" +
#                                     "Z-Score: %{y:.2f}<br>" +
#                                     "Total Events: %{marker.size}<br>" +
#                                     "Cases: %{customdata[0]}<br>" +
#                                     "Activities: %{customdata[1]}<br>" +
#                                     "<extra></extra>"
#                             )
#                         )
#                         st.plotly_chart(fig, use_container_width=True)

#                         # Show resource details
#                         if resource_details:
#                             selected_resource = st.selectbox(
#                                 "Select resource for details",
#                                 options=list(resource_details.keys())
#                             )
#                             if selected_resource:
#                                 self._display_resource_details(selected_resource,
#                                                                resource_details[selected_resource])
#                     except Exception as e:
#                         logger.error(f"Error creating resource visualization: {str(e)}")
#                         st.error("Error displaying resource visualization")

#             with col2:
#                 workload_metrics = {
#                     'market_maker_b': len(self.resource_events.get('Market Maker B', [])),
#                     'client_desk_d': len(self.resource_events.get('Client Desk D', [])),
#                     'options_desk_a': len(self.resource_events.get('Options Desk A', []))
#                 }
#                 st.markdown("### Understanding Resource Distribution")
#                 try:
#                     explanation = self.get_explanation('resource', workload_metrics)
#                     st.markdown(explanation.get('summary', 'No summary available'))

#                     if explanation.get('insights'):
#                         st.markdown("#### Key Insights")
#                         for insight in explanation['insights']:
#                             st.markdown(f"• {insight}")

#                     if explanation.get('recommendations'):
#                         st.markdown("#### Recommendations")
#                         for rec in explanation['recommendations']:
#                             st.markdown(f"• {rec}")
#                 except Exception as e:
#                     logger.error(f"Error getting resource explanation: {str(e)}")
#                     st.error("Unable to generate resource insights")

    # def display_time_outlier_tab(self):
    #     """Display the Time Outlier tab content."""
    #     logger.debug("Processing Time Analysis tab")

    #     # Validate duration data exists
    #     duration_data = self.outliers.get('duration', {})
    #     if not duration_data:
    #         st.warning("No time analysis data available")
    #         return

    #     col1, col2 = st.columns([2, 1])

    #     with col1:
    #         try:
    #             time_data=self._analyze_time_data(duration_data) # return dictionay with pd data frame contains
    #             time_gaps=self._analyze_timing_gaps(duration_data) # return pandas dataframe
    #         except Exception as e:
    #             logger.error(f"Error displaying time analysis: {str(e)}")
    #             st.error("Error displaying time analysis visualizations")

    #     with col2:
    #         timing_metrics = {
    #             'avg_duration': f"{np.mean([len(events) for events in self.case_events.values()]):.2f}",
    #             'outlier_count': len(
    #                 [m for m in duration_data.values() if getattr(m, 'is_outlier', False)]),
    #             'typical_duration': "2-4 hours"
    #         }
    #         st.markdown("### Understanding Time Patterns")

    #         try:
    #             explanation = self.get_explanation('time', timing_metrics)
    #             st.write(explanation.get('summary', 'No summary available'))

    #             if explanation.get('insights'):
    #                 st.markdown("#### Key Insights")
    #                 for insight in explanation['insights']:
    #                     st.markdown(f"• {insight}")

    #             if explanation.get('recommendations'):
    #                 st.markdown("#### Recommendations")
    #                 for rec in explanation['recommendations']:
    #                     st.markdown(f"• {rec}")
    #         except Exception as e:
    #             logger.error(f"Error getting time explanation: {str(e)}")
    #             st.error("Unable to generate time insights")
