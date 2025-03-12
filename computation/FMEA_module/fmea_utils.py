from typing import Dict, List, Any, Set
from utils import get_azure_openai_client
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import traceback
# create logger object
import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_fmea_insights(fmea_results: List[Dict]) -> Dict[str, str]:
    """Generate AI insights for FMEA analysis"""
    client = get_azure_openai_client()

    # Create analysis context
    analysis_context = {
        'total_failures': len(fmea_results),
        'high_risk': len([r for r in fmea_results if r['rpn'] > 200]),
        'medium_risk': len([r for r in fmea_results if 100 < r['rpn'] <= 200]),
        'object_types': list(set(r['object_type'] for r in fmea_results)),
        'top_activities': list(set(r['activity'] for r in fmea_results
                                   if r['rpn'] > 150))[:5]
    }

    prompt = f"""
    Analyze this FMEA (Failure Mode and Effects Analysis) data:

    Process Statistics:
    - Total Failure Modes: {analysis_context['total_failures']}
    - High Risk Items: {analysis_context['high_risk']}
    - Medium Risk Items: {analysis_context['medium_risk']}
    - Object Types Affected: {', '.join(analysis_context['object_types'])}
    - Critical Activities: {', '.join(analysis_context['top_activities'])}

    Provide detailed analysis focusing on:
    1. Pattern identification in failure modes
    2. Critical risk areas and their implications
    3. Object interaction complexity
    4. Process vulnerability assessment
    5. Specific recommendations for improvement

    Format response as:
    1. Key findings about failure patterns
    2. Critical insights about process risks
    3. Specific, actionable recommendations
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an expert in FMEA and process mining analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        # Parse response into sections
        content = response.choices[0].message.content
        sections = content.split('\n\n')

        return {
            'findings': sections[0] if len(sections) > 0 else '',
            'insights': sections[1] if len(sections) > 1 else '',
            'recommendations': sections[2] if len(sections) > 2 else ''
        }
    except Exception as e:
        logger.error(f"Error getting AI insights: {str(e)}")
        return {
            'findings': 'Error generating insights',
            'insights': '',
            'recommendations': ''
        }

def display_rpn_distribution(fmea_results: List[Dict]):
    """
    Display comprehensive RPN distribution visualization with analysis breakdowns.
    This shows the spread of risk levels across different failure modes and helps
    identify risk clusters and patterns.
    """
    try:
        # Create base RPN histogram
        df = pd.DataFrame(fmea_results)
        fig = go.Figure()

        # Add main RPN distribution histogram
        fig.add_trace(go.Histogram(
            x=df['rpn'],
            nbinsx=20,
            name='RPN Distribution',
            marker_color='blue',
            opacity=0.7
        ))

        # Add critical threshold line
        fig.add_vline(
            x=200,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical Threshold (RPN=200)",
            annotation_position="top right"
        )

        # Add risk zone annotations
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

        # Update layout with detailed information
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
            # Add annotations for risk interpretation
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

        # Display the main distribution plot
        st.plotly_chart(fig, use_container_width=True)

        # Add distribution statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Average RPN",
                f"{df['rpn'].mean():.1f}",
                help="Average Risk Priority Number across all failure modes"
            )
        with col2:
            st.metric(
                "Median RPN",
                f"{df['rpn'].median():.1f}",
                help="Median Risk Priority Number (50th percentile)"
            )
        with col3:
            st.metric(
                "Std Deviation",
                f"{df['rpn'].std():.1f}",
                help="Standard deviation of RPN values"
            )
        with col4:
            st.metric(
                "90th Percentile",
                f"{df['rpn'].quantile(0.9):.1f}",
                help="90% of RPNs fall below this value"
            )

        # Add risk zone breakdown
        st.subheader("Risk Zone Analysis")
        risk_zones = {
            'Low Risk (RPN ≤ 100)': len(df[df['rpn'] <= 100]),
            'Medium Risk (100 < RPN ≤ 200)': len(df[(df['rpn'] > 100) & (df['rpn'] <= 200)]),
            'High Risk (RPN > 200)': len(df[df['rpn'] > 200])
        }

        # Create risk zone bar chart
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
        st.plotly_chart(risk_fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error in display_rpn_distribution: {str(e)}")
        st.error("Error displaying RPN distribution")


def display_fmea_analysis(fmea_results: List[Dict]):
    """Display comprehensive FMEA analysis with categorized tabs"""
    try:
        # Add explanation expander first
        with st.expander("Understanding FMEA Analysis Logic", expanded=False):
            st.markdown("""
                    # Understanding FMEA Analysis in Object-Centric Process Mining

                    ## Overview
                    The FMEA (Failure Mode and Effects Analysis) system analyzes object-centric event logs through multiple dimensions:

                    ### 1. Object-Level Analysis
                    - Analyzes failures related to object attributes and relationships
                    - Tracks missing required attributes
                    - Validates object relationships against OCEL model
                    - Monitors attribute value patterns and violations

                    ### 2. Activity-Level Analysis
                    - Sequence violations
                      * Compares actual vs expected activity sequences
                      * Tracks missing activities
                      * Identifies wrong order executions
                    - Timing violations
                      * Monitors activity durations
                      * Checks inter-activity gaps
                      * Validates against timing thresholds

                    ### 3. System-Level Analysis
                    - Convergence point analysis
                      * Identifies complex object interactions
                      * Monitors multi-object synchronization
                    - Divergence point analysis
                      * Tracks object lifecycle splits
                      * Validates object path separations

                    ### Risk Priority Number (RPN) Calculation

                    RPN = Severity × Likelihood × Detectability

                    Where:
                    - **Severity (1-10)**: Impact of failure
                      * Object criticality
                      * Business impact
                      * Regulatory implications

                    - **Likelihood (1-10)**: Probability of occurrence
                      * Historical frequency
                      * Process complexity
                      * Control effectiveness

                    - **Detectability (1-10)**: Ability to detect before impact
                      * Monitoring capabilities
                      * Control points
                      * Visibility in process

                    ### Risk Zones
                    - High Risk (RPN > 200)
                    - Medium Risk (100 < RPN ≤ 200)
                    - Low Risk (RPN ≤ 100)

                    ### Implementation Details
                    ```python
                    # Example calculation
                    severity = base_severity + object_criticality + multi_object_impact
                    likelihood = historical_frequency + complexity_factor
                    detectability = 10 - (automation_factor + visibility_factor + control_points)
                    rpn = severity * likelihood * detectability
                    ```

                    ### Data Structure Example
                    ```json
                    {
                      "id": "FM_001",
                      "activity": "Trade Execution",
                      "object_type": "Trade",
                      "description": "Missing required attributes",
                      "severity": 8,
                      "likelihood": 6,
                      "detectability": 4,
                      "rpn": 192,
                      "affected_objects": ["Trade_1", "Position_1"],
                      "root_causes": ["Incomplete data validation", "System timeout"]
                    }
                    ```
                    """)
        @st.cache_data
        def get_summary_stats(results):
            return {
                'total': len(results),
                'high_risk': sum(1 for r in results if r['rpn'] > 200),
                'medium_risk': sum(1 for r in results if 100 < r['rpn'] <= 200),
                'low_risk': sum(1 for r in results if r['rpn'] <= 100)
            }

        stats = get_summary_stats(fmea_results)

        # Display summary metrics
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Failure Modes", stats['total'])
            with col2:
                st.metric("High Risk (RPN > 200)", stats['high_risk'])
            with col3:
                st.metric("Medium Risk", stats['medium_risk'])
            with col4:
                st.metric("Low Risk", stats['low_risk'])

        # Create main tabs
        main_tabs = st.tabs(["Overview", "Detailed Analysis", "Recommendations"])

        with main_tabs[0]:
            if len(fmea_results) > 0:
                display_rpn_distribution(fmea_results)

        with main_tabs[1]:
            # Create subtabs for different failure types
            detailed_tabs = st.tabs(["Object-Level", "Activity-Level", "System-Level"])

            # Object-Level Analysis
            with detailed_tabs[0]:
                object_failures = [r for r in fmea_results if r['object_type'] != 'System' and
                                   r.get('violation_type', '').startswith('missing_')]
                if object_failures:
                    for failure in sorted(object_failures, key=lambda x: x['rpn'], reverse=True):
                        with st.expander(
                                f"{failure['object_type']} - {failure['description']} (RPN: {failure['rpn']})"):
                            # Metrics row
                            cols = st.columns(3)
                            cols[0].metric("Severity", failure['severity'])
                            cols[1].metric("Likelihood", failure['likelihood'])
                            cols[2].metric("Detectability", failure['detectability'])

                            st.markdown("---")

                            # Details section
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("##### Core Information")
                                st.write("**Object Type:**", failure['object_type'])
                                st.write("**Violation Type:**", failure.get('violation_type', 'Unknown'))

                                # Handle affected_objects
                                if 'affected_objects' in failure:
                                    affected_objs = []
                                    for obj in failure['affected_objects']:
                                        if isinstance(obj, dict):
                                            obj_str = f"{obj.get('type', 'Unknown')}: {obj.get('id', 'Unknown')}"
                                            affected_objs.append(obj_str)
                                        else:
                                            affected_objs.append(str(obj))
                                    st.write("**Affected Objects:**", ', '.join(affected_objs))

                            with col2:
                                st.markdown("##### Detailed Information")
                                if failure.get('relationship_details'):
                                    st.markdown("**Relationship Information:**")
                                    st.write(f"• Object ID: {failure['relationship_details'].get('object_id')}")
                                    st.write(f"• Event ID: {failure['relationship_details'].get('event_id')}")
                                    st.write(
                                        f"• Missing: {failure['relationship_details'].get('missing_relationship')}")
                                    st.write("• Current:", ', '.join(
                                        failure['relationship_details'].get('existing_relationships', [])))

                                if failure.get('attribute_details'):
                                    st.markdown("**Attribute Information:**")
                                    st.write(f"• Object ID: {failure['attribute_details'].get('object_id')}")
                                    st.write(f"• Event ID: {failure['attribute_details'].get('event_id')}")
                                    st.write("• Missing:",
                                             ', '.join(failure['attribute_details'].get('missing_attributes', [])))
                                    st.write("• Present:",
                                             ', '.join(failure['attribute_details'].get('present_attributes', [])))
                else:
                    st.info("No object-level failures detected")

            # Activity-Level Analysis
            with detailed_tabs[1]:
                activity_failures = [r for r in fmea_results if r.get('violation_type') in ['sequence', 'timing'] and
                                     r['object_type'] != 'System']
                if activity_failures:
                    for failure in sorted(activity_failures, key=lambda x: x['rpn'], reverse=True):
                        # Create descriptive header based on violation type
                        failure_type = "Time Gap Violation" if failure.get(
                            'violation_type') == 'timing' else "Sequence Violation"
                        failure_header = f"{failure['activity']} - {failure_type} (RPN: {failure['rpn']})"

                        with st.expander(failure_header):
                            # Metrics row
                            cols = st.columns(3)
                            cols[0].metric("Severity", failure['severity'])
                            cols[1].metric("Likelihood", failure['likelihood'])
                            cols[2].metric("Detectability", failure['detectability'])

                            st.markdown("---")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("##### Event Information")
                                if failure.get('event_details'):
                                    for key, value in failure['event_details'].items():
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                                elif failure.get('sequence_details'):  # Add event info for sequence violations
                                    st.write("**Case ID:**", failure['sequence_details'].get('case_id', 'Unknown'))
                                    if failure['sequence_details'].get('wrong_order'):
                                        first_violation = failure['sequence_details']['wrong_order'][0]
                                        st.write("**Event ID:**", first_violation.get('event_id', 'Unknown'))

                            # Only show sequence information for sequence violations
                            if failure.get('violation_type') == 'sequence':
                                with col2:
                                    st.markdown("##### Sequence Information")
                                    if failure.get('sequence_details'):
                                        st.write("**Expected:**",
                                                 ' → '.join(failure['sequence_details'].get('expected_sequence', [])))
                                        st.write("**Actual:**",
                                                 ' → '.join(failure['sequence_details'].get('actual_sequence', [])))
                                        if failure['sequence_details'].get('wrong_order'):
                                            st.markdown("**Violations:**")
                                            for violation in failure['sequence_details']['wrong_order']:
                                                st.write(
                                                    f"• Position {violation['position']}: Expected '{violation['expected']}', got '{violation['actual']}'")

                            # Show timing information for timing violations
                            elif failure.get('violation_type') == 'timing':
                                with col2:
                                    st.markdown("##### Timing Information")
                                    if failure.get('event_details'):
                                        time_diff = failure['event_details'].get('time_difference_hours')
                                        if time_diff:
                                            st.write(f"**Time Gap:** {time_diff:.2f} hours")
                                        st.write("**Previous Activity:**",
                                                 failure['event_details'].get('previous_event_id', 'Unknown'))
                else:
                    st.info("No activity-level failures detected")

            # System-Level Analysis
            with detailed_tabs[2]:
                system_failures = [r for r in fmea_results if r['object_type'] == 'System']
                if system_failures:
                    for failure in sorted(system_failures, key=lambda x: x['rpn'], reverse=True):
                        with st.expander(f"{failure['activity']} (RPN: {failure['rpn']})"):
                            cols = st.columns(3)
                            cols[0].metric("Severity", failure['severity'])
                            cols[1].metric("Likelihood", failure['likelihood'])
                            cols[2].metric("Detectability", failure['detectability'])

                            st.markdown("---")
                            st.markdown("##### System Impact")
                            st.write(failure['description'])
                else:
                    st.info("No system-level failures detected")

        with main_tabs[2]:
            # Display recommendations
            priorities = {
                'High': [r for r in fmea_results if r['rpn'] > 200],
                'Medium': [r for r in fmea_results if 100 < r['rpn'] <= 200],
                'Low': [r for r in fmea_results if r['rpn'] <= 100]
            }

            for priority, items in priorities.items():
                if items:
                    with st.expander(f"{priority} Priority Items ({len(items)})"):
                        for item in items[:5]:
                            st.markdown(f"**{item['activity']}** (RPN: {item['rpn']})")
                            st.write(item['description'])

                            # Handle affected_objects
                            if 'affected_objects' in item:
                                affected_objs = []
                                for obj in item['affected_objects']:
                                    if isinstance(obj, dict):
                                        obj_str = f"{obj.get('type', 'Unknown')}: {obj.get('id', 'Unknown')}"
                                        affected_objs.append(obj_str)
                                    else:
                                        affected_objs.append(str(obj))
                                st.write("**Affected Objects:**", ', '.join(affected_objs))

                            st.markdown("---")

                        if len(items) > 5:
                            st.write(f"... and {len(items) - 5} more items")

    except Exception as e:
        logger.error(f"Error in display_fmea_analysis: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error displaying analysis: {str(e)}")

