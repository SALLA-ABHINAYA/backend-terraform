from typing import Dict, List, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import os
import json
import requests

# Increase pandas display limit
pd.set_option("styler.render.max_elements", 600000)

# Import OCELFailureMode class from OCELFailure.py
from .OCELFailure import OCELFailureMode

# import OCELDataManager class from OCELDataManager.py
from .OCELDataManager import OCELDataManager

from utils import get_azure_openai_client


#Logger
# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Traceback
import traceback






class OCELEnhancedFMEA:
    """Enhanced FMEA analyzer for OCEL data"""

    def __init__(self, ocel_data: Dict):
        """Initialize analyzer with enhanced error handling and logging"""
        logger.info("Initializing OCELEnhancedFMEA")
        try:
            self.ocel_data = ocel_data

            # Initialize OCEL data manager for relationship/attribute management
            self.data_manager = OCELDataManager("api_response/process_data.json")

            # Validate OCEL data structure
            if not isinstance(ocel_data, dict):
                raise ValueError(f"Invalid OCEL data type: {type(ocel_data)}")

            if 'ocel:events' not in ocel_data:
                raise ValueError("Missing ocel:events in data")

            # Convert events to DataFrame with enhanced error handling
            logger.info("Converting events to DataFrame")
            events_data = []
            for event in ocel_data['ocel:events']:
                try:
                    event_data = {
                        'ocel:id': event.get('ocel:id', ''),
                        'ocel:timestamp': pd.to_datetime(event.get('ocel:timestamp')),
                        'ocel:activity': event.get('ocel:activity', ''),
                        'ocel:objects': event.get('ocel:objects', []),
                        'case_id': event.get('ocel:attributes', {}).get('case_id',
                                                                        event.get('case:concept:name',
                                                                                  event.get('ocel:id', ''))),
                    }

                    # Add all attributes
                    attributes = event.get('ocel:attributes', {})
                    for key, value in attributes.items():
                        event_data[key] = value

                    events_data.append(event_data)

                except Exception as e:
                    logger.warning(f"Error processing event: {str(e)}")
                    continue

            self.events = pd.DataFrame(events_data)
            logger.info(f"Created events DataFrame with {len(self.events)} rows")

            self._validate_columns()

            # Initialize core components using data manager
            self.object_types = list(self.data_manager.object_relationships.keys())
            logger.info(f"Found {len(self.object_types)} object types")

            # Build object relationships
            logger.info("Building object relationships")
            self.object_relationships = self._build_object_relationships()
            logger.info(f"Built relationships for {len(self.object_relationships)} objects")

            # Identify convergence points
            logger.info("Identifying convergence points")
            self.convergence_points = self._identify_convergence_points()
            logger.info(f"Identified {len(self.convergence_points)} convergence points")

            # Identify divergence points
            logger.info("Identifying divergence points")
            self.divergence_points = self._identify_divergence_points()
            logger.info(f"Identified {len(self.divergence_points)} divergence points")

            # Compute activity statistics
            logger.info("Computing activity statistics")
            self.activity_stats = self._compute_activity_stats()
            logger.info(f"Computed statistics for {len(self.activity_stats)} activities")

            # Compute object statistics
            logger.info("Computing object statistics")
            self.object_stats = self._compute_object_stats()
            logger.info(f"Computed statistics for {len(self.object_stats)} object types")

            # Analyze sequence patterns
            logger.info("Analyzing sequence patterns")
            self.sequence_patterns = self._analyze_sequence_patterns()
            if self.sequence_patterns:
                logger.info(f"Found patterns for {len(self.sequence_patterns)} object types")
            else:
                logger.warning("No sequence patterns found")

            # Initialize timing thresholds from OCEL model
            logger.info("Initializing timing thresholds")
            self.timing_thresholds = self._initialize_timing_thresholds()

            # Initialize empty failure modes list
            self.failure_modes = []

            # Store expected sequences from OCEL model
            self.expected_sequences = {
                obj_type: data.get('activities', [])
                for obj_type, data in self.data_manager.ocel_model.items()
            }

            # Generate or load FMEA settings
            fmea_settings_path = 'api_response/fmea_settings.json'
            if not os.path.exists(fmea_settings_path):
                logger.info("Generating new FMEA settings...")
                self.fmea_settings = self._generate_fmea_settings()
            else:
                logger.info("Loading existing FMEA settings...")
                with open(fmea_settings_path, 'r') as f:
                    self.fmea_settings = json.load(f)

            logger.info("Initialization complete")

        except Exception as e:
            logger.error(f"Error initializing OCELEnhancedFMEA: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _generate_fmea_settings(self) -> None:
        """Generate FMEA settings using Azure OpenAI based on OCEL model"""
        try:
            # Load OCEL model
            with open('api_response/output_ocel.json', 'r') as f:
                ocel_model = json.load(f)

            # Create context for OpenAI
            context = {
                'object_types': list(ocel_model.keys()),
                'object_details': {
                    obj_type: {
                        'activities': data.get('activities', []),
                        'attributes': data.get('attributes', []),
                        'relationships': data.get('relationships', [])
                    }
                    for obj_type, data in ocel_model.items()
                }
            }

            prompt = f"""
            Analyze this Object-Centric Process Mining (OCPM) model and generate FMEA settings:

            Object Types and Their Details:
            {json.dumps(context, indent=2)}

            Generate a comprehensive FMEA configuration with these components:

            1. object_visibility (scale -3 to 0, where -3 means highly visible/easily detectable):
            - Consider object's observability in process
            - More system interactions = more visible
            - More attributes = more visible
            - More relationships = more visible

            2. object_criticality (scale 1-5, where 5 is most critical):
            - Consider business impact
            - Consider number of relationships
            - Consider number of activities
            - Consider attribute importance

            3. temporal_dependencies:
            - Define required activity sequences
            - Specify timing constraints
            - Consider business logic dependencies
            - Include validation requirements

            4. critical_activities (activities that have high severity impact):
            - List activities that are critical to process success
            - Include execution activities
            - Include validation activities
            - Include settlement/completion activities

            5. regulatory_keywords:
            - List keywords that indicate regulatory/compliance significance
            - Include industry-specific compliance terms
            - Include risk-related terms
            - Include legal/regulatory terms

            Format the response as a JSON object with these exact five keys. For critical_activities and regulatory_keywords, 
            analyze the provided activities and their context to determine appropriate values.
            """

            # Get Azure OpenAI client
            client = get_azure_openai_client()

            # Generate settings - Fixed order of arguments
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in FMEA and process mining analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500,
                response_format={
                    "type": "json_object"
                }
            )

            # Parse and validate response
            settings = json.loads(response.choices[0].message.content)

            # Save settings
            output_path = os.path.join('api_response', 'fmea_settings.json')
            with open(output_path, 'w') as f:
                json.dump(obj=settings, fp=f, indent=2)

            # Send settings to API endpoint via POST request
            # try:
            #     response = requests.post("http://127.0.0.1:8000/fmea_settings", json=settings)
            #     if response.status_code == 200:
            #         logger.info("FMEA settings successfully sent to API endpoint")
            #     else:
            #         logger.error(f"Failed to send FMEA settings to API endpoint: {response.status_code} - {response.text}")
            # except requests.exceptions.RequestException as e:
            #     logger.error(f"Error sending FMEA settings to API endpoint: {str(e)}")

            logger.info(f"Generated FMEA settings saved to {output_path}")
            return settings

        except Exception as e:
            logger.error(f"Error generating FMEA settings: {str(e)}")
            logger.error(traceback.format_exc())
            raise


    def _initialize_timing_thresholds(self) -> Dict[str, Dict]:
        """Initialize timing thresholds from OCEL threshold file"""
        try:
            with open('api_response/output_ocel_threshold.json', 'r') as f:
                thresholds = json.load(f)
            logger.info("Loaded timing thresholds from file")
            return thresholds
        except Exception as e:
            logger.error(f"Error loading timing thresholds: {str(e)}")
            return {}

    def _validate_columns(self):
        """Validate required columns exist in DataFrame"""
        required_columns = ['case_id', 'ocel:activity', 'ocel:timestamp']
        missing = [col for col in required_columns if col not in self.events.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _build_object_relationships(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build graph of object relationships from events"""
        logger.info("Starting to build object relationships")
        try:
            relationships = defaultdict(list)

            # Track processed pairs to avoid duplicates
            processed_pairs = set()

            for _, event in self.events.iterrows():
                try:
                    objects = event.get('ocel:objects', [])
                    # Skip if less than 2 objects
                    if len(objects) < 2:
                        continue

                    # Analyze each object pair in the event
                    for i, obj1 in enumerate(objects):
                        for obj2 in objects[i + 1:]:
                            # Create unique pair identifier
                            pair_id = tuple(sorted([obj1['id'], obj2['id']]))

                            # Skip if already processed
                            if pair_id in processed_pairs:
                                continue

                            processed_pairs.add(pair_id)

                            # Record relationship for both objects
                            relationships[obj1['id']].append({
                                'related_object': obj2['id'],
                                'object_type': obj2.get('type', 'Unknown'),
                                'activity': event['ocel:activity'],
                                'timestamp': event['ocel:timestamp'],
                                'event_id': event['ocel:id']
                            })

                            relationships[obj2['id']].append({
                                'related_object': obj1['id'],
                                'object_type': obj1.get('type', 'Unknown'),
                                'activity': event['ocel:activity'],
                                'timestamp': event['ocel:timestamp'],
                                'event_id': event['ocel:id']
                            })

                except Exception as e:
                    logger.warning(f"Error processing event relationships: {str(e)}")
                    continue

            # Convert defaultdict to regular dict
            result = dict(relationships)
            logger.info(f"Built relationships for {len(result)} objects with {len(processed_pairs)} unique connections")

            # Log some statistics
            relationship_counts = {obj_id: len(rels) for obj_id, rels in result.items()}
            if relationship_counts:
                avg_relationships = sum(relationship_counts.values()) / len(relationship_counts)
                max_relationships = max(relationship_counts.values())
                logger.info(f"Average relationships per object: {avg_relationships:.2f}")
                logger.info(f"Maximum relationships for an object: {max_relationships}")

            return result

        except Exception as e:
            logger.error(f"Error building object relationships: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _check_timing_violation(self, event: pd.Series) -> bool:
        """
        Check for timing violations based on OCEL thresholds

        Args:
            event: Single event series containing timestamp and activity info

        Returns:
            bool: True if timing violation detected, False otherwise
        """
        try:
            # Get activity and case info
            activity = event['ocel:activity']
            case_id = event['case_id']
            timestamp = pd.to_datetime(event['ocel:timestamp'])

            # Get object types for this event
            object_types = []
            for obj in event.get('ocel:objects', []):
                if isinstance(obj, dict) and 'type' in obj:
                    object_types.append(obj['type'])

            # Get case events
            case_events = self.events[self.events['case_id'] == case_id]
            case_events = case_events.sort_values('ocel:timestamp')
            prev_event_idx = case_events.index.get_loc(event.name) - 1

            violation_found = False
            if prev_event_idx >= 0:
                prev_event = case_events.iloc[prev_event_idx]
                time_diff = (timestamp - pd.to_datetime(prev_event['ocel:timestamp'])).total_seconds() / 3600

                # Check thresholds for each object type
                for obj_type in object_types:
                    if obj_type in self.timing_thresholds:
                        obj_rules = self.timing_thresholds[obj_type]

                        # Check activity-specific threshold
                        if activity in obj_rules['activity_thresholds']:
                            threshold = obj_rules['activity_thresholds'][activity]['max_gap_after_hours']
                            if time_diff > threshold:
                                violation_found = True
                                break
                        else:
                            # Use default gap threshold
                            if time_diff > obj_rules['default_gap_hours']:
                                violation_found = True
                                break

                        # Check total duration threshold
                        total_duration = (case_events['ocel:timestamp'].max() -
                                          case_events['ocel:timestamp'].min()).total_seconds() / 3600
                        if total_duration > obj_rules['total_duration_hours']:
                            violation_found = True
                            break

            return violation_found

        except Exception as e:
            logger.error(f"Error checking timing violation: {str(e)}")
            return False

    def _has_temporal_dependency(self, activity: str) -> bool:
        """
        Analyze temporal dependencies of activities in object-centric context.

        This method examines complex temporal relationships between activities, considering:
        - Direct activity dependencies (e.g., validation before execution)
        - Cross-object temporal constraints
        - Business cycle dependencies (e.g., end-of-day processes)
        - Market-driven timing requirements

        Args:
            activity: Name of the activity to check for temporal dependencies

        Returns:
            bool: True if activity has temporal dependencies, False otherwise
        """
        try:
            # Define temporal dependency map for activities
            temporal_dependencies = self.fmea_settings['temporal_dependencies']

            # Check if activity has defined temporal dependencies
            if activity in temporal_dependencies:
                dependency_info = temporal_dependencies[activity]

                # Check for any type of temporal constraint
                has_timing_requirements = any([
                    dependency_info['required_before'],
                    dependency_info['required_after'],
                    dependency_info['timing_constraint'] != '',
                    dependency_info['max_delay'] is not None
                ])

                if has_timing_requirements:
                    # Validate temporal patterns in event log
                    activity_events = self.events[self.events['ocel:activity'] == activity]

                    for _, event in activity_events.iterrows():
                        case_id = event['case_id']
                        timestamp = pd.to_datetime(event['ocel:timestamp'])

                        # Get case events
                        case_events = self.events[self.events['case_id'] == case_id]
                        case_events = case_events.sort_values('ocel:timestamp')

                        # Check required predecessor activities
                        for req_activity in dependency_info['required_before']:
                            prior_events = case_events[
                                (case_events['ocel:timestamp'] < timestamp) &
                                (case_events['ocel:activity'] == req_activity)
                                ]
                            if prior_events.empty:
                                return True

                        # Check required successor activities
                        for req_activity in dependency_info['required_after']:
                            subsequent_events = case_events[
                                (case_events['ocel:timestamp'] > timestamp) &
                                (case_events['ocel:activity'] == req_activity)
                                ]
                            if subsequent_events.empty:
                                return True

                        # Check timing constraints
                        constraint = dependency_info['timing_constraint']
                        if constraint == 'market_hours' and not self._is_market_hours(timestamp):
                            return True
                        elif constraint == 'end_of_day' and not self._is_end_of_day(timestamp):
                            return True
                        elif constraint == 'real_time' and not self._is_real_time(timestamp):
                            return True

                        # Check maximum delay constraint
                        max_delay = dependency_info['max_delay']
                        if max_delay and self._exceeds_max_delay(event, max_delay):
                            return True

                return has_timing_requirements

            return False

        except Exception as e:
            logger.error(f"Error checking temporal dependency for {activity}: {str(e)}")
            return False

    def _is_end_of_day(self, timestamp: datetime) -> bool:
        """Check if timestamp is within end-of-day processing window (16:00-18:00)"""
        return 16 <= timestamp.hour < 18 and timestamp.weekday() < 5

    def _is_real_time(self, timestamp: datetime) -> bool:
        """Check if event happened within real-time processing threshold (< 1 second delay)"""
        current_time = pd.Timestamp.now()
        return (current_time - timestamp).total_seconds() < 1

    def _exceeds_max_delay(self, event: pd.Series, max_delay: timedelta) -> bool:
        """Check if event exceeds maximum allowed delay"""
        case_events = self.events[self.events['case_id'] == event['case_id']]
        case_events = case_events.sort_values('ocel:timestamp')

        event_idx = case_events.index.get_loc(event.name)
        if event_idx > 0:
            prev_event = case_events.iloc[event_idx - 1]
            actual_delay = pd.to_datetime(event['ocel:timestamp']) - pd.to_datetime(prev_event['ocel:timestamp'])
            return actual_delay > max_delay

        return False

    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is within market hours (9:00-17:00)"""
        return 9 <= timestamp.hour < 17 and timestamp.weekday() < 5

    def _has_timing_dependency_violation(self, event: pd.Series, case_events: pd.DataFrame) -> bool:
        """
        Check for violations of timing dependencies between objects

        Dependencies examples:
        - Market data must be validated before trade execution
        - Risk assessment must complete before settlement
        - Position reconciliation must follow settlement
        """
        activity = event['ocel:activity']
        timestamp = pd.to_datetime(event['ocel:timestamp'])

        # Define required predecessor activities
        dependencies = {
            'Trade Execution': ['Market Data Validation', 'Risk Assessment'],
            'Settlement': ['Trade Execution', 'Position Reconciliation'],
            'Position Reconciliation': ['Trade Execution']
        }

        required_activities = dependencies.get(activity, [])
        if required_activities:
            prior_activities = case_events[
                case_events['ocel:timestamp'] < timestamp
                ]['ocel:activity'].unique()

            # Check if any required activity is missing
            if not all(req in prior_activities for req in required_activities):
                return True

        return False

    def _compute_object_stats(self) -> Dict[str, Dict[str, Any]]:
        """Compute object-level statistics"""
        stats = defaultdict(lambda: {
            'count': 0,
            'activities': set(),
            'related_objects': set(),
            'attributes': defaultdict(set)
        })

        for _, event in self.events.iterrows():
            for obj in event.get('ocel:objects', []):
                obj_type = obj.get('type', 'Unknown')
                stats[obj_type]['count'] += 1
                stats[obj_type]['activities'].add(event['ocel:activity'])

                # Track attributes
                for attr, value in obj.items():
                    if attr not in ['id', 'type']:
                        stats[obj_type]['attributes'][attr].add(str(value))

        return dict(stats)

    def _analyze_sequence_patterns(self) -> Dict[str, List[str]]:
        """Analyze common sequence patterns in the log"""
        logger.info("Starting sequence pattern analysis")
        patterns = defaultdict(list)

        try:
            # Get case ID column - try different possible names
            case_id_col = None
            possible_case_columns = ['case:concept:name', 'case_id', 'case', 'case:id']

            for col in possible_case_columns:
                if col in self.events.columns:
                    case_id_col = col
                    logger.info(f"Found case ID column: {col}")
                    break

            if not case_id_col:
                logger.warning("No case ID column found, using ocel:id as case identifier")
                case_id_col = 'ocel:id'

            # Group events by case
            logger.info(f"Grouping events by {case_id_col}")
            case_events = self.events.groupby(case_id_col)

            pattern_count = 0
            for case_id, events in case_events:
                try:
                    sorted_events = events.sort_values('ocel:timestamp')
                    sequence = list(sorted_events['ocel:activity'])

                    # Store sequence for each object type involved
                    for obj in events.iloc[0].get('ocel:objects', []):
                        obj_type = obj.get('type', 'Unknown')
                        if sequence not in patterns[obj_type]:
                            patterns[obj_type].append(sequence)
                            pattern_count += 1

                except Exception as e:
                    logger.error(f"Error processing case {case_id}: {str(e)}")
                    continue

            logger.info(f"Found {pattern_count} unique sequence patterns across {len(patterns)} object types")
            return dict(patterns)

        except Exception as e:
            logger.error(f"Error in sequence pattern analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def calculate_likelihood(self, failure_mode: OCELFailureMode) -> int:
        """Calculate likelihood considering OCPM patterns"""
        likelihood = 1

        # Analyze historical occurrence
        activity_events = self.events[self.events['ocel:activity'] == failure_mode.activity]
        total_instances = len(activity_events)

        if total_instances > 0:
            # Calculate failure rate for this activity
            failure_instances = activity_events[
                activity_events.apply(
                    lambda x: self._has_failure_condition(x, failure_mode.violation_type),
                    axis=1
                )
            ]
            failure_rate = len(failure_instances) / total_instances

            # Scale failure rate to likelihood score (1-10)
            likelihood += int(failure_rate * 10)

        # Add complexity factor
        object_interactions = self._get_object_interactions(failure_mode.activity)
        if len(object_interactions) > 2:
            likelihood += 1

        # Consider temporal patterns
        if self._has_temporal_dependency(failure_mode.activity):
            likelihood += 1

        return min(likelihood, 10)

    def _has_failure_condition(self, event: pd.Series, violation_type: str) -> bool:
        """Check if event has specific failure condition"""
        if violation_type == 'attribute':
            return any(attr not in event for attr in ['currency_pair', 'notional_amount'])
        elif violation_type == 'relationship':
            return len(event.get('ocel:objects', [])) < 2
        elif violation_type == 'timing':
            return self._check_timing_violation(event)
        return False

    def _get_object_interactions(self, activity: str) -> Set[str]:
        """Get unique object types interacting in activity"""
        interactions = set()
        activity_events = self.events[self.events['ocel:activity'] == activity]
        for _, event in activity_events.iterrows():
            for obj in event.get('ocel:objects', []):
                interactions.add(obj.get('type'))
        return interactions

    def calculate_detectability(self, failure_mode: OCELFailureMode) -> int:
        """Calculate detectability in OCPM context"""
        detectability = 10

        # Use settings for object visibility
        detectability += self.fmea_settings['object_visibility'].get(failure_mode.object_type, 0)

        # Keep rest of the method same
        if self._has_automated_monitoring(failure_mode.activity):
            detectability -= 3

        # Process controls
        control_points = self._get_control_points(failure_mode.activity)
        detectability -= min(len(control_points), 3)

        # Multi-object detectability
        if len(self._get_object_interactions(failure_mode.activity)) > 1:
            detectability -= 1

        return max(detectability, 1)

    def _has_automated_monitoring(self, activity: str) -> bool:
        """Check if activity has automated monitoring"""
        automated_activities = {
            'Trade Validation', 'Market Data Validation',
            'Position Reconciliation', 'Risk Assessment'
        }
        return activity in automated_activities

    def _get_control_points(self, activity: str) -> List[str]:
        """Get control points for activity"""
        controls = {
            'Trade Execution': ['Pre-trade validation', 'Price validation', 'Limit checks'],
            'Settlement': ['Position verification', 'Payment validation', 'Reconciliation'],
            'Risk Assessment': ['Exposure validation', 'Limit monitoring', 'Portfolio checks']
        }
        return controls.get(activity, [])


    def _validate_sequence(self, actual: List[str], expected: List[str]) -> bool:
        """Validate if actual sequence follows expected pattern"""
        if not expected:
            return True

        # Check if actual sequence contains expected sequence in order
        i = 0
        for act in actual:
            if i < len(expected) and act == expected[i]:
                i += 1
        return i == len(expected)

    def _get_primary_object_type(self, activity: str) -> str:
        """Determine primary object type for activity"""
        activity_objects = defaultdict(int)
        activity_events = self.events[self.events['ocel:activity'] == activity]

        for _, event in activity_events.iterrows():
            for obj in event.get('ocel:objects', []):
                activity_objects[obj.get('type', 'Unknown')] += 1

        return max(activity_objects.items(), key=lambda x: x[1])[0] if activity_objects else 'Unknown'

    def _identify_convergence_points(self) -> List[Dict[str, Any]]:
        """Identify points where multiple objects converge in the process"""
        logger.info("Starting convergence point identification")
        try:
            convergence_points = []
            for _, event in self.events.iterrows():
                objects = event.get('ocel:objects', [])
                if len(objects) > 1:
                    convergence_point = {
                        'event_id': event['ocel:id'],
                        'activity': event['ocel:activity'],
                        'timestamp': event['ocel:timestamp'],
                        'objects': objects,
                        'object_types': list(set(obj.get('type') for obj in objects))
                    }
                    convergence_points.append(convergence_point)

            logger.info(f"Found {len(convergence_points)} convergence points")
            return convergence_points

        except Exception as e:
            logger.error(f"Error identifying convergence points: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _identify_divergence_points(self) -> List[Dict[str, Any]]:
        """Identify points where object paths diverge"""
        logger.info("Starting divergence point identification")
        try:
            # Track object paths
            object_paths = defaultdict(list)
            for _, event in self.events.iterrows():
                for obj in event.get('ocel:objects', []):
                    object_paths[obj['id']].append({
                        'event_id': event['ocel:id'],
                        'timestamp': event['ocel:timestamp'],
                        'activity': event['ocel:activity']
                    })

            divergence_points = []
            for obj_id, path in object_paths.items():
                # Sort path by timestamp
                sorted_path = sorted(path, key=lambda x: x['timestamp'])

                for i in range(len(sorted_path) - 1):
                    current_event = sorted_path[i]
                    next_event = sorted_path[i + 1]

                    # Check for divergence conditions
                    current_objects = set([obj['id'] for obj in
                                           self.events[self.events['ocel:id'] == current_event['event_id']]
                                          .iloc[0].get('ocel:objects', [])])

                    next_objects = set([obj['id'] for obj in
                                        self.events[self.events['ocel:id'] == next_event['event_id']]
                                       .iloc[0].get('ocel:objects', [])])

                    if len(current_objects.intersection(next_objects)) < len(current_objects):
                        divergence_points.append({
                            'object_id': obj_id,
                            'event_id': current_event['event_id'],
                            'next_event_id': next_event['event_id'],
                            'timestamp': current_event['timestamp'],
                            'activity': current_event['activity'],
                            'next_activity': next_event['activity']
                        })

            logger.info(f"Found {len(divergence_points)} divergence points")
            return divergence_points

        except Exception as e:
            logger.error(f"Error identifying divergence points: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _compute_activity_stats(self) -> Dict[str, Dict[str, Any]]:
        """Compute comprehensive activity statistics"""
        stats = defaultdict(lambda: {
            'count': 0,
            'objects': defaultdict(set),
            'resources': set(),
            'duration_stats': [],
            'sequence_patterns': defaultdict(int),
            'error_rates': defaultdict(int),
            'attribute_patterns': defaultdict(set)
        })

        for _, event in self.events.iterrows():
            activity = event['ocel:activity']
            stats[activity]['count'] += 1

            # Track objects and their relationships
            for obj in event.get('ocel:objects', []):
                stats[activity]['objects'][obj.get('type')].add(obj['id'])

            # Track resources
            if 'resource' in event.get('ocel:attributes', {}):
                stats[activity]['resources'].add(event['ocel:attributes']['resource'])

            # Track attributes
            for attr, value in event.get('ocel:attributes', {}).items():
                stats[activity]['attribute_patterns'][attr].add(str(value))

        return dict(stats)

    def calculate_severity(self, failure_mode: OCELFailureMode) -> int:
        """Calculate severity in OCPM context"""
        try:
            # Get base severity from settings
            severity = self.fmea_settings['object_criticality'].get(failure_mode.object_type, 1)

            # Calculate multi-object impact
            affected_objects = set()
            events = self.events[self.events['ocel:activity'] == failure_mode.activity]
            for _, event in events.iterrows():
                for obj in event.get('ocel:objects', []):
                    affected_objects.add(obj.get('type'))

            # Add severity based on number of affected object types
            severity += min(len(affected_objects), 3)

            # Check if activity is critical (using settings instead of hardcoded values)
            if failure_mode.activity in self.fmea_settings['critical_activities']:
                severity += 1

            # Check for regulatory/compliance impact using keywords from settings
            if any(keyword in failure_mode.description.lower()
                   for keyword in self.fmea_settings['regulatory_keywords']):
                severity += 1

            return min(severity, 10)

        except Exception as e:
            logger.error(f"Error calculating severity: {str(e)}")
            logger.error(traceback.format_exc())
            return 5  # Return moderate severity as fallback

    def _calculate_cascade_depth(self, failure_mode: OCELFailureMode) -> int:
        """Calculate how deep failures can cascade"""
        depth = 0
        affected = {failure_mode.activity}
        current_level = {failure_mode.activity}

        while current_level and depth < 5:
            next_level = set()
            for activity in current_level:
                for _, event in self.events[self.events['ocel:activity'] == activity].iterrows():
                    objects = event.get('ocel:objects', [])
                    for obj in objects:
                        related = self.object_relationships.get(obj['id'], [])
                        for rel_obj in related:
                            rel_events = self.events[
                                self.events['ocel:objects'].apply(
                                    lambda x: rel_obj in [o['id'] for o in x]
                                )
                            ]
                            next_activities = set(rel_events['ocel:activity']) - affected
                            next_level.update(next_activities)

            if next_level:
                depth += 1
                affected.update(next_level)
                current_level = next_level
            else:
                break

        return depth

    def identify_failure_modes(self) -> List[Dict]:
        """Enhanced failure mode identification"""
        failure_modes = []

        # Object-level failures
        failure_modes.extend(self._identify_object_failures())

        # Event-level failures
        failure_modes.extend(self._identify_event_failures())

        # System-level failures
        failure_modes.extend(self._identify_system_failures())

        # Calculate metrics
        for fm in failure_modes:
            fm['severity'] = self.calculate_severity(OCELFailureMode(**fm))
            fm['likelihood'] = self.calculate_likelihood(OCELFailureMode(**fm))
            fm['detectability'] = self.calculate_detectability(OCELFailureMode(**fm))
            fm['rpn'] = fm['severity'] * fm['likelihood'] * fm['detectability']

        return sorted(failure_modes, key=lambda x: x['rpn'], reverse=True)

    def _is_valid_relationship(self, source_type: str, target_type: str) -> bool:
        """Check if relationship between object types is valid based on OCEL model"""
        # Get allowed relationships from OCEL model
        allowed_relationships = self.data_manager.ocel_model.get(source_type, {}).get('relationships', [])
        return target_type in allowed_relationships

    def _get_single_relationship_types(self) -> Set[str]:
        """Get object types that should only have single relationship per event"""
        # Objects that should have 1:1 relationships
        single_relation_types = {
            'Trade',  # One trade per event
            'Position',  # One position per event
            'Market'  # One market data point per event
        }
        return single_relation_types

    def _identify_event_failures(self) -> List[Dict]:
        """Identify event-level failures with detailed tracing"""
        failures = []
        logger.info("Identifying event-level failures")

        for case_id in self.events['case_id'].unique():
            case_events = self.events[self.events['case_id'] == case_id].sort_values('ocel:timestamp')

            # Track timing violations with event details
            for idx, event in case_events.iterrows():
                if self._check_timing_violation(event):
                    # Get previous event details for context
                    prev_events = case_events[case_events.index < idx]
                    prev_event = prev_events.iloc[-1] if not prev_events.empty else None

                    # Calculate time difference if previous event exists
                    time_diff = None
                    if prev_event is not None:
                        time_diff = (pd.to_datetime(event['ocel:timestamp']) -
                                     pd.to_datetime(prev_event['ocel:timestamp'])).total_seconds() / 3600

                    failure = OCELFailureMode(
                        id=f"EF_T_{len(failures)}",
                        activity=event['ocel:activity'],
                        object_type=self._get_primary_object_type(event['ocel:activity']),
                        description=(
                            f"Timing violation in case {case_id}\n"
                            f"Event ID: {event['ocel:id']}\n"
                            f"Previous Activity: {prev_event['ocel:activity'] if prev_event is not None else 'None'}\n"
                            f"Time Gap: {f'{time_diff:.2f} hours' if time_diff else 'N/A'}"
                        ),
                        violation_type='timing',
                        severity=8,
                        likelihood=7,
                        detectability=4,
                        rpn=224,
                        event_details={
                            'event_id': event['ocel:id'],
                            'timestamp': event['ocel:timestamp'],
                            'case_id': case_id,
                            'previous_event_id': prev_event['ocel:id'] if prev_event is not None else None,
                            'time_difference_hours': time_diff
                        },
                        affected_objects=[
                            {
                                'id': obj['id'],
                                'type': obj['type']
                            } for obj in event.get('ocel:objects', [])
                        ]
                    )
                    failures.append(failure.__dict__)  # Convert to dict for storage

            # Check sequence violations with detailed object tracing
            for obj_type in self.object_types:
                expected_sequence = self.data_manager.get_expected_activities(obj_type)
                obj_events = case_events[
                    case_events['ocel:objects'].apply(
                        lambda x: any(obj.get('type') == obj_type for obj in x)
                    )
                ]
                actual_sequence = obj_events['ocel:activity'].tolist()

                if expected_sequence and not self._validate_sequence(actual_sequence, expected_sequence):
                    missing_activities = set(expected_sequence) - set(actual_sequence)
                    wrong_order = []

                    # Identify specific sequence violations
                    for i, (exp, act) in enumerate(zip(expected_sequence, actual_sequence)):
                        if exp != act:
                            wrong_order.append({
                                'position': i,
                                'expected': exp,
                                'actual': act,
                                'event_id': obj_events.iloc[i]['ocel:id']
                            })

                    failure = {
                        'id': f"EF_S_{len(failures)}",
                        'activity': actual_sequence[0] if actual_sequence else 'Unknown',
                        'object_type': obj_type,
                        'description': (
                            f"Sequence violation in case {case_id}\n"
                            f"Missing activities: {', '.join(missing_activities) if missing_activities else 'None'}\n"
                            f"Wrong order activities: {len(wrong_order)}"
                        ),
                        'violation_type': 'sequence',
                        'severity': 7,
                        'likelihood': 6,
                        'detectability': 5,
                        'rpn': 210,
                        'sequence_details': {
                            'case_id': case_id,
                            'expected_sequence': expected_sequence,
                            'actual_sequence': actual_sequence,
                            'missing_activities': list(missing_activities),
                            'wrong_order': wrong_order
                        },
                        'affected_objects': [
                            {
                                'id': obj['id'],
                                'type': obj['type']
                            } for event in obj_events.itertuples()
                            for obj in event._asdict().get('ocel:objects', [])
                            if obj.get('type') == obj_type
                        ]
                    }
                    failures.append(failure)

        return failures

    def _identify_object_failures(self) -> List[Dict]:
        """Identify object-level failures with optimized processing"""
        failures = []
        logger.info("Identifying object-level failures")

        # Pre-process event data for efficient lookup
        event_object_map = defaultdict(list)
        object_event_map = defaultdict(list)
        object_type_map = {}

        # Build indexes in one pass through events
        for event in self.ocel_data['ocel:events']:
            event_id = event['ocel:id']
            event_object_map[event_id] = []

            for obj in event['ocel:objects']:
                obj_id = obj['id']
                obj_type = obj['type']

                event_object_map[event_id].append((obj_id, obj_type))
                object_event_map[obj_id].append(event_id)
                object_type_map[obj_id] = obj_type

        # Process each object type
        for obj_type in self.object_types:
            # Get expected relationships and attributes
            expected_relationships = self.data_manager.get_expected_relationships(obj_type)
            expected_attributes = self.data_manager.get_expected_attributes(obj_type)

            logger.info(f"Getting expected relationships for {obj_type}")
            logger.info(f"Found {len(expected_relationships)} expected relationships for {obj_type}")

            # Track objects of this type
            type_objects = {obj_id for obj_id, type_ in object_type_map.items() if type_ == obj_type}

            # Process each object
            for obj_id in type_objects:
                # Get all events for this object
                obj_events = object_event_map[obj_id]

                # Get all related objects
                related_types = set()
                for event_id in obj_events:
                    for rel_obj_id, rel_type in event_object_map[event_id]:
                        if rel_obj_id != obj_id:
                            related_types.add(rel_type)

                # Check relationship violations
                missing_relationships = expected_relationships - related_types
                if missing_relationships:
                    # Get first event for context
                    first_event = next((e for e in self.ocel_data['ocel:events']
                                        if e['ocel:id'] == obj_events[0]), None)

                    if first_event:
                        failures.append(OCELFailureMode(
                            id=f"OF_R_{len(failures)}",
                            activity=first_event['ocel:activity'],
                            object_type=obj_type,
                            description=f"Missing relationships with: {', '.join(missing_relationships)}",
                            violation_type='missing_relationship',
                            event_id=first_event['ocel:id'],
                            object_id=obj_id,
                            relationship_details={
                                'object_id': obj_id,
                                'event_id': first_event['ocel:id'],
                                'missing_relationship': list(missing_relationships),
                                'current': list(related_types)
                            }
                        ).__dict__)

                # Check attribute violations (only for object's events)
                for event_id in obj_events:
                    event = next(e for e in self.ocel_data['ocel:events'] if e['ocel:id'] == event_id)
                    event_attrs = set(event.get('ocel:attributes', {}).keys())
                    missing_attrs = expected_attributes - event_attrs

                    if missing_attrs:
                        failures.append(OCELFailureMode(
                            id=f"OF_A_{len(failures)}",
                            activity=event['ocel:activity'],
                            object_type=obj_type,
                            description=f"Missing required attributes: {', '.join(missing_attrs)}",
                            violation_type='missing_attribute',
                            event_id=event['ocel:id'],
                            object_id=obj_id,
                            attribute_details={
                                'object_id': obj_id,
                                'event_id': event['ocel:id'],
                                'missing_attributes': list(missing_attrs),
                                'present_attributes': list(event_attrs)
                            }
                        ).__dict__)

        return failures

    def _identify_system_failures(self) -> List[Dict]:
        """Identify system-wide failures"""
        failures = []

        # Check convergence points
        for cp in self.convergence_points:
            if len(cp['object_types']) > 3:  # Too many objects converging
                failures.append({
                    'id': f"SF_C_{len(failures)}",
                    'activity': cp['activity'],
                    'object_type': 'System',
                    'description': f"Complex convergence point with {len(cp['object_types'])} object types",
                    'violation_type': 'convergence',
                    'severity': 7,
                    'likelihood': 6,
                    'detectability': 5,
                    'rpn': 210
                })

        # Check divergence points
        for dp in self.divergence_points:
            # Get event details
            event = self.events[self.events['ocel:id'] == dp['event_id']].iloc[0]
            next_event = self.events[self.events['ocel:id'] == dp['next_event_id']].iloc[0]

            # Check for uncontrolled divergence
            if len(next_event.get('ocel:objects', [])) < len(event.get('ocel:objects', [])) / 2:
                failures.append({
                    'id': f"SF_D_{len(failures)}",
                    'activity': dp['activity'],
                    'object_type': 'System',
                    'description': f"Uncontrolled object divergence in {dp['activity']}",
                    'violation_type': 'divergence',
                    'severity': 8,
                    'likelihood': 7,
                    'detectability': 4,
                    'rpn': 224
                })

        return failures