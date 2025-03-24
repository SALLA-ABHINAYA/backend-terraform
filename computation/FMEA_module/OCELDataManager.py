import pandas as pd
from typing import Dict, List, Any, Set
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Increase pandas display limit
pd.set_option("styler.render.max_elements", 600000)

from backend.MasterApi.Routers.central_log import log_time

class OCELDataManager:
    """Manages OCEL data loading and common operations"""

    def __init__(self, ocel_path: str):
        """Initialize with path to OCEL data"""
        try:
            start=log_time("OCELDataManager constructor","START")
            # Load OCEL relationship definitions
            with open('api_response/output_ocel.json', 'r') as f:
                self.ocel_model = json.load(f)

            

            # Load process data
            with open(ocel_path, 'r') as f:
                self.ocel_data = json.load(f)

            # Create events DataFrame
            self.events_df = self._create_events_df()

            # Build attribute maps for each object type
            self.object_attributes = self._build_object_attribute_maps()
            logger.info(f"Built object attribute maps: {json.dumps(self.object_attributes, indent=2)}")

            # Store object types and their relationships
            self.object_relationships = {
                obj_type: data.get('relationships', [])
                for obj_type, data in self.ocel_model.items()
            }
            logger.info(f"Built object types and their relationships maps: {json.dumps(self.object_relationships, indent=2)}")

            # Build object activities map
            self.object_activities = self._build_object_activities_map()
            logger.info(f"Built object activities map: {json.dumps(self.object_activities, indent=2)}")

            log_time("OCELDataManager constructor","END",start)
        except Exception as e:
            logger.error(f"Error initializing OCEL data manager: {str(e)}")
            raise

    def _create_events_df(self) -> pd.DataFrame:
        start=log_time("_create_events_df","START")
        """Create DataFrame from OCEL events"""
        events_data = []
        for event in self.ocel_data['ocel:events']:
            event_data = {
                'ocel:id': event.get('ocel:id', ''),
                'ocel:timestamp': pd.to_datetime(event.get('ocel:timestamp')),
                'ocel:activity': event.get('ocel:activity', ''),
                'ocel:objects': event.get('ocel:objects', []),
                'case_id': event.get('ocel:attributes', {}).get('case_id',
                                                                event.get('case:concept:name',
                                                                          event.get('ocel:id', '')))
            }

            # Add all attributes
            attributes = event.get('ocel:attributes', {})
            for key, value in attributes.items():
                event_data[key] = value

            events_data.append(event_data)
        log_time("_create_events_df","END",start)
        return pd.DataFrame(events_data)

    def _build_object_attribute_maps(self) -> Dict[str, List[str]]:
        """Build attribute maps for each object type strictly based on OCEL model attributes"""
        try:
            start=log_time("_build_object_attribute_maps","START")
            # Initialize empty maps
            attribute_maps = {}

            # Get expected attributes directly from OCEL model
            for obj_type, obj_data in self.ocel_model.items():
                # Get attributes list directly from model
                expected_attrs = obj_data.get('attributes', [])
                # Store only business attributes, filtering out system/event attributes
                attribute_maps[obj_type] = sorted([
                    attr for attr in expected_attrs
                    if attr not in {'activity', 'object_type', 'case_id', 'resource'}
                ])
                logger.info(
                    f"Loaded {len(attribute_maps[obj_type])} attributes for {obj_type}: {attribute_maps[obj_type]}")

            logger.info(f"Built object attribute maps: {json.dumps(attribute_maps, indent=2)}")

            log_time("_build_object_attribute_maps","END",start)
            return attribute_maps

        except Exception as e:
            logger.error(f"Error building object attribute maps: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _build_object_activities_map(self) -> Dict[str, List[str]]:
        start=log_time("_build_object_activities_map","START")
        """Build map of expected activities for each object type"""
        activities_map = {}

        # Get activities from ocel:objects definitions
        for obj in self.ocel_data.get('ocel:objects', []):
            obj_type = obj.get('ocel:type')
            if obj_type:
                if obj_type not in activities_map:
                    activities_map[obj_type] = set()

                # Get activity from object attributes
                activity = obj.get('ocel:attributes', {}).get('activity')
                if activity:
                    activities_map[obj_type].add(activity)

        # Augment with activities from events
        for event in self.ocel_data.get('ocel:events', []):
            activity = event.get('ocel:activity')
            if activity:
                for obj in event.get('ocel:objects', []):
                    obj_type = obj.get('type')
                    if obj_type:
                        if obj_type not in activities_map:
                            activities_map[obj_type] = set()
                        activities_map[obj_type].add(activity)

        # Convert sets to sorted lists
        for obj_type in activities_map:
            activities_map[obj_type] = sorted(list(activities_map[obj_type]))
            logger.info(f"Activities for {obj_type}: {activities_map[obj_type]}")


        log_time("_build_object_activities_map","END",start)
        return activities_map

    def get_expected_attributes(self, obj_type: str) -> Set[str]:
        """Get expected attributes for object type"""
        try:
            start=log_time("get_expected_attributes","START")
            attributes = set(self.object_attributes.get(obj_type, []))
            logger.info(f"Getting expected attributes for {obj_type}: {sorted(list(attributes))}")

            log_time("get_expected_attributes","END",start)
            return attributes
        except Exception as e:
            logger.error(f"Error getting expected attributes for {obj_type}: {str(e)}")
            return set()

    def get_expected_relationships(self, obj_type: str) -> Set[str]:
        """Get expected relationships for object type from OCEL model"""
        return set(self.object_relationships.get(obj_type, []))

    def get_expected_activities(self, obj_type: str) -> List[str]:
        """Get expected activities for object type"""
        return self.object_activities.get(obj_type, [])