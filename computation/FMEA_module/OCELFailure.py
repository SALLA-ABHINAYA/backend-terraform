from dataclasses import dataclass, field
from typing import Dict, List, Any, Set


@dataclass
class OCELFailureMode:
    """Enhanced failure mode representation for OCEL process mining"""
    id: str
    activity: str
    object_type: str
    description: str
    severity: int = 0
    likelihood: int = 0
    detectability: int = 0
    rpn: int = 0
    violation_type: str = ""
    event_id: str = ""  # Added this field
    object_id: str = "" # Added this field
    event_details: Dict[str, Any] = field(default_factory=dict)
    sequence_details: Dict[str, Any] = field(default_factory=dict)
    relationship_details: Dict[str, Any] = field(default_factory=dict)
    attribute_details: Dict[str, Any] = field(default_factory=dict)
    affected_objects: List[Dict[str, str]] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    causes: List[str] = field(default_factory=list)

    # Business context
    business_impact: Dict[str, Any] = field(default_factory=dict)
    operational_impact: Dict[str, Any] = field(default_factory=dict)
    regulatory_impact: Dict[str, Any] = field(default_factory=dict)

    # Control mechanisms
    current_controls: List[str] = field(default_factory=list)
    detection_methods: List[str] = field(default_factory=list)
    control_effectiveness: Dict[str, float] = field(default_factory=dict)

    # Mitigation
    recommended_actions: List[Dict[str, Any]] = field(default_factory=list)
    mitigation_status: str = "Open"
    priority_level: str = "Medium"

    # Root cause analysis
    direct_causes: List[str] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    systemic_causes: List[str] = field(default_factory=list)

    # Effect analysis
    immediate_effects: List[str] = field(default_factory=list)
    intermediate_effects: List[str] = field(default_factory=list)
    long_term_effects: List[str] = field(default_factory=list)

    # Object relationships
    impacted_resources: List[str] = field(default_factory=list)