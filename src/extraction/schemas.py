"""
Schemas for LLM-extracted claim facts.

These define the structured output the extraction pipeline produces.
The handler sees these as an editable "fact card" in the UI.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ExtractedField(BaseModel):
    """A single extracted field with confidence and source span."""
    value: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    source_snippet: Optional[str] = Field(
        None,
        description="The exact text span from the description that supports this extraction",
    )


class IncidentType(str, Enum):
    COLLISION = "collision"
    SINGLE_VEHICLE = "single_vehicle"
    HIT_AND_RUN = "hit_and_run"
    PARKING = "parking"
    THEFT = "theft"
    VANDALISM = "vandalism"
    WEATHER = "weather"
    UNKNOWN = "unknown"


class InjurySeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class ExtractedFacts(BaseModel):
    """Structured facts extracted from a claim description."""

    # Core incident info
    incident_date: ExtractedField = Field(description="Date of the incident (DD/MM/YYYY)")
    incident_time: ExtractedField = Field(description="Time of the incident (HH:MM)")
    incident_location: ExtractedField = Field(description="Street/road where it happened")
    incident_city: ExtractedField = Field(description="City where it happened")
    incident_type: IncidentType = IncidentType.UNKNOWN

    # Parties
    num_parties: ExtractedField = Field(description="Number of parties/vehicles involved")
    other_vehicle: ExtractedField = Field(description="Make/model of other vehicle(s)")

    # Damage
    damage_description: ExtractedField = Field(description="Description of damage sustained")
    damage_areas: ExtractedField = Field(description="Parts of vehicle damaged")

    # Injuries
    injuries_reported: bool = False
    injury_severity: InjurySeverity = InjurySeverity.UNKNOWN
    injury_details: ExtractedField = Field(
        default_factory=lambda: ExtractedField(),
        description="Details about injuries",
    )

    # Documentation
    police_report_mentioned: ExtractedField = Field(
        description="Whether a police report was filed",
    )
    police_report_number: ExtractedField = Field(
        default_factory=lambda: ExtractedField(),
        description="Police report reference number if mentioned",
    )
    witnesses_mentioned: ExtractedField = Field(
        default_factory=lambda: ExtractedField(),
        description="Whether witnesses are mentioned",
    )


class MissingInfo(BaseModel):
    """Information that appears to be missing from the claim."""
    field: str = Field(description="What's missing (e.g. 'incident_time', 'police_report')")
    importance: str = Field(description="high / medium / low")
    reason: str = Field(description="Why this matters for processing")


class ExtractionResult(BaseModel):
    """Complete output of the extraction pipeline for one claim."""
    claim_id: str
    summary: str = Field(
        description="Concise 2-4 sentence summary of the claim for the handler",
    )
    facts: ExtractedFacts
    missing_info: list[MissingInfo] = Field(
        default_factory=list,
        description="List of important information not found in the description",
    )
    language: str = Field(
        default="en",
        description="Detected language of the description (en/it)",
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Any concerns about the description (inconsistencies, vagueness)",
    )
