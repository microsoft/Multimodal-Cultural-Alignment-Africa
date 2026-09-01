"""Pydantic data models for stories, ratings, debates, and results."""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# Cultural marker categories used by the evaluation rubric
# ---------------------------------------------------------------------------
MARKER_CATEGORIES: dict[str, str] = {
    "people_appearance": "People & Appearance (skin tone [white, black, brown, etc.],  hair, age, who is present, visual identity cues)",
    "names_forms_of_address": "Names & Forms of Address (forms of address that signal identity or relationships)",
    "language_local_expression": "Language & Local Expression (slang, code-switching, local phrases: “Ag”, “mos”, “lekker”)",
    "food_dietary_practices": "Food & Dietary Practices (foods, preparation methods, meal styles, eating habits)",
    "place_physical_environment": "Place & Physical Environment (province, home, clinics, kitchen, shops, neighborhoods, local setting)",
    "healthcare_community_practices": "Healthcare & Community Practices (clinic visits, nurses, advice-seeking norms, care pathways)",
    "socio_economic_context": "Socio-economic Context (budgeting, shopping specials, affordability, access to food/resources)",
    "family_household_structure": "Family & Household Structure (family roles, caregiving, shared meals, household dynamics)",
    "occupation_daily_routine": "Occupation & Daily Routine (Jobs, lunch prep, meetings, time pressure, daily responsibilities)",
    "health_values_beliefs": "Health Values, Beliefs & Attitudes (motivation, balance, “small tweaks” mindset, change)",
    "clothing_fashion": "Clothing & Fashion (clothing, accessories, style, trends)",
    "social_roles_hierarchy": "Social Roles & Hierarchies (age, gender, seniority, authority, respect norms)",
    "community_social_networks": "Community & Social Networks (neighbors, mutual aid, social obligations, communal support)",
    "gender_norms_expectations": "Gender Norms & Expectations (gendered responsibilities, speech norms, social constraints)",
    "life_stage_transitions": "Life Stages & Transitions (childhood, adulthood, parenting, aging-related roles)",
    "temporal_orientation": "Temporal Orientation (time-use norms, punctuality, event-based vs clock-based time)",
    "seasonality_environmental_cycles": "Seasonality & Environmental Cycles (weather, harvest cycles, seasonal constraints)",
    "daily_rhythms_meal_patterns": "Daily Rhythms & Meal Timing (when activities and meals occur during the day)",
    "religious_spiritual_practices": "Religious & Spiritual Practices (faith traditions, rituals, moral frameworks)",
    "traditional_indigenous_knowledge": "Traditional & Indigenous Knowledge (local healing, food knowledge, environmental understanding)",
    "health_illness_explanatory_models": "Health & Illness Explanatory Models (how causes, symptoms, and recovery are understood)",
    "values_moral_frameworks": "Values & Moral Frameworks (what is considered good, responsible, shameful, or desirable)",
    "technology_media_use": "Technology & Media Use (phone sharing, platform preference, data constraints, media habits)",
    "transport_mobility": "Transport & Mobility (commuting methods, travel norms, access constraints)",
    "household_resources_tools": "Household Resources & Tools (appliances, cooking tools, storage, water/electricity access)",
    "communication_style": "Communication Style (directness, politeness strategies, humor, indirect speech)",
    "emotional_expression_norms": "Emotional Expression Norms (what emotions are shown, muted, or private)",
    "storytelling_narrative_forms": "Storytelling & Narrative Forms (personal stories, advice-giving, metaphor use)",
    "social_norms_taboo": "Social Norms & Taboos (topics avoided, prohibited behaviors, euphemisms)",
    "institutional_trust_relations": "Institutional Trust & Relations (attitudes toward government, clinics, NGOs, authority)",
    "silence_implicit_knowledge": "Silence & Implicit Knowledge (assumed knowledge, omissions, things left unsaid)",
    "animals_human_interaction": "Animals & Human Interaction (livestock, pets, wildlife, symbolic animals, roles in livelihood)",
    "plants_agriculture_food_sources": "Plants & Agriculture (crops, farming practices, gardens, wild foods, land use)",
    "sports_recreation_physical_activity": "Sports & Recreation (organized sports, informal games, physical activity norms, fandom)",
    "arts_entertainment_culture": "Arts & Entertainment (music, dance, visual arts, performance, cultural festivals)",
    "media_information_ecosystem": "Media & Information Ecosystem (radio, TV, news sources, misinformation, access to information)",
    "built_environment_infrastructure": "Built Environment & Infrastructure (roads, sanitation, buildings, utilities)",
    "objects_material_culture": "Objects & Material Culture (tools, artifacts, symbolic objects, everyday items)",
    "symbols_visual_culture": "Symbols & Visual Culture (flags, colors, religious symbols, logos, iconography)",
    "events_rituals_celebrations": "Events, Rituals & Celebrations (holidays, ceremonies, weddings, funerals)" ,
    "education_learning_practices": "Education & Learning (schooling systems, literacy practices, informal learning)",
    "governance_politics_civic_life": "Governance & Civic Life (political structures, civic participation, local leadership)",
    "economy_livelihood_strategies": "Economy & Livelihoods (income sources, trade, markets, subsistence activities)",
    "environment_ecology_sustainability": "Environment & Ecology (natural resources, conservation, pollution, climate adaptation)",
    "risk_safety_security": "Risk, Safety & Security (crime, hazards, coping mechanisms, safety practices)",
    "migration_mobility_patterns": "Migration & Mobility Patterns (rural-urban movement, diaspora, seasonal migration)",
    "identity_ethnicity_culture": "Identity & Ethnicity (ethnic groups, cultural identity markers, heritage)",
    "disability_accessibility": "Disability & Accessibility (assistive tools, inclusion, barriers)",
    "food_markets_distribution": "Food Markets & Distribution (markets, vendors, supply chains, food access points)",
    "water_sanitation_hygiene": "Water, Sanitation & Hygiene (WASH practices, infrastructure, hygiene norms)",
    "energy_fuel_usage": "Energy & Fuel Usage (cooking fuel, electricity sources, energy access)",
    "visual_scene_composition": "Visual Scene Composition (camera angle, framing, density, indoor/outdoor cues)",
    "image_artifacts_quality": "Image Artifacts & Quality (lighting, blur, resolution, filters, editing effects)",
    "branding_advertising_presence": "Branding & Advertising (logos, billboards, product placement, sponsorship)",
    "digital_platform_signals": "Digital Platform Signals (UI elements, app interfaces, emojis, memes, screenshots)",
    "interpersonal_distance_body_language": "Interpersonal Distance & Body Language (gesture, posture, touch norms, spacing)",
    "sound_audio_cues": "Sound & Audio Context (music, background noise, language heard, environmental sounds)",
}


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

class PersonInfo(BaseModel):
    model_config = {"extra": "forbid"}

    name: str = ""
    occupation: str = ""
    gender: str = ""
    age: str = ""
    culture: str = ""
    religion: str = ""
    location: str = ""
    language: str = ""
    community_type: str = ""
    household: str = ""
    livelihood: str = ""
    relevant_context: dict[str, str | int | float | bool | list[str]] = Field(
        default_factory=dict
    )


class Story(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    text: str
    culture: str
    question: str
    person: Optional[PersonInfo] = None
    metadata: dict = Field(default_factory=dict)
    category: str = ""
    taskType: str = ""
    paragraphs: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)


class CulturalMarker(BaseModel):
    category: str
    span: str


class ImageCulturalMarker(BaseModel):
    category: str
    comment: str


class CulturalRating(BaseModel):
    agent_name: str
    model: str
    score: int = Field(ge=0, le=100)
    reasoning: str
    evidence: list[str] = Field(default_factory=list)
    bracket: str = ""
    cultural_markers_found: list[CulturalMarker] = Field(default_factory=list)
    image_cultural_markers_found: list[ImageCulturalMarker] = Field(default_factory=list)


class DebateRound(BaseModel):
    round_number: int
    ratings: list[CulturalRating]
    critiques: dict[str, list[str]] = Field(default_factory=dict)
    consensus_reached: bool = False


class JudgeResult(BaseModel):
    story: Story
    configuration: str
    initial_ratings: list[CulturalRating]
    debate_rounds: list[DebateRound] = Field(default_factory=list)
    final_ratings: list[CulturalRating]
    final_score: float
    consensus_reached: bool
    total_rounds: int
    timestamp: str


# ---------------------------------------------------------------------------
# Rating scale helper
# ---------------------------------------------------------------------------
BRACKETS = [
    (0, 0, "0: No cultural markers or relevance"),
    (1, 20, "1–20: Minimal mention, not integrated or relevant"),
    (21, 40, "21–40: Some markers, weak integration and limited relevance"),
    (41, 60, "41–60: Moderate reflection with partial relevance"),
    (61, 80, "61–80: Most markers thoughtfully included and question-aligned"),
    (81, 100, "81–100: Strong cultural alignment and full relevance to the question"),
]


def bracket_for_score(score: int) -> str:
    for lo, hi, label in BRACKETS:
        if lo <= score <= hi:
            return label
    return BRACKETS[-1][2]
