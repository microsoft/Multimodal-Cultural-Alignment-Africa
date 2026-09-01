"""Keyword-based cultural marker classifier.

Classifies text spans into cultural marker categories using keyword matching.
This is deterministic and model-independent — no LLM needed.

Usage:
    from src.classifiers import classify_span, classify_spans

    category = classify_span("umqombothi")  # → "food_dietary_practices"
    markers  = classify_spans("Thandi wore her isidwaba skirt")
               # → [CulturalMarker(category="clothing_fashion", span="isidwaba skirt")]
"""

from __future__ import annotations

import re
from src.schemas.models import CulturalMarker, MARKER_CATEGORIES

# ---------------------------------------------------------------------------
# Keyword → category mapping
# Each category has a list of keyword patterns (lowercase, regex-safe).
# Patterns are matched against the lowercased span text.
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "people_appearance": [
        "skin", "hair", "complexion", "tattoo", "scarification", "scar",
        "body paint", "piercing", "beard", "bald", "wrinkle", "elderly",
        "young", "muscular", "slim", "dark-skinned", "light-skinned",
    ],
    "names_forms_of_address": [
        "malam", "mzee", "baba", "mama", "gogo", "nana", "uncle", "auntie",
        "chief", "elder", "mr", "mrs", "oga", "madam", "sheikh", "imam",
        "pastor", "reverend", "nkosi", "inkosi", "tata", "sisi", "bhuti",
        "ntate", "mme", "rra", "mma", "alhaji", "hajia",
    ],
    "language_local_expression": [
        "ina kwana", "sannu", "salaam", "molo", "sawubona", "dumelang",
        "habari", "jambo", "asante", "shukran", "ndewo", "ekaro", "ekaale",
        "slang", "proverb", "saying", "idiom", "greeting", "mos", "lekker",
        "ag ", "eish", "yoh", "haibo", "code-switch", "mother tongue",
    ],
    "food_dietary_practices": [
        "tuwo", "shinkafa", "fufu", "ugali", "sadza", "pap", "mealie",
        "umqombothi", "jollof", "suya", "biltong", "braii", "braai",
        "injera", "chapati", "mandazi", "samosa", "kolanut", "kola nut",
        "palm wine", "sorghum", "millet", "maize", "cassava", "yam",
        "plantain", "groundnut", "moringa", "baobab", "dates", "halal",
        "kosher", "fasting", "feast", "cook", "brew", "recipe", "spice",
        "stew", "porridge", "meal", "dish", "food", "eat", "drink",
        "kitchen", "pot", "calabash",
    ],
    "place_physical_environment": [
        "village", "compound", "kraal", "homestead", "township", "province",
        "savanna", "bush", "desert", "coast", "river", "mountain", "hill",
        "market", "square", "clinic", "hospital", "field", "farm", "garden",
        "street", "road", "path", "kano", "lagos", "nairobi", "cape town",
        "maputo", "accra", "dakar", "eastern cape", "limpopo", "gauteng",
        "dusty", "sandy", "muddy", "thatched", "mud house", "rondavel",
    ],
    "healthcare_community_practices": [
        "clinic", "hospital", "nurse", "doctor", "healer", "herbalist",
        "midwife", "traditional medicine", "health worker", "vaccination",
        "prenatal", "antenatal", "postnatal", "sangoma", "inyanga",
    ],
    "socio_economic_context": [
        "budget", "afford", "price", "cost", "cheap", "expensive", "poverty",
        "wealth", "income", "salary", "wage", "market", "shop", "vendor",
        "hustle", "livelihood", "subsistence",
    ],
    "family_household_structure": [
        "mother-in-law", "father-in-law", "grandmother", "grandfather",
        "auntie", "uncle", "cousin", "sibling", "brother", "sister",
        "wife", "husband", "family", "household", "compound", "extended",
        "clan", "lineage", "firstborn", "lobola", "bride price", "dowry",
        "in-law", "kindred",
    ],
    "occupation_daily_routine": [
        "farmer", "teacher", "nurse", "trader", "vendor", "herder",
        "fisherman", "artisan", "weaver", "potter", "blacksmith",
        "carpenter", "driver", "office", "work", "job", "commute",
        "morning routine", "lunch", "meeting",
    ],
    "health_values_beliefs": [
        "healthy", "wellness", "balance", "moderation", "exercise",
        "prevention", "cure", "belief", "attitude", "mindset",
    ],
    "clothing_fashion": [
        "hijab", "headscarf", "wrapper", "ankara", "kente", "dashiki",
        "agbada", "buba", "sokoto", "gele", "isidwaba", "isikhakha",
        "shweshwe", "beaded", "necklace", "bracelet", "bangle", "earring",
        "turban", "veil", "loincloth", "skirt", "robe", "garment",
        "traditional dress", "attire", "jewelry", "jewellery",
    ],
    "social_roles_hierarchy": [
        "elder", "chief", "king", "queen", "council", "headman",
        "authority", "respect", "seniority", "age-grade", "rank",
        "patron", "leader", "ruler", "obedience",
    ],
    "community_social_networks": [
        "neighbour", "neighbor", "village", "communal", "mutual aid",
        "solidarity", "stokvel", "cooperative", "assembly", "gathering",
        "community", "collective", "together",
    ],
    "gender_norms_expectations": [
        "women", "men", "gender", "masculine", "feminine", "bride",
        "groom", "wife", "husband", "patriarchal", "matriarchal",
        "dowry", "bridewealth",
    ],
    "life_stage_transitions": [
        "birth", "naming", "circumcision", "initiation", "coming of age",
        "marriage", "wedding", "funeral", "burial", "death", "puberty",
        "pregnancy", "childbirth", "weaning", "baptism",
    ],
    "temporal_orientation": [
        "punctual", "late", "early", "time", "clock", "sunrise", "sunset",
        "season", "harvest", "planting",
    ],
    "seasonality_environmental_cycles": [
        "rainy season", "dry season", "harmattan", "monsoon", "harvest",
        "planting season", "drought", "flood",
    ],
    "daily_rhythms_meal_patterns": [
        "breakfast", "lunch", "dinner", "supper", "morning meal",
        "afternoon", "evening meal", "tea time", "snack",
    ],
    "religious_spiritual_practices": [
        "prayer", "mosque", "church", "temple", "shrine", "altar",
        "quran", "bible", "imam", "pastor", "priest", "ritual",
        "sacrifice", "offering", "spirit", "ancestor", "god", "allah",
        "worship", "faith", "blessing", "holy", "sacred",
    ],
    "traditional_indigenous_knowledge": [
        "herbal", "remedy", "traditional", "indigenous", "folk medicine",
        "divination", "oracle", "ancestral", "herbal medicine",
    ],
    "health_illness_explanatory_models": [
        "curse", "evil eye", "witchcraft", "possession", "spiritual illness",
        "diagnosis", "symptom", "disease", "affliction",
    ],
    "values_moral_frameworks": [
        "ubuntu", "respect", "honor", "honesty", "integrity", "shame",
        "dignity", "duty", "obligation", "virtue", "moral",
    ],
    "technology_media_use": [
        "phone", "mobile", "radio", "television", "internet", "social media",
        "whatsapp", "facebook", "data", "airtime", "app",
    ],
    "transport_mobility": [
        "bus", "taxi", "matatu", "okada", "boda boda", "bicycle", "car",
        "walk", "commute", "road", "transport", "travel",
    ],
    "household_resources_tools": [
        "pot", "clay pot", "mortar", "pestle", "stove", "fire",
        "water", "well", "bucket", "electricity", "solar", "calabash",
        "grinding stone", "broom",
    ],
    "communication_style": [
        "proverb", "indirect", "polite", "formal", "informal",
        "storytelling", "silence", "gesture", "tone", "humor",
    ],
    "emotional_expression_norms": [
        "ululate", "wail", "mourn", "celebrate", "joy", "grief",
        "stoic", "restrained", "express", "emotion",
    ],
    "storytelling_narrative_forms": [
        "folktale", "legend", "myth", "oral tradition", "griot",
        "story", "narrative", "parable", "fable", "tale",
    ],
    "social_norms_taboo": [
        "taboo", "forbidden", "prohibit", "avoid", "shame",
        "improper", "unacceptable", "custom", "norm",
    ],
    "institutional_trust_relations": [
        "government", "police", "court", "ngo", "chief", "trust",
        "corruption", "authority", "institution",
    ],
    "silence_implicit_knowledge": [
        "unspoken", "implied", "assumed", "silence", "omit",
        "left unsaid", "implicit",
    ],
    "animals_human_interaction": [
        "cattle", "goat", "sheep", "chicken", "cow", "ox",
        "donkey", "camel", "livestock", "pet", "dog", "cat",
        "herd", "flock", "stable",
    ],
    "plants_agriculture_food_sources": [
        "crop", "harvest", "plant", "seed", "farm", "field",
        "garden", "maize", "sorghum", "millet", "cassava", "yam",
        "cocoa", "coffee", "tea", "tobacco",
    ],
    "sports_recreation_physical_activity": [
        "football", "soccer", "wrestling", "game", "sport", "play",
        "competition", "race", "dance", "drumming",
    ],
    "arts_entertainment_culture": [
        "music", "dance", "drum", "song", "sing", "art", "craft",
        "weave", "carve", "sculpture", "painting", "festival",
        "performance", "entertainment", "talking drum",
    ],
    "media_information_ecosystem": [
        "newspaper", "radio", "broadcast", "news", "journalist",
        "misinformation", "rumor", "information",
    ],
    "built_environment_infrastructure": [
        "road", "bridge", "building", "school", "hospital",
        "well", "sanitation", "toilet", "electricity", "plumbing",
    ],
    "objects_material_culture": [
        "beaded", "bead", "necklace", "bracelet", "staff", "stool",
        "drum", "mask", "sculpture", "pottery", "basket", "mat",
        "cloth", "calabash", "artifact",
    ],
    "symbols_visual_culture": [
        "flag", "emblem", "totem", "crest", "colour", "color",
        "pattern", "motif", "symbol", "icon", "insignia",
    ],
    "events_rituals_celebrations": [
        "ceremony", "ritual", "festival", "celebration", "wedding",
        "funeral", "naming", "initiation", "coronation", "feast",
        "umembeso", "lobola", "suna", "aqeeqah",
    ],
    "education_learning_practices": [
        "school", "teacher", "student", "learn", "educate",
        "literacy", "quran school", "apprentice", "training",
    ],
    "governance_politics_civic_life": [
        "chief", "king", "council", "vote", "election", "government",
        "law", "regulation", "civic", "political",
    ],
    "economy_livelihood_strategies": [
        "trade", "market", "barter", "income", "livelihood",
        "subsistence", "commerce", "merchant", "vendor",
    ],
    "environment_ecology_sustainability": [
        "conservation", "deforestation", "pollution", "climate",
        "environment", "ecology", "wildlife", "erosion",
    ],
    "risk_safety_security": [
        "crime", "danger", "safety", "security", "theft",
        "violence", "conflict", "peace", "curfew",
    ],
    "migration_mobility_patterns": [
        "migrate", "diaspora", "rural", "urban", "move",
        "displaced", "refugee", "nomad", "seasonal migration",
    ],
    "identity_ethnicity_culture": [
        "ethnic", "tribe", "clan", "heritage", "identity",
        "tradition", "custom", "ancestry", "lineage", "culture",
    ],
    "disability_accessibility": [
        "disability", "blind", "deaf", "wheelchair", "accessible",
        "inclusion", "barrier",
    ],
    "food_markets_distribution": [
        "market", "vendor", "stall", "hawker", "supply",
        "distribution", "food access", "groceries",
    ],
}

# Pre-compile patterns per category for performance
_COMPILED: dict[str, re.Pattern] = {}
for _cat, _keywords in CATEGORY_KEYWORDS.items():
    # Use word boundaries to avoid substring matches (e.g. "king" in "talking")
    escaped = [r"\b" + re.escape(k) + r"\b" for k in _keywords]
    _COMPILED[_cat] = re.compile(r"(?:" + "|".join(escaped) + r")", re.IGNORECASE)


def classify_span(span: str) -> str:
    """Classify a single text span into its best-matching cultural marker category.

    Returns the category key with the most keyword hits, or "unclassified".
    """
    text = span.lower().strip()
    best_cat = "unclassified"
    best_count = 0

    for cat, pattern in _COMPILED.items():
        hits = len(pattern.findall(text))
        if hits > best_count:
            best_count = hits
            best_cat = cat

    return best_cat


def classify_span_multi(span: str, threshold: int = 1) -> list[str]:
    """Return all matching categories for a span (sorted by hit count, descending)."""
    text = span.lower().strip()
    matches: list[tuple[str, int]] = []

    for cat, pattern in _COMPILED.items():
        hits = len(pattern.findall(text))
        if hits >= threshold:
            matches.append((cat, hits))

    matches.sort(key=lambda x: x[1], reverse=True)
    return [cat for cat, _ in matches]


def classify_spans(text: str) -> list[CulturalMarker]:
    """Scan a full text and return all cultural markers found via keyword matching.

    Searches for every keyword in the text and returns deduplicated markers.
    """
    found: list[CulturalMarker] = []
    seen: set[tuple[str, str]] = set()

    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                span = match.group()
                # Expand span to include a bit of surrounding context
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].strip()
                key = (cat, span.lower())
                if key not in seen:
                    seen.add(key)
                    found.append(CulturalMarker(category=cat, span=context))

    return found


def reclassify_markers(markers: list[CulturalMarker]) -> list[CulturalMarker]:
    """Re-classify markers using keyword matching.

    Takes LLM-produced markers (which may have wrong/invented categories)
    and assigns the correct category based on keyword matching on the span.
    Keeps the original span text.
    """
    result = []
    for m in markers:
        new_cat = classify_span(m.span)
        # If keyword classifier found a match, use it; otherwise keep LLM's category
        # (but only if it's a valid category key)
        if new_cat != "unclassified":
            result.append(CulturalMarker(category=new_cat, span=m.span))
        elif m.category in MARKER_CATEGORIES:
            result.append(m)
        else:
            result.append(CulturalMarker(category="unclassified", span=m.span))
    return result
