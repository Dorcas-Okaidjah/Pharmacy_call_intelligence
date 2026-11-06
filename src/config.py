"""Central configuration. Edit here, not inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"          # data/real/*.json and data/synthetic/*.json
REPORT_DIR = ROOT / "reports"
CACHE_DIR = ROOT / "cache"        # cached LLM responses (re-runs cost nothing)

HF_DATASET = "dataframer/insure-dial"

# Which key in each script turn holds the turn number the annotations refer to.
# Run `python -m src.data --inspect` and check that annotation turn ranges line up
# with this field. Set to None to use the list position (0-based) instead.
TURN_KEY = "turn"

# Outcome fields we extract from each call. These are the facts an operations
# team (and a prior-authorization workflow) needs from a benefits-verification call.
OUTCOME_FIELDS = [
    "record_found",          # was the patient's record located?          True/False
    "coverage_status",       # plan status                                ACTIVE/INACTIVE/NOT_ANSWERED
    "drug1_formulary",       # is drug 1 covered?                         YES/NO
    "drug1_restrictions",    # does drug 1 need PA/step therapy/QL?       YES/NO
    "drug2_formulary",
    "drug2_restrictions",
]

# Turn-level phase labels (for the classical phase classifier).
PHASES = ["ivr", "greeting", "patient_info", "coverage_check",
          "drug_formulary", "drug_restrictions", "drug_copay",
          "agent_interaction", "other"]

# LLM settings. Fill in prices from your provider's pricing page to get cost per call.
LLM_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4.1-mini",
}
PRICE_PER_MTOK = {  # USD per million tokens: {"input": x, "output": y}; None = don't compute cost
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},  # prompts <=100K tokens
    "gpt-4.1-mini": None,
}
