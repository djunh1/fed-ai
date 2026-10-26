from enum import StrEnum


class DataCategory(StrEnum):
    PROMPT = "prompt"
    QUERIES = "queries"

    INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"
    INSTRUCT_DATASET = "instruct_dataset"
    PREFERENCE_DATASET_SAMPLES = "preference_dataset_samples"
    PREFERENCE_DATASET = "preference_dataset"

    FED_DOCUMENTS = "fed_documents"
    FED_TEXT = "fed_text"


class FedInfoType(StrEnum):
    MINUTES = "minutes"
    STATEMENT = "statement"
    TRANSCRIPT = "transcript"
    MEMORANDA_OF_DISCUSSION = "memoranda_of_discussion"
    MINUTES_OF_ACTIONS = "minutes_of_actions"
    BEIGE_BOOK = "beige_book"
    TEAL_BOOK = "teal_book"
    BRIEFING_BOOK = "briefing_book"
    RECORD_OF_POLICY_ACTIONS = "record_of_policy_actions"
    PRESS_CONFERENCE = "press_conference"
    UNKNOWN = "unknown"


class SourceSystem(StrEnum):
    FED_GOV = "fed_gov"
    FRASER = "fraser"
