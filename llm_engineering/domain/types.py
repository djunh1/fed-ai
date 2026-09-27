from enum import StrEnum


class DataCategory(StrEnum):
    PROMPT = "prompt"
    QUERIES = "queries"

    INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"
    INSTRUCT_DATASET = "instruct_dataset"
    PREFERENCE_DATASET_SAMPLES = "preference_dataset_samples"
    PREFERENCE_DATASET = "preference_dataset"

    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"


class FedInfoType(StrEnum):
    MINUTES = "minutes"
    STATEMENT = "statement"
    TRANSCRIPT = "transcript"
    MEMORANDA_OF_DISCUSSION = "memoranda_of_discussion"
    BEIGE_BOOK = "beige_book"
    BRIEFING_BOOK = "briefing_book"
    UNKNOWN = "unknown"


class SourceSystem(StrEnum):
    FED_GOV = "fed_gov"
    FRASER = "fraser"
