from zenml import pipeline

from steps.etl import crawl_fed_links, get_or_create_fed_info_type


@pipeline
def digitalfed__data_etl(fed_meeting_type: str, links: list[str]) -> str:
    fed_notes_type = get_or_create_fed_info_type()
    last_step = crawl_fed_links(type=fed_notes_type, links=links)

    return last_step.invocation_id
