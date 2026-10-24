import re


def clean_text(text: str) -> str:
    text = re.sub(r"\s*\n\s*", "\n", text)  # any run of newlines, plus spaces around it -> one \n
    text = re.sub(r"[^\S\n]+", " ", text)   # other whitespace (tabs, double spaces, \xa0) -> one space

    return text.strip()

