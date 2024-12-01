import re

BASE_MODEL_NAME = 'haoranxu/ALMA-7B-R'


def make_prompt(ru: str, faculty: str, direction: str):
    prompt = f'<Faculty>: {faculty}\n<Direction>: {direction}\nTranslate <Russian> to <English>.\n<Russian>: {ru}\n<English>:'
    return prompt


def clean_text(text: str):
    text = text.strip()
    text = text.replace('\n', '').replace('\r', '')
    text = re.sub(r'[\s\u200b]+', ' ', text)
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text