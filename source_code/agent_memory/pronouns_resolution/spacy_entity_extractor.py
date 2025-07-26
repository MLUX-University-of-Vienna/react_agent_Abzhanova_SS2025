import spacy
from .entity_extractor import EntityExtractor
from typing import Optional

class SpacyEntityExtractor(EntityExtractor):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str) -> Optional[str]:
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG"}:
                return ent.text.strip()
        return None