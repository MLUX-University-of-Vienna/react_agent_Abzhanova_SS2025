import spacy
from .entity_extractor import EntityExtractor

class SpacyEntityExtractor(EntityExtractor):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    