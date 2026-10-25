"""Optional MedSpaCy entity representation.

IMPORTANT: MedSpaCy default rules are not a guaranteed reproduction of the
historical project's Spanish-specific patterns / ontology setup. An audited
precomputed entities_text field is preferred for strict reproduction.
"""
def extract_entities(text, nlp):
    doc = nlp(str(text))
    return ' '.join(ent.text for ent in doc.ents)


def get_medspacy_pipeline():
    import medspacy
    return medspacy.load()
