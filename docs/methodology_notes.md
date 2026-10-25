# Reproduction vs reconstruction

The article is authoritative for claims about the *published* study; recovered
notebooks are evidence for code-level choices. The repository is a documented
reconstruction, **not** a verified reproduction of publication-level metrics.

## Original methodological decisions reported in the article

- Three checkpoints: Spanish BETO, multilingual XLM-RoBERTa, Spanish biomedical RoBERTa.
- Two text representations: complete EHR notes and MedSpaCy-derived clinical entities.
- Patient visits ordered chronologically; end at the last positive visit for cases,
  otherwise at the last available visit; preserve recent history within 512 tokens.
- Two-logit classifier with fine-tuning; dropout, batch size and learning rate tuned
  via grid search using BETO; other checkpoints use selected configurations.
- 80% development / 20% test, 80% train / 20% validation within development.
- Article describes downsampling *before* data partitioning to about 70:30.
- F1, ROC AUC, PR AUC reported in the paper.

## Explicit reconstruction / correction choices

1. The original notebooks are partial and internally inconsistent. The epoch count
   in the YAML is a **reconstruction default**, not a claim about the paper.
2. Chronological strings are tokenized completely, then left-truncated to keep the
   newest model-specific tokens, with model-specific special tokens added after.
   Text may begin mid-visit. Tokenization of the concatenation differs from encoding
   each visit independently. Original boundary handling cannot be fully verified.
3. A 2-logit linear head over BETO's pooler resembles recovered code. For RoBERTa,
   the first hidden token is used (architectural adaptation). It is not established
   whether this exactly matches each paper-era checkpoint implementation.
4. Downsampling occurs **only on training** after a patient-disjoint split, to
   avoid changing held-out population prevalence. This intentionally differs from
   the article's described ordering, so metrics are not expected to match exactly.
5. Validation ROC AUC chooses best epochs and Grid Search trials. It is a
   reconstruction selection rule; the original stopping/selection rule is unknown.
6. `average_precision` is computed as a practical PR metric; the publication's
   PR AUC numerical definition is not specified, and AP != trapezoidal PR AUC.
7. No clinical Spanish entity-recognition dictionary, MedSpaCy version or complete
   historical ontology was recovered. Provide an audited entities_text input to
   reproduce the entities representation. Default MedSpaCy rules are NOT comparable.
8. The positive-class endpoint may include a documented apnea diagnosis; the study
   must not be represented as proven diagnosis-free early prospective detection.
9. Patient-linked test predictions are PRIVATE, and must never be committed.
10. No dataset or original trained checkpoint is included. Published results are
    historical reports, not metrics validated by executing this repository.

## Questions for an eventual exact reproduction

- Training epochs, early stopping and exact validation-selection criterion.
- Exact note clean-up and MedSpaCy Spanish extraction rules.
- Token boundary conventions / partial-visit truncation.
- Matching Hugging Face checkpoint revisions and original package versions.
- Any intermediate data curation not recoverable from the notebooks.
- Exact PR AUC computation and sampling procedure.
