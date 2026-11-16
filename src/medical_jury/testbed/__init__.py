"""Chronic-disease known-answer testbed (plan phase 3, PROTOCOL-V3-TESTBED).

Synthetic patients -> conservative reference plan -> one injected, catalogued error -> prose written by a local model
-> deterministic verification that the error (and nothing else from the drug lexicon) is really in the prose.
Nothing here is clinical guidance: the rules are a small, cited, project-written subset and are not physician-reviewed.
"""
TESTBED_VERSION = 'chronic-v2'   # v1 (100 patients, no split) was only used for a 6-case writer trial
