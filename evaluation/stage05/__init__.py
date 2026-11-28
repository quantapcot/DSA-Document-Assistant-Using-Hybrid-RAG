"""Stage 05 visualization, comparative analysis, validation, and manifest package."""

from __future__ import annotations

COMBO_ORDER = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4")
ALL_GEN_ORDER = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "NO_RAG")
KS = (1, 3, 5, 10, 15, 20, 30, 45, 60)

COMBO_LABELS = {
    "A1": "A1: Recursive + BM25",
    "A2": "A2: Recursive + Dense",
    "A3": "A3: Recursive + Hybrid RRF",
    "A4": "A4: Recursive + Hybrid Rerank",
    "B1": "B1: Adaptive + BM25",
    "B2": "B2: Adaptive + Dense",
    "B3": "B3: Adaptive + Hybrid RRF",
    "B4": "B4: Adaptive + Hybrid Rerank",
    "NO_RAG": "NO_RAG: Baseline (No Context)",
}

COMBO_SHORT = {
    "A1": "A1\n(Rec+BM25)",
    "A2": "A2\n(Rec+Dense)",
    "A3": "A3\n(Rec+RRF)",
    "A4": "A4\n(Rec+Rerank)",
    "B1": "B1\n(Ada+BM25)",
    "B2": "B2\n(Ada+Dense)",
    "B3": "B3\n(Ada+RRF)",
    "B4": "B4\n(Ada+Rerank)",
    "NO_RAG": "NO_RAG\n(Baseline)",
}

PROFILE_LABELS = {
    "recursive_512_100": "Profile A: recursive_512_100",
    "structure_adaptive": "Profile B: structure_adaptive",
}
