"""Ranking metrics over explicitly supplied relevance labels."""
def precision_at_k(retrieved, relevant, k=5):
    if k <= 0: raise ValueError("k must be positive")
    return len(set(retrieved[:k]) & set(relevant)) / k

def recall_at_k(retrieved, relevant, k=5):
    if k <= 0: raise ValueError("k must be positive")
    if not relevant: raise ValueError("ground truth must contain at least one relevant chunk")
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))

def reciprocal_rank(retrieved, relevant):
    relevant = set(relevant)
    return next((1 / i for i, item in enumerate(retrieved, 1) if item in relevant), 0.0)
