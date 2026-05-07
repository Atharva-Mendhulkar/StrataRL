# training/domain_guard.py

def assert_batch_domain_homogeneity(domains: list):
    """
    All prompts in a batch must share the same domain.
    SAN requires >= 2 prompts per domain to compute Z-statistics.
    Mixed-domain batches produce zero advantages for singleton domains,
    wasting the entire training step.
    """
    unique = set(domains)
    assert len(unique) == 1, (
        f"DOMAIN HETEROGENEITY: batch contains {len(unique)} domains: {unique}. "
        f"UCB scheduler must sample one domain per step, then fill the batch from it."
    )
