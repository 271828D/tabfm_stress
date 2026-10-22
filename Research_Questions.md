| Goal | Current design OK? |
|------|--------------------|
| "What's the total variance of performance across random splits?" | ✅ Yes — each seed gives a different split + different model state; the spread is your CI |
| "How much variance comes from the *split* vs. the *model*?" | ❌ No — you can't separate them |

- The only model where it might matter is TabFM (400M params, possible dropout at inference) or TabDPT (FAISS retrieval might sample).
- If you want to decouple, the cleanest approach is to use a derived seed for the model in runner.py:
```
for seed in seeds:
    for n_ctx in n_context_list:
        ctx_df, test_df, ... = split_context_test(
            df, subject_col=subject_col,
            n_context=n_ctx, random_state=seed,
        )
        ...
        for name in model_names:
            set_seed(seed + 10_000)  # offset: model seed ≠ split seed
            ...
```

This way:

- Split variance is controlled by seed alone
- Model variance is controlled by seed + 10_000 (deterministic but independent)
- If you want to isolate split effect: run all models with a fixed model seed and vary only the split seed
