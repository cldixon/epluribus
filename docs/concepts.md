# How it works

## A model call is a simulator

epluribus treats an LLM call as a simulator: you can draw samples from it, but you can't ask how likely an answer was. Frontier APIs don't expose the token probabilities you would need, so epluribus never depends on them. Distributions are **empirical**: the share of samples that gave each answer.

Structured outputs make the answers computable. The schema you write is sent to the model as the answer format, and it also decides how each field is summarized.

```
Task(template, schema, system)        Clients: agenerate(prompt, schema, ...) -> Generation
          \                                 /
    run(task, models, n, grid, seed, cache, concurrency, max_retries, ...)
                          |
                ResultSet(schema, [Record, ...])
                          |
   rs["field"]  -> FieldView: freq, mean, ... (by the field's kind)
   rs.group_by(...)["field"] -> one result per group
   rs.filter / usage / summary / to_pandas / arrays
```

## The run plan

For each grid cell, for each template variant, and for each entry in `models`, a run makes `n` calls. An entry is a client or an `ep.Mix`. For a mix, each call's model is drawn by weight.

Each call gets a seed from the run's `seed`, its cell, its model (or mix) and its index. The seed identifies the sample in the cache, and it decides which model a mix draws. So raising `n` keeps every earlier sample, and only the new ones cost anything.

## Uncertainty

Every share comes with a 95% **Wilson** interval, which behaves well with small samples and with shares near 0 or 1. Means and medians come with 95% **percentile bootstrap** intervals. Differences between groups use **Newcombe** intervals for shares and bootstrap intervals for means.

These intervals describe sampling noise: how much the estimate would move if you drew another `n` samples from the same model with the same prompt. They don't describe anything outside that.

## Limits

Keep these in mind when you read results:

- **Sampling variance is not population variance.** Models collapse toward the same answers, defer to the framing, and carry biases. A spread of simulated "voters" is the spread of one model's outputs, not of real voters.
- **Samples cost money, so `n` is small.** Read the intervals before you read a split.
- **Phrasing matters.** A finding that changes with a paraphrase is a finding about the phrasing. Use [template variants](guides/sweeps.md#ask-in-several-phrasings).
- **Stated confidence is not calibrated.** A model's "80% sure" is another answer, not a probability. The spread across samples is the real distribution.
- **Seeds don't make answers reproducible.** Most providers ignore them. The cache is what makes a rerun give the same answers.
- **Refusals and invalid answers are part of the result.** They are recorded, not retried, so they don't silently bias the shares. Check `rs.usage()["error_kinds"]`.

epluribus is for exploring: finding tendencies, disagreements and the conditions that flip an answer. It is not for forecasting what real people will do.
