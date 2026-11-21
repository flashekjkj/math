# vLLM acceptance results

Validation set only; numeric scoring is not proof of reasoning or teaching quality.

| Backend | Confirmed numeric accuracy | Request failures |
|---|---:|---:|
| Transformers | 18.0% | 0 |
| vLLM | 18.0% | 0 |

Per-request evidence and concurrency measurements are in comparison.json. No speedup claim without matched successful samples.

The vLLM container and gateway are left running. The owned Transformers gateway was stopped to release GPU memory.
