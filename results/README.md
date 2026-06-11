
# Results
---

Evaluated on 1,000 sentences from [`gplsi/ES-VA_translation_test`](https://huggingface.co/datasets/gplsi/ES-VA_translation_test).

### Automatic MT Quality

| System | chrF ↑ | BLEU ↑ | TER ↓ | BLEURT ↑ | COMET ↑ |
|---|---|---|---|---|---|
| Baseline | 69.02 | 39.22 | 40.30 | 0.258 | 0.906 |
| SFT | 83.16 | 60.16 | 22.80 | 0.524 | 0.934 |
| GRPOv1 | 81.65 | 56.94 | 23.96 | 0.481 | 0.926 |
| **GRPOv2** ★ | **84.68** | **62.16** | **20.63** | **0.544** | **0.936** |

### Dialectal Valencian Score (DVS)

DVS measures the rate at which a system produces Valencian-specific morpho-lexical forms over 35 contrastive CA–VA pairs.

| System | DVS ↑ |
|---|---|
| Baseline | 3.2% |
| **SFT** | **41.0%** |
| GRPOv1 | 15.9% |
| GRPOv2 ★ | 36.2% |

### Feature-level DVS (selected pairs)

| CA form | VA form | Baseline | SFT | GRPOv1 | GRPOv2 |
|---|---|---|---|---|---|
| *petit* | *xicotet* | 0% | 100% | 0% | 100% |
| *veure* | *vore* | 0% | 83% | 0% | 17% |
| *avui* | *hui* | 0% | 71% | 0% | 86% |
| *seva* | *seua* | 0% | 100% | 45% | 100% |

---

© 2026 The authors. No derivative works permitted.