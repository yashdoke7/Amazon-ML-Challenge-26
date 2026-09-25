# Local validation protocol

_Frozen 25 September 2026, before fitting a matcher._

`code/business_entity_resolution/src/validation.py` hashes each Source 1 ID with a fixed seed into disjoint development (1%), validation (4%), and training (95%) partitions. This grouping keeps every labeled Source 2/3 positive with its Source 1 owner. The split is deterministic and independent of names, addresses, candidate scores, and labels. It remains fixed while methods change.

The full train target corpus may be indexed to simulate the retrieval problem. When training a matcher or choosing negatives, remove **every target owned by a development or validation Source 1** from the training pair labels. Otherwise a held-out positive could appear as a training negative for another query. Fit learned text statistics, calibration, thresholds, and hyperparameters using only the training and development partitions; evaluate validation only for final model selection. Never train on test labels, which do not exist.

The audit found:

| Split | India S1 | US S1 | True links | Singletons |
| --- | ---: | ---: | ---: | ---: |
| Development | 8,824 | 13,309 | 76,472 | 1,272 |
| Validation | 35,605 | 52,939 | 306,411 | 4,844 |
| Training | 838,759 | 1,257,385 | 7,255,482 | 117,131 |

Use development for quick iteration and threshold selection; reserve validation for fewer checkpoint decisions. For each candidate route report link recall, complete-set recall, candidate-count percentiles, country/source/noise slices, and oracle macro F0.5. For each matcher report actual macro F0.5, precision/recall, singleton accuracy, and country/source slices. The score implementation includes every S1 and gives an empty prediction for a true singleton score 1. It has sanity checks in `analysis/validation_split_audit.py`.

France has no training or validation truth. Inspect French candidate coverage/volume, accent behavior, and output validity, but do not describe an unmeasured French F0.5 as known. A claim about languages beyond US/India/France also needs evidence from a held-out script or other controlled stress test.
