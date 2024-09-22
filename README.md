# local-llms

## hybrid determination
Lightweight fine-tuned distilBERD LLM to determine which service (local/cloud/vision) should handle a query
### To Build:
- Run "training/trainers/hybrid_determination_trainer.py"
### To Test:
- Run "model-testers/hybrid_determination_tester.py"
### Labels
| Type               | Explanation                                                                                           |
|--------------------|-------------------------------------------------------------------------------------------------------|
| `offline_question` | A query which can be answered by the local LLM                                                        |
| `basic_question`   | A Google search-like query which does not require advanced reasoning                                  |
| `complex_question` | A query which requires either advanced reasoning or linguistic analysis                               |
| `vision`           | A query which requires vision processing                                                              |
| `explicit`         | A query which is inappropriate and should be immediately thrown out to avoid wasting processing power |

