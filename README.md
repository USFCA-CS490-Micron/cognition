# local-llms

## hybrid-determination
Lightweight fine-tuned distilBERD LLM to determine which service (local/cloud/vision) should handle a query
### To Build:
- Run "training/trainers/hybrid_determination_trainer.py"
### To Test:
- Run "model-testers/hybrid_determination_tester.py"
### Labels
| Type               | Explanation                                                                                               |
|--------------------|-----------------------------------------------------------------------------------------------------------|
| `offline_question` | A query which can be answered by the local lightweight LLM                                                |
| `basic_question`   | A Google search-like query which does not require advanced reasoning but does need up-to-date information |
| `complex_question` | A query which requires either advanced reasoning or linguistic analysis                                   |
| `vision`           | A query which requires vision processing                                                                  |
| `explicit`         | A query which is inappropriate and should be immediately thrown out to avoid wasting processing power     |

### Generate Training Data with an LLM
To generate additional training data with an LLM (GPT-4o preferred), use the following prompt:
```
Please generate distilBERD fine-tuning training data for a model which responds in the following ways:

"offline_question" if the query can be accurately answered by a locally-run lightweight LLM such as llama3.
"basic_question" if the query is similar to a google search and/or requires up-to-date information; these are queries best handled by Google Natural Language.
"complex_question" if the query requires linguistic analysis, reasoning, summarization, or explanation; these are queries best handled by a model like GPT-4.
"vision" if the query requires vision processing.
"explicit" if the query is inappropriate in any way (sex, drugs, swear words, etc). Queries which ask about health effects of drugs or substances are not explicit.

Please generate the data in the following CSV format with no header row:
"query","type".

Please output text instead of a file.

Please ensure the data is balanced across all types of queries.
```
Copy output into `training/data/decision_data.csv`, train, then test.

### Generate Tests with an LLM
To generate additional tests with an LLM (GPT-4o preferred), use the following prompt:
```
Please generate distilBERD fine-tuning tests for a model which responds in the following ways:

"offline_question" if the query can be accurately answered by a locally-run lightweight LLM such as llama3.
"basic_question" if the query is similar to a google search and/or requires up-to-date information; these are queries best handled by Google Natural Language.
"complex_question" if the query requires linguistic analysis, reasoning, summarization, or explanation; these are queries best handled by a model like GPT-4.
"vision" if the query requires vision processing.
"explicit" if the query is inappropriate in any way (sex, drugs, swear words, etc).

Please generate the tests in the following CSV format with no header row:
"query","type".

Please output text instead of a file.

Please ensure the tests are balanced across all types of queries.
```
Copy output into `model-testers/tests/determination_test_data.csv`, train, then test.
