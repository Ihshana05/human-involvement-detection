# human-involvement-detection

Human Involvement Detection in AI-Assisted Plagiarism using HIRS Framework



\# Human Involvement Detection in AI-Assisted Plagiarism



> \*\*HIRS — Human Involvement Risk Score Framework\*\*  

> Anna University | Department of Computer Science \& Engineering  

> Under the guidance of Dr. K. Latha



---



\## Results



| Metric | Value |

|--------|-------|

| Test Accuracy | \*\*90.74%\*\* |

| Macro F1 | \*\*0.9063\*\* |

| ChatGPT F1 | 1.00 |

| AI-Polished F1 | 0.84 |

| Human-Fusion F1 | 0.81 |

| Human F1 | 0.98 |

| 5-Fold CV Accuracy | 89.94% ± 0.73% |



Outperforms Liu et al. (78.5%) and Zeng et al. (82.0%) on the same CHEAT dataset.  

Only 4-class system on CHEAT with full \*\*SHAP explainability\*\*.



---



\## What is HIRS?



HIRS (Human Involvement Risk Score) is a continuous 0–1 score that quantifies

the degree of human involvement in an AI-assisted academic abstract.



| HIRS Range | Class | Human Involvement |

|------------|-------|-------------------|

| 0.00 – 0.25 | ChatGPT | None — fully AI-generated |

| 0.25 – 0.50 | AI-Polished | Low — AI rewrote human text |

| 0.50 – 0.75 | Human-Fusion | Medium — human + AI mixed |

| 0.75 – 1.00 | Human | High — human-written |



---



\## Dataset



\*\*CHEAT Dataset\*\* — 18,056 IEEE abstracts across 4 classes (4,514 each):



| Label | Class | Description |

|-------|-------|-------------|

| 0 | ChatGPT | Fully AI-generated |

| 1 | AI-Polished | AI rewrote original human text |

| 2 | Human-Fusion | Human + AI content merged |

| 3 | Human | Original human-written |



Split: \*\*70% Train / 15% Validation / 15% Test\*\* at abstract\_id level (zero leakage).



> Note: CHEAT\_DATASET.csv is not included due to size.  

> Download from: https://github.com/brickee/ChEAT



---



\## Feature Modules



| Module | Method | What It Measures |

|--------|--------|-----------------|

| \*\*L\*\* — Lexical | N-gram Jaccard + TF-IDF Cosine | Vocabulary overlap vs human reference |

| \*\*S\*\* — Semantic | SBERT all-MiniLM-L6-v2 | Meaning preservation vs human reference |

| \*\*C\*\* — Stylometric | Function-word variance, Sentence entropy, POS drift, Perplexity fluctuation | Writing style irregularity |

| \*\*A\*\* — AI Signal | Burstiness, Repetition rate, GPT-2 log-probability, Ensemble detector | AI generation patterns |



---



\## HIRS Formula

```

HIRS = 0.7 × P(Human) + 0.3 × max(P(Human-Fusion), P(Human))

```



Computed from Multinomial Logistic Regression trained on \[L, S, C, A].



---



\## Run Order

```bash

python script/step0\_reconstruct.py       # 4,514 rows → 18,056 long format

python script/step1\_preprocessing.py     # unicode fix, clean + raw versions

python script/step2\_split.py             # 70/15/15 split at abstract\_id level

python script/step3\_module\_L.py          # N-gram + TF-IDF lexical features

python script/step4\_module\_S.py          # SBERT semantic features

python script/step5\_module\_C.py          # Stylometric features (needs GPU)

python script/step6\_module\_A.py          # AI signal features (needs GPU)

python script/step7\_combine\_train.py     # Train logistic regression + HIRS

python script/step8\_test\_eval.py         # Final test evaluation (run once)

python script/step9\_shap.py              # SHAP explainability

```



---



\## Repository Structure

```

human-involvement-detection/

├── script/          ← 10 pipeline scripts (step0 to step9)

├── model/           ← hirs\_model.pkl + hirs\_scaler.pkl

├── outputs/         ← classification\_report, confusion\_matrix, SHAP files

├── requirement.txt  ← Python dependencies

└── README.md

```



---



\## Requirements

```

pandas

numpy

scikit-learn

sentence-transformers

torch

transformers

nltk

shap

joblib

```



---



\## Team



\- \*\*S. Ihshana Thahsin\*\* (810022104029)

\- \*\*M. Kaviya\*\* (810022104016)



Department of Computer Science \& Engineering  

Anna University, Chennai — 600 025  

Under the guidance of \*\*Dr. K. Latha\*\*, Assistant Professor

