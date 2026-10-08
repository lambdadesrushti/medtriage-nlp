# MedTriage-NLP

**Symptom-to-Department Classification with Bias & Explainability Audit**

Srushti Lambdade | PRN: 25030421021 | B.Sc. (AI)
Symbiosis Artificial Intelligence Institute (SAII)
NLP — CA3: Build-Audit-Pitch Mini Project

## Problem Statement
Manually routing patients to the correct hospital department based on their symptom description is slow and inconsistent. This project builds an NLP text classifier that predicts the most relevant medical department (e.g., Cardiology, Orthopedics, ENT) from a symptom/case description. It audits the model's accuracy on short vs. long descriptions to uncover hidden performance gaps, and uses explainability techniques to make its predictions transparent.

## Dataset
[Medical Transcriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) (Kaggle, tboyle10), sourced from mtsamples.com. Licensed CC0: Public Domain.
After cleaning, 2,232 real clinical cases across 11 hospital departments were used.

## Method
- **Preprocessing:** removed non-department categories, mapped specialties to 11 departments, and combined each description with its keywords after removing any keyword that named a department (to prevent label leakage).
- **Features:** TF-IDF (unigrams + bigrams, sublinear TF, 8,000 features)
- **Model:** Logistic Regression (C = 1, class-balanced)
- **Split:** 80/20 stratified (1,855 train / 377 test)
- **Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Google Colab

## Results
| Metric | Score |
|---|---|
| Accuracy | **84.3%** |
| Macro F1 | 0.83 |
| Weighted F1 | 0.85 |

## Bias / Subgroup Audit
| Subgroup | Accuracy |
|---|---|
| Short descriptions (≤ 33 words) | 82.3% |
| Long descriptions (> 33 words) | 86.6% |

The model is modestly more reliable on longer, detail-rich descriptions.

## Explainability
Top predictive words per department, taken from the Logistic Regression coefficients. For example, "chest, cardiac, coronary, artery" for Cardiology and "knee, fracture, joint, tendon" for Orthopedics.

## Limitations
- The descriptions are clinician-written summaries, not patient self-reported text, so casual real-world phrasing may reduce accuracy.
- Pediatrics (F1 = 0.53) and Neurology (F1 = 0.69) are weaker because of smaller sample sizes and vocabulary overlap with other departments.
- This is a closed-set classifier. It can only choose among its 11 trained departments.

## Repository Contents
| File | Description |
|---|---|
| `MedTriage_NLP.ipynb` | Full pipeline: cleaning, training, audit, explainability, live demo widget |
| `mtsamples.csv` | Dataset |
| `MedTriage_Writeup.docx` | One-page write-up (Build component) |
| `MedTriage_Model_Card.docx` | One-page model card (Audit component) |
| `MedTriage_Pitch_Slides.pptx` | Pitch slides |

## How to Run
Open `MedTriage_NLP.ipynb` in Google Colab, then go to Runtime → Run all. The dataset loads automatically from this repository, so no manual upload is needed.
