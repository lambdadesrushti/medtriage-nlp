# MedTriage 🩺

**An NLP-based symptom-to-department triage classifier**

MedTriage reads a patient's free-text symptom description and predicts which of 11 hospital departments they should be routed to — along with a calibrated, honest confidence score, and a fallback to human triage when the model isn't sure.

Built for the NLP CA3 "Build–Audit–Pitch" mini project at SAII.

---

## 🔍 Overview

| | |
|---|---|
| **Task** | Multi-class text classification (11 hospital departments) |
| **Dataset** | [Medical Transcriptions (MTSamples)](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) — CC0, public domain |
| **Model** | TF-IDF (1–2 grams) + Logistic Regression, with temperature-scaled calibration |
| **Test Accuracy** | 79.8% (vs. 19.6% majority-class baseline) |
| **Macro F1** | 0.76 |
| **Demo** | Live Gradio app with top-3 predictions + confidence |

---

## 🧠 Why This Project

Patients describe symptoms in unstructured free text — at reception, on intake forms, or via telehealth chat — and deciding which department should see them first is still done manually. MedTriage explores whether a lightweight, interpretable NLP pipeline can make that first-pass routing decision reliably enough to be useful, while being transparent about where it struggles and how sure it actually is.

---

## 📊 Dataset

- **Source:** MTSamples medical transcription records (scraped from mtsamples.com), CC0 license
- **Raw size:** 4,999 records across 40 specialty labels
- **After cleaning:** 1,883 records across **11 real hospital departments** (Cardiology, Orthopedics, Gastroenterology, Neurology, Gynecology, Urology, ENT, Ophthalmology, Nephrology, Pediatrics, Psychiatry)
- **Split:** 80/20 stratified (random_state=42) → 1,506 train / 377 test
- **Class imbalance:** Cardiology (372 records) has ~7x more data than Psychiatry (53 records)

### Preprocessing
- Filtered out non-clinical categories (e.g., "Office Notes", "Consult")
- Removed any keyword directly naming a department, to prevent label leakage
- **Trained and evaluated on the `description` field only** — matching exactly what the live demo receives at inference time (see [Design Decisions](#-key-design-decisions))

---

## ⚙️ Pipeline

```
Symptom Text → Clean & Filter → TF-IDF Vectorize → Logistic Regression → Calibrated Top-3 Departments
```

1. **Vectorization:** TF-IDF, unigrams + bigrams, 8,000 features, `min_df=2`, sublinear TF scaling
2. **Classifier:** Logistic Regression (`C=1`, `class_weight='balanced'`, `max_iter=3000`)
3. **Calibration:** Temperature scaling (`T = 2.95`), learned via 5-fold cross-validation on the training set only
4. **Confidence fallback:** Predictions below **0.50 confidence** are flagged for human triage instead of being forced to a department

**Inference latency:** ~1.9 ms per prediction — runs on a laptop, no GPU required.

---

## 📈 Results

### Overall (held-out test set, n=377)

| Metric | Score |
|---|---|
| Accuracy | 79.8% |
| Macro F1 | 0.76 |
| Weighted F1 | 0.80 |
| Baseline (majority class) | 19.6% |

### Per-department

| Department | Precision | Recall | F1 |
|---|---|---|---|
| Cardiology | 0.90 | 0.86 | 0.88 |
| Orthopedics | 0.87 | 0.86 | 0.87 |
| Gastroenterology | 0.86 | 0.83 | 0.84 |
| Urology | 0.92 | 0.75 | 0.83 |
| Gynecology | 0.78 | 0.88 | 0.82 |
| Nephrology | 0.81 | 0.81 | 0.81 |
| ENT | 0.79 | 0.75 | 0.77 |
| Ophthalmology | 0.76 | 0.76 | 0.76 |
| Psychiatry | 0.60 | 0.90 | 0.72 |
| Neurology | 0.65 | 0.67 | 0.66 |
| **Pediatrics** | 0.35 | 0.43 | **0.39** |

---

## 🔍 Bias Audit

Performance was checked across text-style subgroups, not just overall accuracy:

| Subgroup | Accuracy |
|---|---|
| Short descriptions (≤15 words) | 76.0% |
| Long descriptions (>15 words) | 83.8% |
| Narrative style ("a 23-year-old presents with...") | 73.1% |
| Terse/procedure-title style | 82.4% |

**Key finding:** The model is meaningfully less reliable on short, narrative, generically-worded text — exactly the kind of input real patients are most likely to type.

**Weakest department:** Pediatrics (F1 = 0.39) — its vocabulary ("child", "baby", "month old") is generic and overlaps with how other departments describe similar presentations in younger patients.

---

## 💡 Explainability

Top TF-IDF/Logistic Regression features per department (a sample):

| Department | Top words |
|---|---|
| Cardiology | chest pain, cardiac, coronary, pulmonary |
| Ophthalmology | eye, cataract, vitrectomy |
| Urology | bladder, prostate, inguinal |
| Pediatrics | child, baby, month old (generic) |

**Honest limitation:** The model matches words, not context. *"4-month-old with tachycardia"* is classified as **Pediatrics at 99% confidence** — the age-related words overwhelm the single cardiac term. **This is not safe for real clinical use** without human oversight.

---

## 🎯 Confidence Calibration

A raw Logistic Regression on sparse TF-IDF is under-confident — correct predictions often score as low as 23–39%. MedTriage fixes this with **temperature scaling**:

- Learned `T = 2.95` using 5-fold CV on the training set only (test set never touched)
- Predictions and accuracy are **unchanged** — only the confidence scores become honest
- **Expected Calibration Error (ECE):** 0.426 → 0.079
- **Mean confidence:** 0.37 → 0.78

---

## 🖥️ Live Demo

Built with [Gradio](https://www.gradio.app/) — run the final notebook cell and it launches a public shareable link instantly (`demo.launch(share=True)`), no deployment needed.

**Example predictions:**

| Input | Prediction |
|---|---|
| "sharp chest pain radiating to the left arm with shortness of breath" | Cardiology — 93% |
| "blurry vision and redness in the left eye for three days" | Ophthalmology — 100% |
| "feeling tired and unwell for a few days" | Low confidence → routed to human triage |

---

## 🧩 Key Design Decisions

**Why description-only, not description+keywords?**
An earlier version trained on `description + keywords` scored higher (84.4%) in testing — but the live demo only ever receives typed free text, with no keywords. An ablation confirmed the mismatch: that same model dropped to 80.1% when tested on description-only input. MedTriage now trains and evaluates consistently on description-only text (79.8%), so the reported accuracy reflects real demo conditions rather than an inflated lab number.

**Why Logistic Regression over deep learning?**
With ~1,900 records, a transformer model risks overfitting and sacrifices the coefficient-level interpretability this project's audit and explainability sections depend on.

---

## 🛠️ Tech Stack

- Python 3, scikit-learn, pandas, numpy
- matplotlib, seaborn (visualizations)
- scipy (temperature-scaling optimization)
- Gradio (live demo interface)

---

## ⚠️ Limitations & Ethical Considerations

- **Not for clinical use** — a classroom demonstration and routing suggestion only
- Purely lexical matching — no real contextual/medical understanding
- Trained on US-sourced transcription data; may not generalize to other populations or settings
- Class imbalance means smaller departments (Psychiatry, Pediatrics) are less reliably served

---

## 🚀 Future Work

- Replace TF-IDF with contextual embeddings (e.g., a fine-tuned clinical BERT) and re-run the same audit
- Collect more data for low-support departments
- Expand the confidence-threshold fallback into a tiered triage system.
