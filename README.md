# NLP Disaster Tweets Classification

Kaggle competition: [nlp-getting-started](https://www.kaggle.com/competitions/nlp-getting-started)

Kaggle notebook (BERTweet run): [usingbertweet](https://www.kaggle.com/code/putsalaharshapriya/usingbertweet?scriptVersionId=343445966)

Binary classification of tweets as referring to a real disaster (`1`) or not (`0`).

## Project Log

### 1. Baseline EDA + classical ML
- Loaded the dataset, checked shape/nulls/duplicates, dropped `keyword` and `location` columns.
- Preprocessed text for TF-IDF: lowercased, removed stopwords, lemmatized.
- Vectorized with `TfidfVectorizer` (1-2 grams, max 3000 features).
- Trained baselines: Logistic Regression, SVM, Gradient Boosting, SGD, Random Forest, KNN, Naive Bayes.

### 2. First DistilRoBERTa run
- Fine-tuned `distilroberta-base`, `learning_rate=2e-5`, `epochs=3`.
- Result: **~83% accuracy** → Kaggle score **0.83328**.

### 3. Second DistilRoBERTa run
- Changed to `learning_rate=1e-5`, `epochs=5`.
- Result: accuracy dropped to **~81%**.
- Root cause investigation found this wasn't really an LR effect:
  - The same fine-tuned `model` object was being reused/continued across training runs instead of reloading a fresh pretrained model each time.
  - `compute_metrics` was referenced before it was defined, which meant `EarlyStoppingCallback` never actually got attached to the training run that executed.

### 4. Rewrote the training pipeline
Fixes applied:
- Fresh model reload before every training run.
- `compute_metrics` defined before any `Trainer` is built.
- Fixed random seed (`42`) everywhere for reproducible comparisons.
- Learning rate reset to `2e-5` (standard fine-tuning range).
- `EarlyStoppingCallback` properly attached.
- Rows with label-conflicting duplicate text dropped (55 rows).
- Light, transformer-appropriate text cleaning (URL/user normalization only — no stopword removal/lemmatization, which hurts transformer models).
- Optional Stratified K-Fold cross-validation (`RUN_CV` flag) for a trustworthy accuracy estimate instead of trusting a single split.

### 5. Reran fixed DistilRoBERTa
- Validation accuracy **0.8357**, F1 **0.7984**.
- Kaggle scores: **0.83266** and **0.83328** (consistent across two submissions).

### 6. Tried RoBERTa-base
- Larger model (125M vs 82M params), same training config (adapted batch size/grad accumulation for memory).
- Validation accuracy **0.8216**, F1 **0.7903** — underperformed distilroberta.
- Loss curve showed it was still descending at epoch 4 (undertrained relative to distilroberta in the same number of epochs).

### 7. Tried BERTweet-base
- Tweet-domain-pretrained model (`vinai/bertweet-base`), tokenizer normalization enabled (URL → `HTTPURL`, `@mention` → `@USER`, emoji handling).
- Validation accuracy **0.8310**, F1 **0.7938** — close to distilroberta but didn't beat it.

### 8. Best result so far
- Latest submission: **0.83971** — a real improvement over the earlier 0.83328 baseline.

### 9. Current focus: pushing past 0.83971
Options under consideration, ranked by expected effort vs. payoff:
1. **Ensemble** distilroberta + roberta-base + bertweet-base + classical TF-IDF models (soft-vote on probabilities).
2. **Further label cleaning** — spot-check high-confidence misclassifications for possible label noise beyond the duplicates already removed.
3. **Pseudo-labeling** — add high-confidence test predictions back into training data.
4. **Feature engineering** — hashtag count, exclamation marks, all-caps ratio, tweet length, and the currently-dropped `keyword`/`location` columns as auxiliary signal.
5. **Larger/task-matched models** — `roberta-large`, `vinai/bertweet-large`, `cardiffnlp/twitter-roberta-base` (higher cost, likely smaller marginal gain at this point).

## Results Summary

| Model | Val Accuracy | Val F1 | Kaggle Score |
|---|---|---|---|
| Classical ML (TF-IDF + LogReg/SVM/etc.) | — | — | — |
| DistilRoBERTa (lr=2e-5, 3 epochs, buggy pipeline) | ~0.83 | — | 0.83328 |
| DistilRoBERTa (lr=1e-5, 5 epochs, buggy pipeline) | ~0.81 | — | 0.81581 |
| DistilRoBERTa (fixed pipeline) | 0.8357 | 0.7984 | 0.83266 / 0.83328 |
| RoBERTa-base (fixed pipeline) | 0.8216 | 0.7903 | — |
| BERTweet-base (fixed pipeline) | 0.8310 | 0.7938 | — |
| **Best submission** | — | — | **0.83971** |

## Key Lessons

- **Always reload a fresh pretrained model** before each training run you intend to compare — reusing a fine-tuned model object silently continues training on top of it and invalidates hyperparameter comparisons.
- **Define `compute_metrics` before building any `Trainer`** that references it, or callbacks like early stopping may silently fail to attach.
- **Don't reuse TF-IDF-style preprocessing (stopword removal, lemmatization) for transformer models** — they rely on full sentence context and this preprocessing actively hurts performance.
- **A single 80/20 split has real variance** on a ~7.5k row dataset; results within ~1 point of each other across model runs may just be noise, not a genuine model difference.
- **Bigger/domain-pretrained models aren't automatically better** in a fixed low number of epochs — both roberta-base and bertweet-base were still visibly undertrained (descending loss curve) at epoch 4, while distilroberta had already converged.
- **Ensembling is usually the cheapest reliable gain** once several independently-trained models already exist.
