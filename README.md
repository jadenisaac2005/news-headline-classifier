# 📰 Fake News Classifier

A machine learning project that classifies news articles as **real or fake** using TF-IDF vectorization and Logistic Regression, trained on the WELFake dataset.

---

## 📊 Results

Numbers below are on the deduplicated dataset (63,557 rows, after dropping exact-duplicate `title+text` rows — see the Data Leakage section below).

| Version | Accuracy |
|---|---|
| Title only, bigrams + Logistic Regression | 88.91% |
| Title + Text, bigrams + Naive Bayes | 86.27% |
| **Title + Text, bigrams + Logistic Regression** | **95.22%** |
| Title + Text, bigrams + LR, source/format tokens removed | 93.54% |

The original (non-deduplicated) split reports 95.76%, but that number is inflated by train/test leakage — see below.

---

## 🗂️ Dataset

- **Name:** WELFake Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- **Size:** ~72,000 labeled news articles (63,557 after removing exact duplicates)
- **Labels:** `0` = Real, `1` = Fake — verified by inspecting sampled article titles per label, not assumed from the raw column name

---

## ⚠️ Data Leakage

16,528 rows sit in duplicate groups on the cleaned `title+text` field (7,951 groups); deduplicating (keep first) removes 8,577 rows, taking the dataset from 72,134 to 63,557 rows. Under a random 80/20 split with `random_state=42` on the non-deduplicated data, 2,656 of the 14,427 test rows (18.4%) had an exact duplicate sitting in the training set — the model could partly memorize rather than generalize. The notebook now deduplicates before splitting; accuracy drops from 95.76% (leaky) to 95.22% (deduped), which is the number reported here.

---

## 🛠️ Tech Stack

- Python 3.x
- pandas
- scikit-learn
- NLTK
- Jupyter Notebook

---

## 📁 Project Structure

```
news-headline-classifier/
│
├── WELFake_Dataset.csv       # Dataset (download from Kaggle)
├── fakeNewsClassifier.ipynb  # Main notebook
└── README.md
```

---

## ⚙️ Setup

1. **Clone or download this repo**

2. **Install dependencies**
   ```bash
   pip install pandas scikit-learn nltk
   ```

3. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) and place `WELFake_Dataset.csv` in the project folder

4. **Run the notebook**
   ```bash
   jupyter notebook fakeNewsClassifier.ipynb
   ```

---

## 🔍 How It Works

1. **Preprocessing** — Lowercasing, punctuation removal, stopword filtering using NLTK
2. **Feature Engineering** — Title and article text are combined into a single input field
3. **Vectorization** — TF-IDF with bigrams (`ngram_range=(1,2)`) and top 10,000 features
4. **Model** — Logistic Regression (`max_iter=1000`)
5. **Evaluation** — 80/20 train-test split (deduplicated first), accuracy + classification report

---

## 💡 Key Findings

- **Feature engineering > model selection** — switching from title-only to title+text gave a +6.3 point accuracy jump with no model change
- **Naive Bayes underperformed** Logistic Regression on this dataset — LR handles the richer feature interactions in full articles better
- **Deduplication matters** — 18.4% of the test set leaked from the training set under the naive split; the honest accuracy is 95.22%, not 95.76%

---

## 🧠 What the Model Actually Learned

Inspecting the top 20 highest/lowest `LogisticRegression` coefficients shows the model leans heavily on **source and formatting artifacts**, not just article content:

- Tokens pushing toward **Fake**: `via`, `video`, `image`, `image via`, `hillary`, `breaking`, `trump`, `obama`, `fbi`, `wire`, `share`
- Tokens pushing toward **Real**: `reuters`, `said`, `breitbart`, `washington reuters`, `twitter`, `follow`, `york times`, weekday names (`thursday`, `friday`, `tuesday`, `monday`)

Removing a fixed list of 38 source/format tokens reduces accuracy from 95.22% to 93.54%. Removing them costs only 1.7 points — not because the model barely used them, but because it falls back on other stylistic cues that carry the same information.

After ablation, the top tokens shift, but outlet style is still present:

- Pushing toward **Fake**: `hillary`, `obama`, `trump`, `however`, `america`, `watch`, `entire`, `today`, `president trump`, `fbi`, `even`, `please`, `dc`, `yearold`, `flickr`, `article`, `fact`, `mosul`, `know`, `photo`
- Pushing toward **Real**: `said`, `breitbart`, `president donald`, `trumps`, `https`, `thats`, `us president`, `dont`, `said statement`, `mr`, `im`, `new`, `theres`, `hes`, `didnt`, `ms`, `spokesman`, `islamic state`, `doesnt`, `partys`

The post-ablation features are still house style rather than claims: how people are named (`hillary`, `president trump` vs `president donald`, `us president`, `mr`, `ms`), attribution language (`said`, `said statement`, `spokesman`), and captions or calls to action (`watch`, `photo`, `flickr`, `please`). Tokens like `https`, `yearold`, and apostrophe-stripped contractions (`thats`, `dont`) are artifacts of the cleaning step.

**Caveats not addressed here:** the NLTK stopword list strips negations (`not`, `no`), which can flip the meaning of a sentence before the model ever sees it; and the cleaning regex strips all digits, discarding dates, percentages, and counts that could be genuine signal.

---

## 🚀 Possible Extensions

- Use full BERT embeddings for further accuracy gains
- Build a simple web UI with Flask or Streamlit
- Experiment with Random Forest or SVM classifiers
- Deploy as an API endpoint

---

## 👤 Author

**Jaden**
BTech (AI & ML) — Amity University Bangalore
