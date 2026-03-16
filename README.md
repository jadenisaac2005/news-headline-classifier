# 📰 Fake News Classifier

A machine learning project that classifies news articles as **real or fake** using TF-IDF vectorization and Logistic Regression, trained on the WELFake dataset.

---

## 📊 Results

| Version | Accuracy |
|---|---|
| Title only, unigrams | 89.38% |
| Title only, bigrams + Naive Bayes | 86.98% |
| Title only, bigrams + Logistic Regression | 89.30% |
| **Title + Text, bigrams + Logistic Regression** | **95.76%** |

---

## 🗂️ Dataset

- **Name:** WELFake Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- **Size:** ~72,000 labeled news articles
- **Labels:** `0` = Fake, `1` = Real

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
5. **Evaluation** — 80/20 train-test split, accuracy + classification report

---

## 💡 Key Findings

- **Feature engineering > model selection** — switching from title-only to title+text gave a +6.4% accuracy jump with no model change
- **Naive Bayes underperformed** Logistic Regression on this dataset — LR handles the richer feature interactions in full articles better
- **Bigrams helped marginally** on full text but made no difference on headlines alone

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
