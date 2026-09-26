# 📊 Telegram NLP Classifier & Filtration System
## Telegram E'lonlarini Avtomatik Klassifikatsiya va Spam Filtrlash Tizimi

A machine learning pipeline that automatically classifies Telegram marketplace posts (Uzbekistan) into 7 product/service categories, while filtering out spam and casual chat noise.

O'zbekiston Telegram marketplace guruhlaridagi e'lonlarni 7 ta toifaga avtomatik ajratuvchi va spam/suhbat shovqinini filtrlaydigan machine learning tizimi.

---

## 🎯 1. Business Understanding / Biznesni Tushunish

**EN:** Telegram marketplace groups in Uzbekistan mix real product/service ads with spam links, invite messages, and casual chat. Manually sorting tens of thousands of messages is impractical. This project builds an NLP pipeline that separates genuine ads from noise and classifies them by category, enabling faster search, moderation, and analytics.

**UZ:** O'zbekiston Telegram marketplace guruhlarida haqiqiy e'lonlar spam havolalar, taklif xabarlari va oddiy suhbatlar bilan aralashib ketadi. O'n minglab xabarlarni qo'lda saralash amalda mumkin emas. Ushbu loyiha haqiqiy e'lonlarni shovqindan ajratadigan va ularni toifalarga bo'ladigan NLP pipeline yaratadi — bu qidiruv, moderatsiya va tahlilni tezlashtiradi.

---

## 🗂️ 2. Dataset / Ma'lumotlar

- **Source / Manba:** Telegram marketplace groups, collected via Telethon (`telegram_data.csv`)
- **Raw size / Xom hajm:** 113,415 messages
- **Columns:** `category`, `group`, `msg_id`, `date`, `sender_id`, `views`, `has_media`, `text`
- **Encoding issue solved:** File required `utf-8-sig` decoding; malformed rows handled with `on_bad_lines='skip'`

---

## 🧹 3. Data Cleaning & Filtration / Tozalash va Filtrlash

Rule-based pre-classification using Regex, applied before ML modeling:

| Rule | Condition | Result |
|---|---|---|
| Spam detection | `t.me/` links > 2, invite phrases | → Spam |
| Chat noise | Text length < 20 characters | → Chat (Suhbat) |
| Clean ad | Passes both filters | → Cleaned & kept |

**Result / Natija:**
- ✅ Clean ads / Toza e'lonlar: **27,018**
- 🗑️ Spam & chat / Spam va suhbat: **13,739**

---

## 📈 4. Exploratory Data Analysis (EDA)

- Category distribution shows strong imbalance: **Elektronika** (8,831) dominates while **Transport** originally had only 2 samples.
- **Uy-Joy** (real estate) ads have the longest average text length — consistent with detailed listings (rooms, price, condition, location).
- **Transport** ads are shortest on average before balancing.

---

## ⚖️ 5. Balancing & Transformation / Balansirovka va Transformatsiya

- Class imbalance corrected: all **7 categories balanced to 2,000 samples each** (14,000 total) via over/undersampling.
- **TF-IDF vectorization** applied to convert cleaned text into numerical features.
- Train/test split: 80/20 → 11,200 train / 2,800 test samples.

---

## 🛠️ 6. Feature Engineering / Belgilarni Muhandislik Qilish

Two structural features extracted via Regex from the **raw** text (before stopword/symbol cleaning, to preserve signals like `$` and phone digits):

| Feature | Description |
|---|---|
| `Has_Phone` | Boolean flag — detects Uzbek phone number patterns |
| `Has_Price` | Boolean flag — detects currency keywords (so'm, $, usd, narxi...) |

---

## 🤖 7 & 8. Model Training & Evaluation / Model O'qitish va Baholash

**Algorithm:** Logistic Regression (`max_iter=1000`, `random_state=42`)

### Version 1 — Unigram TF-IDF (max_features=5000)
| Metric | Score |
|---|---|
| **Accuracy** | 82.11% |
| Weakest class | Ish / Kiyim-kechak (frequently confused) |
| Strongest class | Transport (F1 = 0.96) |

### Version 2 — Improved: N-gram TF-IDF (1,2), max_features=8000, min_df=2
| Metric | Score |
|---|---|
| **Accuracy** | **84.00%** (+1.89%) |
| Improvement | Consistent gains across nearly all classes (Hayvonlar F1: 0.84 → 0.88) |

**Full classification report (v2):**

```
                precision    recall  f1-score   support

   Elektronika       0.85      0.78      0.82       400
     Hayvonlar       0.87      0.89      0.88       400
           Ish       0.74      0.70      0.72       400
  Kiyim-kechak       0.69      0.85      0.76       400
     Transport       0.93      1.00      0.96       400
Usta/Xizmatlar       0.89      0.82      0.85       400
        Uy-Joy       0.91      0.81      0.86       400

      accuracy                           0.84      2800
     macro avg       0.84      0.84      0.84      2800
  weighted avg       0.84      0.84      0.84      2800
```

**Confusion matrix insight:** The main source of error is overlap between **"Ish"** (jobs) and **"Kiyim-kechak"** (clothing) — likely because job ads (e.g., "tikuvchi kerak" / "seamstress needed") share vocabulary with clothing ads.

---

## 🔧 9. Post-Processing: Keyword Override Rule

**Problem:** Car brand names (Toyota, Cobalt, Nexia, etc.) are rare/unseen tokens in TF-IDF, causing the model to occasionally misclassify Transport ads with low confidence.

**Solution:** A lightweight rule-based override checks for known transport keywords (car brands, mechanical terms) and reassigns the category to **Transport** when the model's confidence is below a threshold (60%) and a keyword match is found — without retraining the model.

```python
test_msg = "Cobalt sotiladi! Yili 2022, yurgani 35,000 km..."
# Before override: uncertain, low-confidence wrong class
# After override → Transport | Confidence: 31.82% (overridden by keyword rule)
```

---

## ⚠️ 10. Limitations / Cheklovlar

- Rare/unseen brand names and slang can still confuse the base ML model — mitigated but not fully solved by keyword overrides.
- "Ish" and "Kiyim-kechak" categories remain the hardest to separate due to vocabulary overlap.
- The model is trained on Uzbek/Russian mixed marketplace text; performance on other domains or languages is untested.
- Balancing via oversampling may cause the model to slightly overfit on repeated minority-class samples (e.g., original Transport had only 2 raw examples before balancing).

---

## 💾 11. Saved Artifacts / Saqlangan Fayllar

| File | Description |
|---|---|
| `telegram_classifier_model.pkl` | Trained Logistic Regression model (v2, 84% accuracy) |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer (n-gram 1,2) |
| `telegram_dataset_toza.xlsx` | Cleaned ads dataset |
| `telegram_dataset_iflos.xlsx` | Filtered spam/chat dataset |
| `telegram_dataset_features.xlsx` | Dataset with engineered features |

---

## 🚀 Usage / Foydalanish

```python
import joblib

model = joblib.load('telegram_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

text = "3 xonali kvartira ijaraga beriladi, markazda"
text_tfidf = vectorizer.transform([text])
prediction = model.predict(text_tfidf)

print(prediction)  # → ['Uy-Joy']
```

---

## 🧰 Tech Stack

`Python` · `pandas` · `scikit-learn` · `Telethon` · `matplotlib` / `seaborn` · `Google Colab`

---

## 📌 Pipeline Summary / Pipeline Xulosasi

```
Raw Telegram Data (113,415 msgs)
        ↓
Regex-based Spam/Chat Filtering → 27,018 clean ads
        ↓
EDA & Class Balancing (7 categories × 2,000)
        ↓
TF-IDF Vectorization (n-grams) + Feature Engineering
        ↓
Logistic Regression Training → 84% Accuracy
        ↓
Keyword-based Post-processing (Transport edge cases)
        ↓
Saved Model Ready for Inference
```

---

## 👤 Author

Built as a portfolio project demonstrating an end-to-end NLP classification pipeline — from raw noisy social data to a deployable, evaluated, and post-processed machine learning model.
