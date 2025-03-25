# Customer Sentiment Classification Model

## Overview
This project explores different machine learning and deep learning techniques for customer sentiment classification. We experimented with traditional machine learning models using TF-IDF and Bag of Words (BoW) representations, as well as deep learning models utilizing word embeddings.

## Model Comparison

### Machine Learning Approaches
We implemented three classifiers: **XGBoost, Random Forest (RF), and AdaBoost**, evaluating them under both **Bag of Words (BoW) and TF-IDF representations**.

#### **Bag of Words (BoW) Results:**
- **Random Forest (RF):** **84%** Accuracy
- **XGBoost:** **89%** Accuracy
- **AdaBoost:** **71%** Accuracy

#### **TF-IDF Results:**
- **Random Forest (RF):** **87%** Accuracy
- **XGBoost:** **87%** Accuracy
- **AdaBoost:** **87%** Accuracy

### Deep Learning Approaches
To improve the performance further, we explored deep learning models using **word embeddings with a max vocabulary size of 5000**.

#### **Deep Learning Results:**
- **LSTM:** **88%** Accuracy
- **Bi-LSTM:** **88%** Accuracy
- **GRU-LSTM:** **87%** Accuracy

## **Comparative Analysis**
1. **Effectiveness of Traditional ML Models**
   - XGBoost performed the best under BoW (**89%**) but did not show improvements under TF-IDF (**87%**).
   - Random Forest performed slightly better under TF-IDF (**87%**) compared to BoW (**84%**).
   - AdaBoost showed **significant improvement** when using TF-IDF (**87%**) over BoW (**71%**), suggesting that TF-IDF is a better feature representation for this classifier.

2. **Deep Learning vs. Traditional ML**
   - Deep learning models performed **comparably to XGBoost and RF** with TF-IDF, with LSTM and Bi-LSTM achieving **88%** accuracy.
   - The **GRU-LSTM model performed slightly lower at 87%**, suggesting that GRU did not provide a significant advantage over standard LSTMs in this case.
   
3. **Feature Representation Impact**
   - **BoW is less effective for AdaBoost**, while **TF-IDF improved all traditional models**.
   - **Deep learning models using embeddings performed on par with the best TF-IDF-based models**, showing that embeddings with a limited vocabulary (5000 words) were sufficient for high accuracy.

## **Conclusion**
- **XGBoost remains a strong choice for sentiment classification**, performing well under both feature representations.
- **Deep learning models (LSTM/Bi-LSTM) achieved competitive results**, making them a viable alternative, especially when working with large datasets and sequence-based text processing.
- **TF-IDF representation is generally more effective** than BoW, particularly for boosting AdaBoost’s performance.
- **Future work** can explore attention mechanisms or transformer-based models (e.g., BERT) for further improvements.


