# EmailFilterAI 🚀

A simple and effective **email spam classifier** built using **Python, TF-IDF, and a Neural Network**.  
This project can classify emails as **Spam** or **Not Spam** with high accuracy.

---

## Features

- Preprocesses emails: removes punctuation, numbers, and extra spaces.
- Converts emails to numerical features using **TF-IDF Vectorization**.
- Trains a **Neural Network** to detect spam.
- Provides **live predictions** on new emails with probabilities.
- Includes a **confusion matrix** to evaluate performance.

---
## Performance

- **Test Accuracy:** 98.65%
- **Validation Accuracy:** ~98–99%
- **Precision / Recall / F1-score:**  
  - Spam: 98%  
  - Not Spam: 99%
- Confusion matrix included to visualize model performance.
<img width="695" height="634" alt="image" src="https://github.com/user-attachments/assets/b69d98fd-f9ee-44a6-a46e-64199d09fed7" />

## Demo

```python
from EmailSpamClassifier import predict_mail

print(predict_mail("Congratulations! You won $1000 cash, click here to claim now!"))
# Output: 🚨 Spam (Probability: 1.00)

print(predict_mail("Hey, are we still on for the meeting tomorrow?"))
# Output: ✅ Not Spam (Probability: 0.00)
git clone https://github.com/MaheshReddy-ML/EmailFilterAI.git
