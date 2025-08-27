# ==============================
# 1️⃣ Load and Clean Data
# ==============================
import pandas as pd
import re
import string

# Load dataset
df = pd.read_csv('email_spam.csv')

# Drop unnecessary columns and rename
df = df.drop(columns=['Unnamed: 0','label'])
df = df.rename(columns={'text':'message', 'label_num':'target'})

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning
df['cleaned_message'] = df['message'].apply(clean_text)
print(df.head())


# ==============================
# 2️⃣ Prepare Data for Training
# ==============================
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

x = df["cleaned_message"]
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorize text
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train).toarray()
x_test = vectorizer.transform(x_test).toarray()

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)


# ==============================
# 3️⃣ Build and Train Model
# ==============================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(256, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train, y_train, 
    epochs=10, 
    batch_size=32, 
    validation_data=(x_test, y_test)
)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy is : {acc:.4f}")


# ==============================
# 4️⃣ Confusion Matrix
# ==============================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Predict on test set
y_pred = (model.predict(x_test) > 0.5).astype(int)

# Create and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Spam', 'Spam'], 
            yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ==============================
# 5️⃣ Predict Individual Emails
# ==============================
def predict_mail(mail):
    mail_cleaned = clean_text(mail)
    mail_vector = vectorizer.transform([mail_cleaned]).toarray()
    prob = model.predict(mail_vector)[0][0]
    
    if prob >= 0.5:
        return f"🚨 Spam (Probability: {prob:.2f})"
    else:
        return f"✅ Not Spam (Probability: {prob:.2f})"


# Example usage
emails = [
    "Congratulations! You won $1000 cash, click here to claim now!",
    "You have won a brand new iPhone! Click here to claim your prize.",
    "URGENT: Your bank account will be locked unless you verify now!",
    "Get cheap medicines online without prescription. Limited offer!",
    "Earn $5000 per week working from home. Sign up today!",
    "Hey, are we still on for the meeting tomorrow?",
    "Don't forget to bring the books for class.",
    "Happy birthday! Wishing you an amazing year ahead.",
    "Can you send me the notes from yesterday's lecture?",
    "Hey bro, let’s meet for lunch tomorrow.",
    "Your invoice for the subscription is attached.",
    "Reminder: Your appointment with Dr. Smith is tomorrow.",
    "Get 20% off your favorite products – valid until midnight!",
    "Please review the attached document and provide feedback."
]

for email in emails:
    print(predict_mail(email))
