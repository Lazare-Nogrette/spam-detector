import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import streamlit as st

df = pd.read_csv('data/emails.csv')

df['Spam']=df['Category'].apply(lambda x:1 if x=='spam' else 0)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Message'])
y = df['Spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

st.title("Dashboard Model Performance")

plt.figure(figsize=(6, 4))
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)


y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('FPR')
ax2.set_ylabel('TPR')
ax2.set_title('ROC Curve')
ax2.legend()
st.pyplot(fig2)

report = classification_report(y_test, y_pred, output_dict=True)

fig3, ax3 = plt.subplots()
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='YlGnBu', ax=ax3)
plt.title("Classification Report")
#plt.show()
st.pyplot(fig3)