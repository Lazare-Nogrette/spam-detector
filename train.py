import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import sys

data_path = sys.argv[1]
model_path = sys.argv[2]

#df = pd.read_csv('data/emails.csv')
df = pd.read_csv(data_path)

def train_model(dataframe):
    # ham = 0, spam = 1
    df['Spam']=df['Category'].apply(lambda x:1 if x=='spam' else 0)
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Message'])
    y = df['Spam']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

model = train_model(df)
#joblib.dump(model, "model.pkl")
joblib.dump(model, model_path)