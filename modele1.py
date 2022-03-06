import pandas as pd
import numpy as np

'''
Review_ID: unique id given to each review
Rating: ranging from 1 (unsatisfied) to 5 (satisfied)
Year_Month: when the reviewer visited the theme park
Reviewer_Location: country of origin of visitor
Review_Text: comments made by visitor
Disneyland_Branch: location of Disneyland Park
'''

df = pd.read_csv(' DisneylandReviews.csv', encoding='cp1252')
df.head()
df['Rating'].unique()
df.info()
df = df.drop(['Review_ID', 'Year_Month', 'Reviewer_Location'], axis=1)



# Préparation des données

#df ya Rating, Review_Text, Disneyland_Branch
df = df.drop(['Review_ID', 'Year_Month', 'Reviewer_Location'], axis=1)
from nltk.corpus import stopwords
from nltk.tokenize import NLTKWordTokenizer

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stop_words.update(["'ve", "", "'ll", "'s", ".", ",", "?", "!", "(", ")", "..", "'m", "n", "u"])
    tokenizer = NLTKWordTokenizer()
    
    text = text.lower()
    
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    
    return ' '.join(tokens)

df['Review_Text'] = df['Review_Text'].apply(preprocess_text)
df.head()
df['Review_Text'][1]


#Premier Modele 
#df1 ya Rating, Review_Text
df1 = df.drop(['Branch'], axis=1)
df1.head()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

features = df['Review_Text']
target = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(features, target)
count_vectorizer_unique = CountVectorizer(max_features=2000)
X_train_cv = count_vectorizer_unique.fit_transform(X_train)
X_test_cv = count_vectorizer_unique.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# model_unique = RandomForestClassifier(max_depth=3, n_estimators=100)
model_unique = LogisticRegression()
# model_unique = DecisionTreeClassifier(max_depth=8)

model_unique.fit(X_train_cv, y_train)

model_unique.score(X_test_cv, y_test)