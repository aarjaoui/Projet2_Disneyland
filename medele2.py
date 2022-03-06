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


# Deuxieme modele
df['Branch'].unique()
count_vectorizers = {}
models = {}

for branch in df['Branch'].unique():
    count_vectorizer = CountVectorizer(max_features=2000)
#     model = LogisticRegression()
    model = RandomForestClassifier(n_estimators=20, max_depth=5)
    
    df_temp = df[df['Branch'] == branch]
    
    X_train, X_test, y_train, y_test = train_test_split(df_temp['Review_Text'], df_temp['Rating'])
    
    X_train_cv = count_vectorizer.fit_transform(X_train)
    X_test_cv = count_vectorizer.transform(X_test)
    
    model.fit(X_train_cv, y_train)
    print(branch, ':', model.score(X_test_cv, y_test))
    
    count_vectorizers[branch] = count_vectorizer
    models[branch] = model