# -*- coding: utf-8 -*-
"""
Created on Mon May 11 23:50:32 2020

@author: Rahul
"""

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("labeledTrainData.tsv",delimiter = '\t',quoting = 0)
del dataset['id']
dataset = dataset[['review','sentiment']]

def removePat(strg,pat,nstr):
    return re.sub(pat,nstr,strg)
dataset['review'] = np.vectorize(removePat)(dataset['review'],'@[\w]','')
dataset['review'] = np.vectorize(removePat)(dataset['review'],r'#[0-9a-zA-Z]+','')
dataset['review'] = np.vectorize(removePat)(dataset['review'],'[^a-zA-Z ]','')

from sklearn.feature_extraction.text import TfidfVectorizer
df = dataset[dataset['sentiment'].isin([0,1])]

df.to_csv('cleanedData.tsv',sep='\t',index=False)

tv = TfidfVectorizer(max_df = 0.9,min_df = 2,max_features = 5000, stop_words ='english',ngram_range=(1,2))
sparse = tv.fit_transform(df['review']).toarray()

X = sparse
y = df['sentiment']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classifier.score(X_test,y_test))

import joblib

# Save the model as a pickle in a file
joblib.dump(classifier, 'nlpmodel.pkl')

#plotting confusion matrix
import seaborn as sn
df_cm = pd.DataFrame(cm)
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt='d') # font size
plt.title("Confusion Matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('cm.png')
plt.show()

#s = 'life is good'
#s1 = tv.transform([s]).toarray()
#print(classifier.predict(s1))

