# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
# numpy
import numpy as np
# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
# random, itertools, matplotlib
import random
import itertools
import matplotlib.pyplot as plt
model = Doc2Vec.load('./imdb.d2v')
X_train = np.zeros((25000, 100))
y_train = np.zeros(25000)

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    X_train[i] = model.docvecs[prefix_train_pos]
    X_train[12500 + i] = model.docvecs[prefix_train_neg]
    y_train[i] = 1
    y_train[12500 + i] = 0
#print(y_train)
X_test = np.zeros((25000, 100))
y_test = np.zeros(25000)
for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    X_test[i] = model.docvecs[prefix_test_pos]
    X_test[12500 + i] = model.docvecs[prefix_test_neg]
    y_test[i] = 1
    y_test[12500 + i] = 0
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

n = {'data': [{'created_time': '2019-02-22T10:08:20+0000', 'from': {'name': 'Trần Bá Quân Thụy', 'id': '2204184089677893'}, 'message': 'This is good', 'id': '465092984027677_465093104027665'}, {'created_time': '2019-03-16T14:13:03+0000', 'from': {'name': 'Trần Bá Quân Thụy', 'id': '2204184089677893'}, 'message': 'This is bad', 'id': '465092984027677_475259343011041'}], 'paging': {'cursors': {'before': 'MgZDZD', 'after': 'MQZDZD'}}}
m  = n['data']
#jdata = json.loads(n)
commentss = [x['message'] for x in m]
idpost  = [x['from'] for x in m]       
print(commentss[0])
tf = TfidfVectorizer(min_df=0,max_df= 1,max_features=1000,sublinear_tf=True)
text =[[commentss[0]]]
for i in text:
  print(i)
  f = np.transpose(i)

  test = tf.fit_transform(f)
  #test = tf.transform(i)
  print(test)
  print(classifier.predict(test))