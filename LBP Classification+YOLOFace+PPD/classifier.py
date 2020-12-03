import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


X_train, X_test, y_train, y_test = \
    np.loadtxt('X_train.txt'), np.loadtxt('X_test.txt'), np.loadtxt('y_train.txt'), np.loadtxt('y_test.txt')
y_test = y_test.astype(np.int)
y_train = y_train.astype(np.int)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

clf = SVC(kernel='linear', random_state=1, probability=True)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
train_proba = clf.predict_proba(X_train)
print(acc)
plot_confusion_matrix(clf, X_test, y_test, normalize='true')
plt.show()
plt.clf()

# ada_clf = AdaBoostClassifier(base_estimator=SVC(kernel='linear', random_state=1, probability=True), random_state=1)
# ada_clf.fit(X_train, y_train)
# ada_train_proba = ada_clf.predict_proba(X_train)
# ada_acc = ada_clf.score(X_test, y_test)
# print(ada_acc)
# plot_confusion_matrix(ada_clf, X_test, y_test, normalize='true')
# plt.show()
# plt.clf()

# secondary_clf = SVC(kernel='linear', random_state=1, probability=True)
# secondary_clf.fit(ada_train_proba, y_train)
# secondary_acc = secondary_clf.score(ada_clf.predict_proba(X_test), y_test)
# print(secondary_acc)
# plot_confusion_matrix(secondary_clf, ada_clf.predict_proba(X_test), y_test, normalize='true')
# plt.show()

secondary_clf = SVC(kernel='linear', random_state=1, probability=True)
secondary_clf.fit(train_proba, y_train)
secondary_acc = secondary_clf.score(clf.predict_proba(X_test), y_test)
print(secondary_acc)
plot_confusion_matrix(secondary_clf, clf.predict_proba(X_test), y_test, normalize='true')
plt.show()




