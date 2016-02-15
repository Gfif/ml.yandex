import pandas as pn
from sklearn.svm import SVC

data = pn.read_csv('svm-data.csv', header=None)

X = data[[1, 2]]
y = data[0]

svc = SVC(kernel='linear', C=100000, random_state=241)
svc.fit(X, y)

print " ".join([str(i+1) for i in svc.support_])