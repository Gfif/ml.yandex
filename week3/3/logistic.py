import pandas as pn
import numpy as np
import math

from sklearn.metrics import roc_auc_score
from numpy.linalg import norm

def norm(x, y):
    return math.sqrt((y[0] - x[0])**2 + (y[1] - x[1])**2)

class Grad:
    def __init__(self, w1=0, w2=0, C=10, k=0.1, N=10000, l2=False):
        self.w1 = w1
        self.w2 = w2
        self.C = C
        self.k = k
        self.N = N
        self.l2 = l2

    def __iter__(self):
        return self

    def y(self, i):
        return float(self.Y[i])

    def x1(self, i):
        return float(self.X[i][0])

    def x2(self, i):
        return float(self.X[i][1])
        
    def fit(self, X, y):
        self.X = X
        self.Y = y
        self.l = len(self.X)
        self.i = 0

        for it in g:
            self.w1, self.w2 = it


    def next(self):
        self.i += 1

        w1 = self.w1 + self.k / self.l * \
            sum([self.y(i) * self.x1(i) * (1 - (1 / (1 + math.e ** (-self.y(i) * (self.w1 * self.x1(i) + self.w2 * self.x2(i)))))) \
                for i in xrange(1, self.l)])
            
        w2 = self.w2 + self.k / self.l * \
            sum([self.y(i) * self.x2(i) * (1 - (1 / (1 + math.e ** (-self.y(i) * (self.w1 * self.x1(i) + self.w2 * self.x2(i)))))) \
                for i in xrange(1, self.l)])
        
        if self.l2:
            w1 -= self.k * self.C * self.w1
            w2 -= self.k * self.C * self.w2

        if norm([self.w1, self.w2], [w1, w2]) < 1e-5 or self.i >= self.N:
            raise StopIteration()

        return (w1, w2)

    def a(self, x1, x2):
        return 1 / (1 + math.e ** (-self.w1 * x1 - self.w2 * x2))

    def predict(self, X):
       return map(lambda x: self.a(x[0], x[1]), X)

data = pn.read_csv('data-logistic.csv', header=None)
X, y = data[[1, 2]].as_matrix(), data[0].as_matrix()

g = Grad()
g.fit(X, y)

print "%.3f" % roc_auc_score(y, g.predict(X))

g = Grad(l2=True)
g.fit(X, y)

print "%.3f" % roc_auc_score(y, g.predict(X))