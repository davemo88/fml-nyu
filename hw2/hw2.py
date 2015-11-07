#!/usr/bin/env/python
"""

"""

from sys import path
path.insert(0,'./libsvm-320/python/')
## in ./libsvm-320/python/
from svmutil import *
from sklearn import svm, cross_validation
import numpy as np

def part_a():

    y, x = svm_read_problem('./splice_noise_train_scaled.txt')

    results = {}
    for p in range(1,6,2):
        results[p] = {}
        for c in range(-5,6):
            print 'training with polynomial kernel degree {} and c = 5^{}'.format(p,c)
            c = 5 ** c
            results[p][c] = svm_train(y,x,'-v 10 -t 1 -p {} -c {}'.format(p, c))

    return results

## save time
part_a_results = \
{1: {0.00032: 76.0421052631579,
  0.0016: 77.05263157894737,
  0.008: 79.15789473684211,
  0.04: 77.43157894736842,
  0.2: 74.10526315789474,
  1: 70.86315789473684,
  5: 69.38947368421053,
  25: 70.02105263157895,
  125: 69.6421052631579,
  625: 69.47368421052632,
  3125: 68.50526315789473},
 3: {0.00032: 75.87368421052632,
  0.0016: 77.38947368421053,
  0.008: 79.2,
  0.04: 77.85263157894737,
  0.2: 73.26315789473684,
  1: 71.66315789473684,
  5: 69.81052631578947,
  25: 68.50526315789473,
  125: 68.8,
  625: 69.26315789473684,
  3125: 68.25263157894737},
 5: {0.00032: 76.0,
  0.0016: 77.30526315789474,
  0.008: 78.90526315789474,
  0.04: 78.10526315789474,
  0.2: 73.22105263157894,
  1: 70.82105263157895,
  5: 69.72631578947369,
  25: 68.29473684210527,
  125: 68.63157894736842,
  625: 69.05263157894737,
  3125: 68.88421052631578}}

def sk_part_a():

    y, x = svm_read_problem('./splice_noise_train_scaled.txt')
    y = np.array(y)
    x = np.array([_.values() for _ in x])

    kf = cross_validation.KFold(len(x), n_folds=10)

    results = {}

    for degree in range(1,6,2):
        results[degree] = {}
        for c in range(-5,6):
            avg_cv_accuracy = 0
            C = 5 ** c
            clf = svm.SVC(kernel='poly', degree=degree, C=C)
            print 'poly kernel degree {}, C = 5^{}'.format(degree, c) 
            for train, test in kf:
                x_train, x_test, y_train, y_test =\
                    x[train], x[test], y[train], y[test]

                clf.fit(x_train,y_train)

                avg_cv_accuracy += clf.score(x_test, y_test)

            avg_cv_accuracy = avg_cv_accuracy / kf.n_folds

            results[degree][C] = avg_cv_accuracy
            
            print avg_cv_accuracy

    return results


def part_b():

    y, x = svm_read_problem('./splice_noise_train.txt')

    results = {}
    for g in range(1,6,2):
        results[g] = {}
        for c in range(-5,6):
            print 'training with gaussian kernel with gamma 2^{} and c = 5^{}'.format(g,c)
            c = 5 ** c
            g = 2 ** g
            results[g][c] = svm_train(y,x,'-q -v 10 -t 2 -g {} -c {}'.format(g, c))

    return results

def plot_results(results):

    pass

if __name__ == 'main':

    part_a()
