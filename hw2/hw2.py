#!/usr/bin/env python
"""

"""

import csv
from sys import path
## what to do for mapreduce?
path.insert(0,'/Users/davemo88/libsvm-320/python/')
from svmutil import *
import numpy as np

y, x = svm_read_problem('./splice_noise_train_scaled.txt')

y_test, x_test = svm_read_problem('./splice_noise_test_scaled.txt')

def polynomial_kernels(output_filename='polydata.csv'):

    with open(output_filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['C','d1','d3','d5'])
        for k in range(-5,6):
            C = 5 ** k
            row = [C]
            for d in range(1,6,2):
                print 'training with polynomial kernel degree {} and C = 5^{}'.format(d,k)
                row.append(svm_train(y,x,'-q -v 10 -t 1 -d {} -c {}'.format(d, C)))
            writer.writerow(row)

def get_poly_model(p, C):

    return svm_train(y,x,'-q -t 1 -p {} -c {}'.format(p, C))

def gaussian_kernels(output_filename='gaussdata.csv'):

    with open(output_filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['C','g1','g3','g5'])
        for k in range(-5,6):
            C = 5 ** k
            row = [C]
            for p in range(1,7,2):
                g = 1.0/(2 ** (p * 2 + 1))
                print 'training with gaussian kernel with sig 2^{} and C = 5^{}'.format(p,k)
                row.append(svm_train(y,x,'-q -v 10 -t 2 -g {} -c {}'.format(g, C)))
            writer.writerow(row)

def get_polygauss_kernel(degree=3, gamma=0.0078125):

    k = np.empty((len(x), len(x)))

    for i in range(len(x)):
        for j in range(len(x)):

            u = np.array(x[i].values())

            v = np.array(x[j].values()) 

            k[i][j] = np.exp(-gamma * (np.linalg.norm(u-v)**2)) + (np.dot(u,v)**degree)

    return k

def get_margin(w,rho,svs):

    return max([abs(_) for _ in map(lambda v: np.dot(w, v) + rho, svs)])

def get_correctly_classified_vectors(y,x,w,rho):

    ccvs = []

    for i in range(len(x)):

        if predict_with_w(w,rho,x[i].values()) == y[i]:

            ccvs.append(x)

    return ccvs

def predict_with_w(w,rho,example):

    return np.sign(np.dot(w,np.array(example)) + rho)
    
def get_w_and_rho(model):

    coef = np.array([_[0] for _ in model.get_sv_coef()])

    svs = np.array(clean_sv(model.get_SV()))

    w = np.dot(np.transpose(svs), coef)

## normalize w
    w = w / np.linalg.norm(w)

    return w, -model.rho.contents.value

def clean_sv(support_vectors):

    svs = []

    for sv in support_vectors:
## not sure what key -1 is
        sv.pop(-1)
        svs.append(sv.values())

    return svs

def sk_polynomial_kernels():

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

if __name__ == '__main__':

    pass
