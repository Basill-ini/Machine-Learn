#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None   # весовой вектор
        self.w0_ = None # смещение
        self.cost_ = None # история штрафов в процессе обучения
        
    def net_input(self, X):
        return np.dot(X, self.w_) + self.w0_
    
    def activate(self, z):
        return z
    
    def theta(self, phi):
        return np.where( phi >= 0, 1, -1 )

    def predict(self, X):
        z = self.net_input(X)
        phi = self.activate(z)
        return self.theta(phi)
    
    def fit(self, X, y):
        # инициализация процесса обучения
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.w0_ = rgen.normal(loc=0.0, scale=0.01, size=None)
        self.cost_ = []
        
        # обучаем
        for _ in range(0, self.n_iter):
            net_input = self.net_input(X)
            phi = self.activate(net_input)
            errors = ( y - phi )
            self.w_ += self.eta * X.T.dot(errors)
            self.w0_ += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
        
    
    
    
    
    