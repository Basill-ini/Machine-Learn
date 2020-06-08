#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# S. Rashka. Python Machine Learning


class AdalineSGD(object):

    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle
        self.w_initialized = False
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
        self._initialize_weights(n_features=X.shape[1])
        
        # обучаем
        for _ in range(0, self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X,y)
            costs = []
            for xi, yi in zip(X,y):
                c = self._update_weights(xi, yi)
                costs.append(c)
            avg_cost = sum(costs) / len(costs)
            self.cost_.append(avg_cost)
            
        # обучение закончено
        return self
        
    def _initialize_weights(self, n_features):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self.w0_ = self.rgen.normal(loc=0.0, scale=0.01, size=None)
        self.cost_ = []
        self.w_initialized = True
        
    def _update_weights(self, xi, yi):
        net_input = self.net_input(xi)
        phi = self.activate(net_input)
        error = yi - phi
        self.w_ += self.eta * xi * error
        self.w0_ += self.eta * error
        cost = error**2 / 2.0
        return cost
    
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    