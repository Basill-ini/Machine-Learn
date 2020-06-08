#!/usr/bin/env python3
# -*- coding:utf-8

# eta - обучающее отношение
# n_iter - полное количество проходов
# random_state - инициализатор случайных чисел 

import numpy as np


class Perceptron(object):
    
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        # Для библиотеки scikit параметры алгоритма принято задавать открытыми 
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None # весовой вектор
        self.w0_ = None # смещение
        self.errors_ = None # история ошибок в процессе обучения
        
    def net_input(self, X):
        # скалярное произведение 
        return np.dot(X, self.w_) + self.w0_
    
    def activate(self, z):
        return np.where( z>=0 , 1, -1 )
    
    def theta(self, phi):
        return np.where(phi>=0, 1,-1)
    
    def predict(self, X):
        z = self.net_input(X)
        phi = self.activate(z)
        return self.theta(phi)
    
    def fit(self, X, y):
        # иницализация процесса обучения
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=1.0, size=X.shape[1])
        self.w0_ = rgen.normal(loc=0.0, scale=1.0, size=None)
        self.errors_ = []
        
        # обучение
        for _ in range(0, self.n_iter):
            errors = 0
            for xi, trg in zip(X,y):
                trg_pred = self.predict(xi)
                update = (trg - trg_pred)*self.eta
                self.w_ += update * xi
                self.w0_ += update
                errors += int(update != 0)
            self.errors_.append(errors)
        
        # обучение закончено
        return self