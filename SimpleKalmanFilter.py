# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 00:22:26 2020

@author: sstucker
"""

import time
import numpy as np
import matplotlib.pyplot as plt

class SimpleKalmanFilter:

    def __init__(self, N, A, B, C, Q, R, X0, P0, U0) -> None:
        self.n = np.size(X0)  # Number of variables on each state X
        self.h = np.identity(N)
        self.a = A  # State evolution transition matrix
        self.b = B  # Control transition matrix
        self.c = C  # Measurement matrix
        self.q = Q  # Process covariance
        self.r = R  # Measurement covariance

        self.Xt = X0  # Current state
        self._Xtp1 = None  # New state
        self.Pt = P0  # Current state covariance matrix
        self._Ptp1 = None  # New state covariance
        self.Ut = U0  # Control variables
        self.Yt = None  # Current observation
        
        self.K = None  # Current gain
        
        
    def _predict(self) -> None:
        self._Xtp1 = self.a.dot(self.Xt) + self.b.dot(self.Ut)  # + w
        self._Ptp1 = self.a.dot(self.Pt).dot(self.a.T) + self.q
        
        self.Xt = self._Xtp1
        self.Pt = self._Ptp1
        
        
    def _update(self) -> None:
        
        S = np.dot(np.dot(self.h, self.Pt), self.h.T) + self.r
        self.K = self.Pt.dot(self.h.T).dot(np.linalg.inv(S))
        
        self._Xtp1 = self.Xt + self.K.dot(self.Yt)
        self._Ptp1 = np.eye(self.n) - self.K.dot(self.h).dot(self.Pt)
        
        self.Xt = self._Xtp1
        self.Pt = self._Ptp1

    def measure(self, measured_value, measured_variance) -> None:
        self.Yt = measured_value - self.h.dot(self.Xt)  # + z
        self.r = measured_variance

    def update_control(self, u) -> None:
        self.Ut = u

    def update_trans_matrix(self, a) -> None:
        self.a = a
        
    def get_state(self):
        return self.Xt
        
    def estimate(self):
        self._predict()
        self._update()
        return self.Xt


if __name__ is "__main__":
    
    test_sig_n = 1000
    t = np.linspace(0, 2*np.pi, test_sig_n)
    omega = np.linspace(1, 6, test_sig_n)
    v_observations = 8 * np.sin(t * omega) + np.random.randn(test_sig_n) * 2
    
    dt = 1  # Time between measurements
    
    N = 2  # x, v
    Z = 1
    
    # Measurement function
    C = np.array([[0],  # x
                  [1]]) # v
    
    # State evolution matrix
    A = np.array([[1, dt],
                  [0, 1]])
    
    # Control transition matrix
    B = np.array([[0],
                  [0]])
    U = 0
    
    # Initial state
    init_X = np.array([[0],    # x
                       [0]])   # v
    
    init_P = 0.01 * np.eye(N)
    Q = 0.4 * np.eye(N)
    R =  2
    
    filt = SimpleKalmanFilter(N, A, B, C, Q, R, init_X, init_P, U)
    
    ests = []
    models = []
    ys = []
    
    start = time.time()
    for i in range(len(v_observations)):
        y = v_observations[i]
        filt.measure(y, R)
        est = filt.estimate()
        
        models.append(filt.get_state())
        ests.append(est)
        ys.append(y)
    
    elapsed = time.time() - start
    print('Kalman filter rate', len(v_observations) / elapsed, 'hz')
    
    plt.close()
    plt.plot(np.array(ys), '-k', label='obs')
    plt.plot(np.array(models)[:, 1], '-y', label='modeled velocity')
    plt.plot(np.array(ests)[:, 1],'-r', label='est. velocity', alpha=0.5)
    # plt.plot(np.array(ests)[:, 0], '--b', label='est. position')
    plt.legend()





