import time
import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    
    def __init__(self, A, B, H, Q, R, X0, P0):
        
        self._A = A  # State evolution matrix
        self._B = B  # Control matrix
        self._H = H  # Measurement matrix
        self._Q = Q  # Action uncertainty matrix
        self._R = R  # Sensor noise matrix
        
        self._X = X0  # Init state
        self._P = P0  # Init cov matrix
        
        self._K = None  # Kalman gain begins undefined

        self._I = np.eye(np.shape(self._A)[0])

    def predict(self, u=None):
        # TODO implement control
        if u is not None:
            self._X = self._A.dot(self._X) + self._B.dot(u)  # State evolution
        else:
            self._X = self._A.dot(self._X)
        # Cov error prediction
        self._P = self._A.dot(self._P).dot(self._A.T) + self._Q
        self._P = (self._P + self._P.T) / 2   # Ensure symmetry
    
    def update(self, m, q=None, r=None):
        if q is not None:
            self._Q = q
        if r is not None:
            self._R = r
        y = m - self._H.dot(self._X)  # Measurement
        try:  # Compute Kalman gain
            S = self._H.dot(self._P.dot(self._H.T)) + self._R
            self._K = self._P.dot(self._H.T).dot(np.linalg.inv(S))
        except np.linalg.LinAlgError:
            print('Cannot invert singular matrix', S)
            return
        self._X = self._X + self._K.dot(y)
        self._P = (self._I - self._K.dot(self._H)).dot(self._P)
    
    def get_state(self):
        return self._X


if __name__ is "__main__":
    
    figdex = 1
    
    # Generate test data
    test_sig_n = 1000
    t = np.linspace(0, 2*np.pi, test_sig_n)
    omega = np.linspace(1, 30, test_sig_n)
    x_truth = 8 * np.sin(t * omega)
    y_truth = 4 * np.sin(t * omega)
    z_truth = 7 * np.sin(t * omega)
    x_obs = x_truth + np.random.randn(test_sig_n) * 2
    y_obs = y_truth + np.random.randn(test_sig_n) * 1
    z_obs = z_truth + np.random.randn(test_sig_n) * 3
    
    print('std x', np.std(x_obs))
    print('std y', np.std(y_obs))
    
    dt = 1  # Time between measurements 
    d = 1  # Velocity decay
    g = 1  # Acceleration decay
    
    # State evolution matrix
    A = np.array([[d, dt, 0, 0, 0, 0],
                  [0, g, 0, 0, 0, 0],
                  [0, 0, d, dt, 0, 0],
                  [0, 0, 0, g, 0, 0],
                  [0, 0, 0, 0, d, dt], 
                  [0, 0, 0, 0, 0, g]]) 
    
    # Control transition matrix
    B = np.zeros([6, 6])

    # Measurement function (measuring velocity)
    H = np.array([[0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1]])
    
    q = 28  # Empirical process noise value
        
    # Process noise
    Q = np.eye(6) * q

    
    # Measurement noise
    vs = np.concatenate([x_obs[1::] - x_obs[0::-1], y_obs[1::] - y_obs[0::-1], z_obs[1::] - z_obs[0::-1]])
    R = np.eye(3) * np.std(vs)**2
#                R = np.ones([3, 3]) * 0.1 + np.eye(3) * np.std(np.concatenate([x_obs, y_obs, z_obs])) ** 2
    

    # filt = SimpleKalmanFilter(N, A, B, C, Q, R, init_X, init_P, U)
    filt = KalmanFilter(A, B, H, Q, R, X0=np.zeros(6).T, P0=np.zeros([6, 6]))
    
    truth = []
    ests = []
    start = time.time()
    lastx = 0
    lasty = 0
    lastz = 0
    for i in range(len(x_obs)):
        vx = (x_obs[i] - lastx) / dt
        vy = (y_obs[i] - lasty) / dt
        vz = (z_obs[i] - lastz) / dt
        y = np.array([vx, vy, vz])
        lastx = x_obs[i]
        lasty = y_obs[i]
        lastz = z_obs[i]
        filt.predict()
        filt.update(y)
        est = filt.get_state()
        if np.isnan(est[0]):
            break
        ests.append(est)
    
    elapsed = time.time() - start
    print('Kalman filter rate', len(x_obs) / elapsed, 'hz')
    
    plt.close(figdex)
    plt.figure(figdex)
    figdex += 1
    plt.title('q = ' + str(q) + ', g =' + str(g) + ', d =' + str(d))
    plt.scatter(np.arange(len(x_obs)), np.array(x_obs), c='r', s=0.2, alpha=0.5, label='x obs')
    plt.scatter(np.arange(len(y_obs)), np.array(y_obs), c='b', s=0.2, alpha=0.5, label='y obs')
    plt.scatter(np.arange(len(z_obs)), np.array(z_obs), c='y', s=0.2, alpha=0.5, label='z obs')
    plt.plot(x_truth, '--r', alpha=0.5, label='x truth')
    plt.plot(y_truth, '--b', alpha=0.5, label='y truth')
    plt.plot(z_truth, '--y', alpha=0.5, label='z truth')
    plt.plot(np.array(ests)[:, 0], '-r', label='x est')
    plt.plot(np.array(ests)[:, 2], '-b', label='y est')
    plt.plot(np.array(ests)[:, 4], '-y', label='z est')
    plt.legend()
    plt.xlim(800, 900)

        



