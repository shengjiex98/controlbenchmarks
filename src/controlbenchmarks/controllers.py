import numpy as np
import control as ctrl
from typing import Callable


class AbstractController():
    def __init__(self, sys, h):
        self.sys = sys
        self.h = h

    def __call__(self, x, t):
        raise NotImplementedError
    
class DelayLQR(AbstractController):
    def __init__(self, sys: ctrl.StateSpace, h: float, Q=None, R=None):
        """Design a discrete-time LQR controller for a system with a one-period control delay."""
        super().__init__(sys, h)
        self.K = delay_lqr(sys, h, Q, R)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.K @ x
    
class PolePlacement(AbstractController):
    def __init__(self, sys: ctrl.StateSpace, h: float, pole=0.9):
        """Design a discrete-time pole-placement controller for a system with a one-period control delay."""
        super().__init__(sys, h)
        p, q = dsys_delay.B.shape
        dsys_delay = ctrl.sample_system(sys, h, method="foh")
        p_vec = np.concatenate(([0], np.full(p, pole)))
        self.K = ctrl.place(dsys_delay.A, dsys_delay.B, p_vec)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.K @ x

def delay_lqr(dsys_augmented: ctrl.StateSpace, Q: np.ndarray = None, R: np.ndarray = None):
    """Design a discrete-time LQR controller for an augmented system with a one-period control delay."""
    p = dsys_augmented.nstates - dsys_augmented.ninputs
    q = dsys_augmented.ninputs
    if Q is None:
        Q = np.eye(p)
    if R is None:
        R = np.eye(q)
    Q_augmented = np.block([[Q, np.zeros((p, q))], [np.zeros((q, p + q))]])
    K, _, _ = ctrl.lqr(dsys_augmented, np.eye(p+q), R)
    return K

def pole_place(dsys_augmented: ctrl.StateSpace, pole: float = 0.9):
    """Design a discrete-time pole-placement controller for a system with a one-period control delay."""
    p, q = dsys_augmented.nstates - dsys_augmented.ninputs, dsys_augmented.ninputs
    p_vec = np.concatenate((np.zeros(q), np.full(p, pole)))
    K = ctrl.place(dsys_augmented.A, dsys_augmented.B, p_vec)
    return K

def augment(dsys: ctrl.StateSpace):
    """Augment a discrete-time state-space model with a one-period control delay."""
    p = dsys.A.shape[0]
    q = dsys.B.shape[1]
    r = dsys.C.shape[0]
    A = np.block([[dsys.A, dsys.B], [np.zeros((q, p + q))]])
    B = np.block([[np.zeros((p, q))], [np.eye(q)]])
    C = np.block([dsys.C, np.zeros((r, q))])
    D = dsys.D
    Ts = dsys.dt
    return ctrl.StateSpace(A, B, C, D, Ts)
