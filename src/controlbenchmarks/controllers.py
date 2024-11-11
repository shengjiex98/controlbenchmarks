import numpy as np
import control as ctrl

def delay_lqr(dsys_augmented: ctrl.StateSpace, Q: np.ndarray = None, R: np.ndarray = None):
    """Design a discrete-time LQR controller for an augmented system with a one-period control delay."""
    p = dsys_augmented.nstates - dsys_augmented.ninputs
    q = dsys_augmented.ninputs
    if Q is None:
        Q = np.pad(np.eye(p), (0, q))
    if R is None:
        R = np.eye(q)
    K, _, _ = ctrl.lqr(dsys_augmented, Q, R)
    return K

def pole_place(dsys_augmented: ctrl.StateSpace, pole: float = 0.9):
    """Design a discrete-time pole-placement controller for a system with a one-period control delay.
    (need further analysis for setting the pole vector)"""
    p, q = dsys_augmented.nstates - dsys_augmented.ninputs, dsys_augmented.ninputs
    p_vec = np.concatenate((np.zeros(q), np.full(p, pole)))
    K = ctrl.place(dsys_augmented.A, dsys_augmented.B, p_vec)
    return K

def augment(dsys: ctrl.StateSpace):
    """Augment a discrete-time state-space model with a one-period control delay."""
    p = dsys.nstates
    q = dsys.ninputs
    r = dsys.noutputs
    A = np.block([[dsys.A, dsys.B], [np.zeros((q, p + q))]])
    B = np.block([[np.zeros((p, q))], [np.eye(q)]])
    C = np.block([dsys.C, np.zeros((r, q))])
    D = dsys.D
    Ts = dsys.dt
    return ctrl.StateSpace(A, B, C, D, Ts)
