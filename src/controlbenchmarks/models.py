import numpy as np
import control as ctrl

def _sys_rc():
    """Resistor-capacitor network"""
    r_1 = 100000
    r_2 = 500000
    r_3 = 200000
    c_1 = 0.000002
    c_2 = 0.000010
    A = np.array([[-1/c_1 * (1/r_1 + 1/r_2), 1/(r_2*c_1)],
                 [1/(r_2*c_2), -1/c_2 * (1/r_2 + 1/r_3)]])
    B = np.array([[1/(r_1*c_1)], [0]])
    C = np.eye(2)
    D = np.zeros((2, 1))

    return ctrl.StateSpace(A, B, C, D)

def _sys_f1():
    """F1-tenth car"""
    v = 6.5
    L = 0.3302
    A = np.array([[0, v], [0, 0]])
    B = np.array([[0], [v/L]])
    C = np.array([[1, 0]])
    D = np.zeros((1, 1))

    return ctrl.StateSpace(A, B, C, D)

def _sys_dc():
    """DC motor"""
    A = np.array([[-10, 1], [-0.02, -2]])
    B = np.array([[0], [2]])
    C = np.array([[1, 0]])
    D = np.zeros((1, 1))

    return ctrl.StateSpace(A, B, C, D)

def _sys_cs():
    """Car suspension system"""
    A = np.array([[0., 1., 0., 0.],
                    [-8., -4., 8., 4.],
                    [0., 0., 0., 1.],
                    [80., 40., -160., -60.]])
    B = np.array([[0.], [80.], [20.], [-1120.]])
    C = np.array([[1, 0, 0, 0]])
    D = np.zeros((1, 1))

    return ctrl.StateSpace(A, B, C, D)

def _sys_ew():
    """Electronic wedge brake"""
    A = np.array([[0, 1], [8.3951e3, 0]])
    B = np.array([[0], [4.0451]])
    C = np.array([[7.9920e3, 0]])
    D = np.zeros((1, 1))

    return ctrl.StateSpace(A, B, C, D)

def _sys_c1():
    """Cruise control 1"""
    A = np.array([[-0.05]])
    B = np.array([[0.01]])
    C = np.array([[1]])
    D = np.zeros((1, 1))

    return ctrl.StateSpace(A, B, C, D)

def _sys_cc():
    """Cruise control 2"""
    A = np.array([[0, 1, 0], [0, 0, 1], [-6.0476, -5.2856, -0.238]])
    B = np.array([[0], [0], [2.4767]])
    C = np.array([[1, 0, 0]])
    D = np.zeros((1, 1))

    return ctrl.StateSpace(A, B, C, D)

# Dictionary of systems
sys_variables = {
    'RC': _sys_rc(),
    'F1': _sys_f1(),
    'DC': _sys_dc(),
    'CS': _sys_cs(),
    'EW': _sys_ew(),
    'C1': _sys_c1(),
    'CC': _sys_cc()
}
