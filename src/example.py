import numpy as np
import control as ctrl

from models import sys_variables
from controllers import delay_lqr, augment

# F1-tenth car system
sys = sys_variables['F1']

# Augment the system with a one-period control delay
sys_delay = augment(ctrl.c2d(sys, 0.1, method='foh'))

# Design a discrete-time LQR controller for the augmented system
K = delay_lqr(sys, 0.1)

print(f"Original system:\n{sys}\n")
print(f"Augmented system:\n{sys_delay}\n")
print(f"Controller gain:\n{K}\n")
