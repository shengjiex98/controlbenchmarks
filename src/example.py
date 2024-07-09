import numpy as np
import control as ctrl

from models import sys_variables
from controllers import delay_lqr, augment

sys = sys_variables['F1']
sys_delay = augment(ctrl.c2d(sys, 0.1, method='foh'))
K = delay_lqr(sys, 0.1)

print(f"Original system:\n{sys}\n")
print(f"Augmented system:\n{sys_delay}\n")
print(f"Controller gain:\n{K}\n")
