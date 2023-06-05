from activation_functions import softmax, softmax_backward
import numpy as np

z = np.array([1.44, -0.4, 0.23])

la = softmax(z)[0]

print("end")