#    Write a gradient descent program that optimizes the following loss function for variables (or weights) x and y

#   z = sin(x) + cos(y)

#    where x and y start with integer values between -5 and 5, i.e. the x values and y values are [-4, -2, 0, 2, 4]. So there are 25 starting points (-4,-4), (-4,-2), (-4, 0) ... (4,4).

#    Find Gradient Loss Function

#    Thanks to Isaac R's Slack comment I reread and realized that it was sin(x) + cos(y) and not sin(x) + cos(x) like I had read originally.

#    From the book on page 178 the function for calculating the gradent is ∇θ MSE(θ) = (2/n)XT(Xθ - y)

#    Wasn't sure what all of the terms in that formula meant, even after reading the chapter. Turns out that https://arjun-mota.github.io/posts/batch-gradient-descent/ has a good breakdown. Mainly I wasn't sure what the X's were and it turns out that it is the matrix of the values being optimized.

#2    List item

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from matplotlib import cm
from matplotlib.ticker import LinearLocator

loss_x = lambda X: X - (math.cos(X) * learningRate)
loss_y = lambda Y: Y + (math.sin(Y) * learningRate)
 # negative sin minus the value is plus
theta = [random.randint(-4,4), random.randint(-4,4)]
learningRate = 0.1
history = []
for i in range(50):
  history.append(np.array(theta))
  theta = [loss_x(theta[0]), loss_y(theta[1])]
history.append(theta)

for i in range(len(history)):
  history[i] = np.append(history[i], np.array(math.sin(history[i][0]) + math.cos(history[i][1])))
gistory = np.array(history)
point_x = gistory[:, 0]
point_y = gistory[:, 1]
point_z = gistory[:, 2]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,5))

# Make data.
X = np.arange(-5, 5, 0.01)
Y = np.arange(-5, 5, 0.01)
X, Y = np.meshgrid(X, Y)
Z = np.sin(X) + np.cos(Y)

# Plot the surface.
wire = ax.plot_wireframe(X, Y, Z, color="green", alpha=0.15)
plt.plot(point_x, point_y, point_z, color='red' )

plt.show()
