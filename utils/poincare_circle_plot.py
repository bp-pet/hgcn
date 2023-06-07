"""
Module for plotting Poincare disk.
"""

import matplotlib.pyplot as plt
import numpy as np


center = np.array([0, 0.9])
radius = 3

theta = np.linspace(0, 2*np.pi, 100)
r = 1
x = r * np.cos(theta)
y = r * np.sin(theta)
plt.plot(x,y, c='k')
plt.axis('equal')

def distance(a, b):
    bottom = (1 - np.linalg.norm(a)**2) * (1 - np.linalg.norm(b)**2)
    result =  np.arccosh(1 + 2 * np.linalg.norm(a - b)**2 / bottom)
    return result


all_points = np.array([[0.0, 0.0]])
boundary = 1
granularity = 0.01

x_vals = np.arange(-boundary, boundary, granularity)
y_vals = np.arange(-boundary, boundary, granularity)

for x in x_vals:
    for y in y_vals:
        if x**2 + y**2 <= 1:
            all_points = np.concatenate([all_points, np.array([[x, y]])], axis=0)



for i in range(all_points.shape[0]):
    point = all_points[i]
    d = distance(point, center)
    if distance(point, center) < 0.5 * radius:
        plt.scatter(point[0], point[1], c='g')
    if 0.5 * radius < d < 0.6 * radius:
        plt.scatter(point[0], point[1], c='r')
    elif 0.6 * radius < d < 0.7 * radius:
        plt.scatter(point[0], point[1], c='b')
    elif 0.7 * radius < d < 0.8 * radius:
        plt.scatter(point[0], point[1], c='g')
    elif 0.8 * radius < d < 0.9 * radius:
        plt.scatter(point[0], point[1], c='r')
    elif 0.9 * radius < d < radius:
        plt.scatter(point[0], point[1], c='b')

plt.scatter(center[0], center[1], marker='x', c='k')

plt.title(f"Circles with radii 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 respectively,\ncenter at {center[0], center[1]}")

plt.savefig("disk.png")