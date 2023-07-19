import networkx as nx
import matplotlib.pyplot as plt


xlim = 4

# define x
x = [0]
val = 0
while True:
    val += 0.01
    if val > xlim:
        break
    x.append(val)

grid = []
tree = []
for i in x:
    grid.append(2 * i * i + 2 * i + 1)
    tree.append((3 ** (i + 1) - 1) / 2)


plt.plot(x, grid)
plt.plot(x, tree)
plt.legend(["grid", "tree"])
plt.xlabel("number of steps")
plt.ylabel("nodes visited")
plt.show()