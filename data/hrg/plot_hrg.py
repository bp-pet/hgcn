import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


noise_factor = 0.0

with open("data\\hrg\\hrg.hyp", 'r') as f:
    hyp_file_str = f.read()

coordinates = []
for line in hyp_file_str.split("\n"):
    if len(line) == 0:
        break
    s = line.split(" ")
    c1 = float(s[0])
    c2 = float(s[1])
    coordinates.append((c1, c2))

coordinates_arr = np.array(coordinates)
coordinates_df = pd.DataFrame(coordinates_arr, columns=["r", "theta"])

coordinates_df["x"] = coordinates_df.apply(lambda row: row["r"] * np.cos(row["theta"]), axis=1)
coordinates_df["y"] = coordinates_df.apply(lambda row: row["r"] * np.sin(row["theta"]), axis=1)

maximum_r = coordinates_df["r"].max()
coordinates_df["x"] = coordinates_df["x"] + np.random.normal(0, noise_factor * maximum_r, (coordinates_df.shape[0]))
coordinates_df["y"] = coordinates_df["y"] + np.random.normal(0, noise_factor * maximum_r, (coordinates_df.shape[0]))

with open("data\\hrg\\hrg.txt", 'r') as f:
    graph_file_str = f.read()

edges = []
for line in graph_file_str.split("\n")[2:]:
    if len(line) == 0:
        break
    s = line.split(" ")
    c1 = int(s[0])
    c2 = int(s[1])
    edges.append((c1, c2))

for edge in edges:
    node1 = coordinates_df.iloc[edge[0]]
    node2 = coordinates_df.iloc[edge[1]]
    plt.plot((node1["x"], node2["x"]), (node1["y"], node2["y"]), c='b', alpha=0.1)

for ind in coordinates_df.index:
    plt.scatter(coordinates_df["x"][ind], coordinates_df["y"][ind], marker='.', c='r')

# print(f"maximum r: {maximum_r}")

maximum_r = 2 * np.log2(100) + 10

# theta_range = np.linspace(0, 2*np.pi, 100)
# x = maximum_r * np.cos(theta_range)
# y = maximum_r * np.sin(theta_range)
# plt.plot(x,y, c='k')



plt.axis('equal')
plt.savefig(f"images\\hrg_temp.png")
