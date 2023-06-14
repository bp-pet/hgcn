import matplotlib.pyplot as plt
import pandas as pd

curvature = pd.read_csv("curvature\\curvature.csv", header=None, index_col=False)
curvature["average"] = curvature.mean(axis=1)

# make plot
# for i in curvature.columns:
curvature.plot(y=curvature.columns)
plt.savefig("curvature\\curvature.png")