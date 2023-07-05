import matplotlib.pyplot as plt
import pandas as pd

curvature = pd.read_csv("curvature\\curvature.csv", index_col=False)
num_cols = len(curvature.columns) - 1

# plot curvatures
fig, axes = plt.subplots(nrows=1, ncols=2)
for i in range(num_cols):
    curvature.rename(columns={str(i): f"Layer {i + 1}"}, inplace=True)
for i in range(num_cols):
    curvature.plot(ax=axes[0], y=f"Layer {i + 1}")

# plot average
curvature["Average"] = curvature.drop(columns=["loss"]).mean(axis=1)
curvature.plot(ax=axes[0], y="Average", linestyle='dashed', color='gray')

# plot loss
curvature.plot(ax=axes[1], y="loss")

# save
axes[0].title.set_text("Layer curvature")
axes[1].title.set_text("Loss")
axes[1].legend([])
plt.savefig("curvature\\curvature.png")