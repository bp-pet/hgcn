import matplotlib.pyplot as plt


colors = ['r', 'b', 'g', 'm']

with open("curvature.txt", 'r') as f:
    lines = f.readlines()

num_lines = len(lines)

while True:
    if len(lines[0].split(" ")) != 4:
        print("here")
        lines = lines[1:]
    else:
        break


for i, line in enumerate(lines):
    if i != 0 and i % 49 != 0:
        continue
    print(f"{i + 1} out of {num_lines}")
    temp = line.split(" ")
    if len(temp) != 4:
        continue
    layer = int(temp[1])
    curvature = float(temp[3])
    plt.scatter(i, curvature, c=colors[layer])

plt.savefig("curvature.png")