from random import shuffle

n_experiments = 1000000

bulbs = [True] * 6 + [False] * 4

success = 0

for _ in range(n_experiments):
    shuffle(bulbs)
    counter = 0
    for i in range(5):
        if not bulbs[i]:
            counter += 1
    if counter == 2:
        success += 1

print(success)
print(success / n_experiments)