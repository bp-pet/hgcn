import math

def mean(l):
    return sum(l) / len(l)

def std(l):
    m = mean(l)
    s = 0
    for i in l:
        s += (i - m) ** 2 / (len(l) - 1)
    s = math.sqrt(s)
    return s


l = [0.926, 0.929, 0.927, 0.929, 0.921]
print(mean(l))
print(std(l))

def error_red(gcn, hgcn):
    error_gcn = 100 - gcn
    error_hgcn = 100 - hgcn
    diff = error_gcn - error_hgcn
    red = diff / error_gcn
    return -red * 100

# print(error_red(gcn=87.4, hgcn=93.1), "cora")
# print(error_red(gcn=91.1, hgcn=96.3), "pubmed")
# print(error_red(gcn=91.5, hgcn=94.8), "airport")
# print(error_red(gcn=56.9, hgcn=63.9), "disease")
# print(error_red(gcn=96.4, hgcn=97.3), "road")
# print(error_red(gcn=86.8, hgcn=97.1), "hrg")
# print(error_red(gcn=90.1, hgcn=91.2), "lfr")
# print(error_red(gcn=92.4, hgcn=92.6), "sbm")


# print(error_red(gcn=90.4, hgcn=92.9), "cora paper")
# print(error_red(gcn=89.3, hgcn=96.4), "airport paper")
# print(error_red(gcn=66.0, hgcn=78.1), "disease paper")



print(error_red(gcn=74.1, hgcn=93.2), "hrg 2 dim")
print(error_red(gcn=86.8, hgcn=97.1), "hrg 16 dim")
print(error_red(gcn=89.0, hgcn=97.3), "hrg 128 dim")
print(error_red(gcn=91.3, hgcn=97.4), "hrg 1024 dim")

print(error_red(gcn=92.1, hgcn=92.7), "sbm 2 dim")
print(error_red(gcn=92.4, hgcn=92.6), "sbm 16 dim")
print(error_red(gcn=92.8, hgcn=92.6), "sbm 128 dim")
print(error_red(gcn=92.5, hgcn=92.8), "sbm 1024 dim")