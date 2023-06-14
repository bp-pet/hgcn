# model paraameters
# number of nodes, alpha, temperature, average degree
n=100
alpha=0.75
t=0.8
deg=10
# output settings
# filename, save adj. list, save hyp. coordinates
file="data\hrg\hrg100"
edge=1
coord=1
# random seeds
# radii seed, angle seed, sampling seed
rseed=12
aseed=130
sseed=1400
# running settings
# number of threads, use NetworKit R
threads=1
nkr=0
"C:\Program Files\girgs\genhrg" -n $n -alpha $alpha -t $t -deg $deg -rseed $rseed -aseed $aseed -sseed $sseed -threads $threads -nkr $nkr -file $file -edge $edge -coord $coord