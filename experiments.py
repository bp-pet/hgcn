"""
Module for running experiments.
"""
import numpy as np
import pandas as pd

from train import train
from config import parser

from curvature.plot_c import main as plot_c

num_runs = {"GCN": 10, "HGCN": 10}

# Ts = [0.0, 0.2, 0.4, 0.6, 0.8, -1]
Ts = [0, 1]
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, -1]

# Ts = [0.0]
# alphas = [0.0]

dims = [2, 16, 128, 1024]


def dimensionality_experiments(args):

    for model in ["GCN", "HGCN"]:

        # for SBM experiments
        if model == "GCN":
            args.dropout = 0.2
        else:
            args.dropout = 0

        args.model = model
        print(model)
        for dim in dims:
            print(f"\tDimension {dim}")
            args.dim = dim
            avg, std = multiple_runs(args, 5)
            avg = round(avg, 3) * 100
            std = round(std, 3) * 100
            with open("results\\dimensionality.txt", 'a') as f:
                f.write(f"\n{model}, {dim} dimensions: {avg} +- {std}")
    



def std(l):
    m = sum(l) / len(l)
    s = 0
    for i in l:
        s += (i - m) ** 2 / (len(l) - 1)
    s = np.sqrt(s)
    return s

def multiple_runs(args, n=20):
    """
    Run same configuration multiple times, save test ROC or accuracy. Return average.

    The n runs are random but the seed for each is based on the original given seed.
    """
    # seed = np.random.randint(0, 100)

    initial_seed = args.seed

    if args.task == "lp":
        metric = "roc"
    else:
        metric = "accuracy" # might not be correct

    results = []
    for i in range(n):
        print(f"\tRun {i + 1} of {n}")
        args.seed = (i + 1) * initial_seed
        try:
            results.append(train(args)[metric])
        except RuntimeError as e:
            print("Error")
            with open("error_log.txt", 'a') as f:
                f.write(f"\n{args.model}, {args.dataset}, {args.noise_std}")
    if len(results) == 0:
        return 0, n
    avg = sum(results) / len(results)
    args.seed = initial_seed
    return avg, std(results)

def vary_params(args):
    """
    Run configurations with array of parameters, save results to file.
    """
    results = {}
    baseline = None
    print(args.model)
    for T in Ts:
        temp = {}
        args.dataset = f"hrg_n1000_t{T}"
        for alpha in alphas:
            args.noise_std = alpha
            print(f"  Alpha = {alpha}, T = {T}")
            avg, _ = multiple_runs(args, n=num_runs[args.model])
            
            temp[alpha] = avg

            if T == 0 and alpha == 0:
                baseline = avg
                rel = "-"
            else:
                rel = avg / baseline - 1
            
            temp[f"{alpha} rel"] = rel
        results[T] = temp
    results = pd.DataFrame.from_dict(results)
    return results


def df_to_latex(df, model):

    # delete labels of rel rows
    row_mapping = {}
    for i in df.iloc:
        if type(i.name) == str and "rel" in i.name:
            row_mapping[i.name] = ""
        else:
            row_mapping[i.name] = i.name
    df.rename(index=row_mapping, inplace=True)

    df = df.round(3)

    latex = df.to_latex()

    with open(f"results\\experiments_{args.model}.txt", "w") as text_file:
        text_file.write(latex)

if __name__=="__main__":
    args = parser.parse_args()

    args.log_info = False
    args.save = 0

    args.model = "HGCN"
    args.dataset = "hrg_n1000"
    args.lr = 0.1

    # args.weight_decay = 0.0001

    args.dim = 16
    args.hidden_dim = 16
    args.num_layers = 2
    
    # multiple_runs(args, n=5)

    dimensionality_experiments(args)


    # for noise to late

    # args.model = "GCN"
    # results = vary_params(args)
    # latex = df_to_latex(results, args.model)

    # args.model = "HGCN"
    # results = vary_params(args)
    # latex = df_to_latex(results, args.model)


    # plot_c(True)

    import winsound
    duration = 1000  # milliseconds
    freq = 1200  # Hz
    winsound.Beep(freq, duration)