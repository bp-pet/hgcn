import numpy as np

from train import train
from config import parser


save_dir = "results\\experiments.txt"

num_runs = {"GCN": 20, "HGCN": 5}

temps = [0.0, 0.2, 0.4, 0.6, 0.8]
noises = [0.0, 0.2, 0.4, 0.6, 0.8, -1]


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
    error_counter = 0
    for i in range(n):
        print(f"Run {i + 1} of {n}")
        args.seed = (i + 1) * initial_seed
        while True:
            try:
                results.append(train(args)[metric])
            except Exception as e:
                print(e)
                error_counter += 1
            finally:
                break
    if len(results) == 0:
        return 0, n
    avg = sum(results) / len(results)
    return avg, error_counter

def vary_params(args):
    """
    Run configurations with array of parameters, save results to file.
    """
    for model in ["GCN", "HGCN"]: # change if GCN also wanted
        with open(save_dir, 'a') as f:
            f.write(f"{model}, {num_runs} runs for each result\n")
        for temp in temps:
            with open(save_dir, 'a') as f:
                f.write(f"\tTemperature {temp}\n")
            args.dataset = f"hrg_n1000_t{temp}"
            for noise in noises:
                args.noise_std = noise
                avg, errors = multiple_runs(args, n=num_runs[model])
                with open(save_dir, 'a') as f:
                    f.write(f"\t\tNoise {noise}, result {avg}, error runs {errors}\n")


if __name__=="__main__":
    args = parser.parse_args()

    args.log_freq=500
    args.save = 0
