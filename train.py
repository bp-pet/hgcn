from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
import random
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics

import matplotlib.pyplot as plt


def train(args):
    
    with open("curvature\\curvature.txt", 'w') as f:
        f.write("")

    np.random.seed(args.seed) # is overwritten by split_seed, so doesn't seem to do anything
    torch.manual_seed(args.seed) # for SGD
    random.seed(args.seed) # for deleting edges, not used currently
    # args.split_seed = args.seed
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.getcwd() + "\logs", args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Load data
    data = load_data(args, os.path.join(os.getcwd() + "\data", args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        # else:
        #     Model = RECModel
        #     # No validation for reconstruction task
        #     args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    logging.info(str(model))
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            # print(embeddings)
            val_metrics = model.compute_metrics(embeddings, data, 'val')

            ################################
            plt.scatter(epoch, 0.5 * (val_metrics["roc"].item() + val_metrics["ap"].item()), marker='x', c='r')
            ################################

            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break
    
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.clf()

    logging.info("Optimization Finished!")
    elapsed_time = time.time() - t_total
    logging.info("Total time elapsed: {:.4f}s".format(elapsed_time))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")

    roc = best_test_metrics["roc"]
    with open("times.txt", 'a') as f:
        f.write(f"\nEpoch {epoch}, time {elapsed_time}, roc {roc}")

    return best_test_metrics
    
def multiple_runs(args, n=20):
    """
    Run same configuration multiple times, save test ROC/accuracy. Return average.

    Generate a different random seed for each run.
    """
    seed = np.random.randint(0, 100)

    if args.task == "lp":
        metric = "roc"
    else:
        metric = "accuracy" # might not be correct

    args.log_freq=1000
    args.save = 0

    results = []
    error_counter = 0
    for i in range(n):
        print(f"Round {i + 1} of {n}")
        args.seed = (i + 1) * seed
        while True:
            try:
                results.append(train(args)[metric])
            except Exception as e:
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
    save_dir = "results.txt"


    for model in ["HGCN"]: # change if GCN also wanted
        args.model = model
        if model == "GCN":
            args.manifold = "Euclidean"
            num_runs = 20
        else:
            args.manifold = "Hyperboloid"
            num_runs = 5
        with open(save_dir, 'a') as f:
            f.write(f"{model}, {num_runs} runs for each result\n")
        for temp in [0.0, 0.2, 0.4, 0.6, 0.8]:
            with open(save_dir, 'a') as f:
                f.write(f"\tTemperature {temp}\n")
            args.temperature = temp
            for noise in [-1, 0.0, 0.2, 0.4, 0.6, 0.8]:
                args.noise_std = noise
                avg, errors = multiple_runs(args, n=num_runs)
                with open(save_dir, 'a') as f:
                    f.write(f"\t\tNoise {noise}, result {avg}, error runs {errors}\n")

if __name__ == '__main__':

    args = parser.parse_args()

    args.model = "HGCN"

    args.dataset = "hrg"

    if args.model == "GCN":
        args.manifold = "Euclidean"
    elif args.model == "HGCN":
        args.manifold = "Hyperboloid"
    
    # args.lr_reduce_freq = 200
    # args.gamma = 0.5
    # args.lr = 0.1
    # args.epochs = 4000
    # args.patience = 300
    # args.weight_decay = 0.001

    args.use_feats = 1

    args.dim = 16

    # args.num_layers = 2

    # torch.Size([100, 16])

    for rseed in range(1):
        args.seed = rseed
        train(args)
        # try:
        #     train(args)
        # except Exception as e:
        #     print(e)
        #     continue


    # multiple_runs(args)
    # vary_params(args)
 