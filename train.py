import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics

from curvature.plot_c import main as plot_c


def train(args):

    # set manifold ##########################
    if args.model == "GCN":
        args.manifold = "Euclidean"
    elif args.model == "HGCN":
        args.manifold = "PoincareBall"
    #########################################

    # preprocess c input ####################
    # if type(args.c) is list:
    #     if len(args.c) != args.num_layers + 1:
    #         raise Exception("Invalid number of curvatures given, must be num_layers + 1")
    # else:
    #     args.c = [args.c] * (args.num_layers + 1)



    # initialize loss and curvature log ##############
    d = {}
    for i in range(args.num_layers):
        d[i] = []
    curvature_log = pd.DataFrame.from_dict(d)
    if args.task == "nc": # if task is nc then last layer is linear non-hyp.
        curvature_log = curvature_log.drop(columns=[args.num_layers - 1])
    #########################################

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    if args.log_info:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join("logs", args.task, date)
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
    data = load_data(args, os.path.join("data", args.dataset))
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
        
        # logging.info(str(model))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
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
                if counter >= args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break
    
        # record curvature and loss ############
        d = {}
        for i in model.children():
            for j in i.children():
                for i, layer in enumerate(j.children()):
                    if isinstance(layer, torch.nn.Linear):
                        continue
                    try:
                        d[i] = [layer.get_in_curvature()]
                    except AttributeError:
                        d[i] = [0.0]
        
        try:
            d[args.num_layers] = layer.get_out_curvature()
        except AttributeError:
            d[args.num_layers] = [0.0]
        d["loss"] = [val_metrics["loss"].item()]
        curvature_log = pd.concat([curvature_log, pd.DataFrame.from_dict(d)])
        #########################################
    

    # save loss and curvature ###########
    curvature_log.to_csv("curvature\\curvature.csv", index=False, header=True)
    configs_json = vars(args)
    configs_json["result"] = best_val_metrics["roc"]
    json.dump(vars(args), open("curvature\\configs.json", 'w'))
    ##################################################


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

        # save heatmap of embeddings ####################
        # plt.imshow(best_emb.detach().numpy(), cmap='hot', interpolation='nearest')
        # plt.savefig("curvature\\embedding_heatmap.png")
        #################################################

        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")
        with open("curvature\\last_experiment_id.txt", 'w') as f:
            f.write(save_dir)
    
    # record time, epochs, accuracy #######################
    if "roc" in best_test_metrics:
        roc = best_test_metrics["roc"]
        # roc = best_val_metrics["roc"]
        with open("curvature\\times.txt", 'a') as f:
            # f.write(f"\nEpoch {epoch}, time {elapsed_time}, roc {roc}")
            f.write(f"\n{epoch} epochs, ROC AUC {round(roc, 3)}")
    else:
        f1 = best_test_metrics["f1"]
        with open("curvature\\times.txt", 'a') as f:
            f.write(f"\nEpoch {epoch}, time {elapsed_time}, f1 {f1}")
    #######################################################

    return best_test_metrics


if __name__ == '__main__':
    args = parser.parse_args()

    args.dataset = "enschede_road"
    args.model = "HGCN"

    args.dim = 2
    args.hidden_dim = 16
    args.num_layers = 4

    args.lr = 0.1
    # args.lr_reduce_freq = 200
    # args.gamma = 0.5

    # args.use_att = True
    args.local_agg = True

    # for cora lp
    # args.dropout = 0.5
    # args.weight_decay = 0.0001

    # for pubmed lp
    # args.dropout = 0.4
    # args.weight_decay = 0.0001

    # for disease_lp
    # args.normalize_feats = False

    # args.seed = 2

    # args.min_epochs = 1000
    # args.epochs = 1
    # args.patience = 300

    args.save = True
    args.log_freq = 10

    args.seed = 1234

    train(args)

    plot_c(stats=True)

    import winsound
    duration = 1000  # milliseconds
    freq = 1200  # Hz
    winsound.Beep(freq, duration)