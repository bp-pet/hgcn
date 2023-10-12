import matplotlib.pyplot as plt
import pandas as pd
import json

fontsize = 12

def main(stats=True, loss=True, curvature=True):
    curvature = pd.read_csv("curvature\\curvature.csv", index_col=False)
    num_layers = len(curvature.columns) - 1
    
    # config string
    with open('curvature\\configs.json') as user_file:
        parsed_json = json.load(user_file)
    model_str = parsed_json["model"]
    dataset_str = parsed_json["dataset"]
    task_str = parsed_json["task"]
    output_dim_str = parsed_json["dim"]
    hidden_dim_str = parsed_json["hidden_dim"]
    num_layers_str = parsed_json["num_layers"]
    c_str = parsed_json["c"]
    lr_str = parsed_json["lr"]
    lr_reduce_freq_str = parsed_json["lr_reduce_freq"]
    lr_reduce_gamma_str = parsed_json["gamma"]
    min_epochs_str = parsed_json["min_epochs"]
    max_epochs_str = parsed_json["epochs"]
    patience_str = parsed_json["patience"]
    dropout_str = parsed_json["dropout"]
    weight_decay_str = parsed_json["weight_decay"]
    normalize_feats_str = parsed_json["normalize_feats"]
    random_seed_str = parsed_json["seed"]
    result_str = parsed_json["result"]
    configs_string = f"{model_str}, {dataset_str}, {task_str}\n" \
                     f"output dim: {output_dim_str}, hidden dim: {hidden_dim_str}, # of layers: {num_layers_str}\n" \
                     f"curvature: {c_str}\n" \
                     f"\n" \
                     f"lr: {lr_str}, lr reduce freq: {lr_reduce_freq_str}, lr reduce gamma {lr_reduce_gamma_str}\n" \
                     f"min epochs: {min_epochs_str}, max_epochs: {max_epochs_str}, patience: {patience_str}\n" \
                     f"dropout: {dropout_str}, weight decay: {weight_decay_str}, normalize feats: {normalize_feats_str}\n" \
                     f"random seed: {random_seed_str}\n" \
                     f"\n" \
                     f"result: {round(result_str, 3)}\n" \
                     f"\n"

    # add final learned curvatures to string
    configs_string += "Final c for each layer: "
    for l in range(num_layers):
        configs_string += str(round(curvature.iloc[-1][l], 2))
        configs_string += "; "
    
    # plot curvatures
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    for i in range(num_layers):
        curvature.rename(columns={str(i): f"Layer {i}"}, inplace=True)
    for i in range(num_layers):
        curvature.plot(ax=axes[1], y=f"Layer {i}")

    # plot configs
    axes[0].axis("off")
    axes[0].text(-0.5, 0.5, configs_string, fontsize=fontsize)

    # plot average
    curvature["Average"] = curvature.drop(columns=["loss"]).mean(axis=1)
    curvature.plot(ax=axes[1], y="Average", linestyle='dashed', color='gray')

    # plot loss
    curvature.plot(ax=axes[2], y="loss")

    # save
    axes[1].title.set_text("Layer curvature")
    axes[2].title.set_text("Loss")
    axes[2].legend(["loss"])

    plt.savefig("curvature\\curvature.png")

if __name__=="__main__":
    main()