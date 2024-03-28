import argparse
import torch, dgl
import model, utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed', 'amazon-computers', 'amazon-photo').",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    # load and preprocess dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = ( dgl.AddSelfLoop() )  # by default, it will first remove self-loops to prevent duplication
    data = utils.data_load(transform, args.dataset)
    graph = data[0]
    graph = graph.int().to(device)
    
    # For Amazon datasets, we need to split the dataset into train/val/test sets
    features = graph.ndata["feat"]
    labels = graph.ndata["label"]
    masks = utils.train_val_test(graph, args.dataset)

    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes
    gcn = model.GCN(in_size, 16, out_size).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        graph = dgl.to_bfloat16(graph)
        features = features.to(dtype=torch.bfloat16)
        gcn = gcn.to(dtype=torch.bfloat16)

    # model training
    print("Training...")
    model.train(graph, features, labels, masks, gcn)

    # test the model
    print("Testing...")
    acc = model.evaluate(graph, features, labels, masks[2], gcn)
    print("Test accuracy {:.4f}".format(acc))
    