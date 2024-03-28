import torch

from dgl.data import (
    CiteseerGraphDataset, 
    CoraGraphDataset, 
    PubmedGraphDataset, 
    AmazonCoBuyComputerDataset,
    AmazonCoBuyPhotoDataset
)


def data_load(transform, keyword):
    if keyword == "cora":
        return CoraGraphDataset(transform=transform)
    
    elif keyword == "citeseer":
        return CiteseerGraphDataset(transform=transform)
    
    elif keyword == "pubmed":
        return PubmedGraphDataset(transform=transform)
    
    elif keyword == "amazon-computers":
        return AmazonCoBuyComputerDataset(transform=transform)
    
    elif keyword == "amazon-photo":
        return AmazonCoBuyPhotoDataset(transform=transform)
    
    else:
        raise ValueError("Unknown dataset: {}".format(keyword))


def train_val_test(graph, keyword):
    if keyword in ["amazon-computers", "amazon-photo"]:
        num_nodes = graph.num_nodes()
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_idx = torch.randperm(num_nodes)[:int(num_nodes * 0.6)]
        val_idx = torch.randperm(num_nodes)[int(num_nodes * 0.6):int(num_nodes * 0.8)]
        test_idx = torch.randperm(num_nodes)[int(num_nodes * 0.8):]
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        return train_mask, val_mask, test_mask
    
    else:
        return graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"]
