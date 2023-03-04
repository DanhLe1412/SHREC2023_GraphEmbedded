import torch
import torch_geometric.data as tgd
import torch_geometric.transforms as tgt
from torch.nn import functional as F
import random
import argparse
import configparser
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path
import numpy as np
import timeit

from dataset.animar import AnimalDataset, InMemoryAnimalDataset
from models.models import GNN
from models.pointnet import PointNet
from models.edge_conv import SimpleEdgeConvModel, EdgeConvModel
from utils.transformation import SamplePoints
from utils.model import remove_final_layer


@torch.no_grad()
def test(model, loader, args):
    model.eval()
    correct = 0
    loss = 0.
    criterion = torch.nn.CrossEntropyLoss()
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.argmax(dim=1)
        batch_loss = criterion(out, data.y)
        correct += int((pred == data.y).sum())
        loss += batch_loss
    return correct / len(loader.dataset)

def train(model, loader, optimizer, criterion, device):
    total_loss = []
    for i, data in enumerate(loader):
        optimzer.zero_grad()
        data = data.to(device)
        out = model(data)
        target = data.y.long()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.detach().item())
    return sum(total_loss) / len(total_loss)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=167,
                        help='seed')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--nhid', type=int, default=256,
                        help='hidden size')
    parser.add_argument('--pooling-ratio', type=float, default=0.5,
                        help='pooling ratio')
    parser.add_argument('--dropout-ratio', type=float, default=0.5,
                        help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=100000,
                        help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=500,
                        help='patience for earlystopping')

    parser.add_argument('--num-examples', type=int, default=-1,
                        help='number of examples, all examples by default')
    parser.add_argument('--meshes-to-points', type=int, default=0,
                        help='convert the initial meshes to points cloud')
    parser.add_argument('--face-to-edge', type=int, default=1,
                        help='convert the faces to edge index')
    parser.add_argument('--model', default="gnn",
                        help='main model')
    parser.add_argument('--layer', default="gnn",
                        help='layer to use if you are using simple_edge_conv or edge_conv')

    parser.add_argument('--set-x', default=1, type=int,
                        help='set x features during data processing')
    parser.add_argument("--num-instances", type=int, default=-1,
                        help="Number of instances per class")
    parser.add_argument("--num-sample-points", type=int, default=-1,
                        help="Number of points to sample when convert from meshes to points cloud")
    parser.add_argument("--load-latest", action="store_true",
                        help="Load the latest checkpoint")
    parser.add_argument("--num-classes", type=int, default=144,
                        help="Number of classes")
    parser.add_argument("--random-rotate", action="store_true",
                        help="Use random rotate for data augmentation")
    parser.add_argument("--k", type=int, default=16,
                        help="Number of nearest neighbors for constructing knn graph")
    parser.add_argument("--in-memory-dataset", action="store_true",
                        help="Load the whole dataset into memory (faster but use more memory)")
    parser.add_argument('--process-only', action="store_true",
                        help='whether to only process the data and then stop')
    parser.add_argument('--save-path', default="./",
                        help='root path for storing processed dataset and models')
    parser.add_argument('--mode', default="train-test", choices=["train-test", "submit", "submit2"],
                        help='root path for storing processed dataset and models')

    args = parser.parse_args()
    random.seed(args.seed)
    config = configparser.ConfigParser()
    config.read("config.ini")
    config_paths = config["PATHS"]
    base_path = config_paths["base_path"]

    ## Dataset
    if args.in_memory_dataset:
        DatasetType = InMemoryAnimalDataset
    else:
        DatasetType = AnimalDataset
    
    train_dataset = DatasetType()

    ## Transforms
    list_transforms = []
    if args.face_to_edge == 1:
        list_transforms.append(tgt.FaceToEdge(True))
    if args.meshes_to_points == 1:
        list_transforms.append(SamplePoints(num=args.num_sample_points))
    transforms = tgt.Compose(list_transforms)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == "pointnet":
        model = PointNet(args).to(args.device)
    elif args.model == "simple_edge_conv":
        model = SimpleEdgeConvModel(args).to(args.device)
    elif args.model == "edge_conv":
        model = EdgeConvModel(args).to(args.device)
    else:
        model = GNN(args).to(args.device)
    

    model_subfolder = f"{args.model}-{configuration}-{args.num_sample_points}-{args.nhid}-{args.num_instances}" 
    model_save_path = f'{args.save_path}saved_models/{model_subfolder}-latest.pth'
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    min_loss = 1e10
    patience = 0
    epoch = 0

    if args.mode == "train":
        total_time = 0
        for epochs in range(args.epochs):
            start = timeit.default_timer()
            model.train()
            training_loss = 0
            training_acc = 0
            loss = train(model, loader, optimzer, criterion, args.device)
            train_acc, train_iou, train_dice = test(model, train_loader, device)
            val_acc, val_iou, val_dice = test(model, test_loader, device)
            if loss < min_loss:
                torch.save(model.state_dict(), model_save_path)
                print("Model saved at epoch {}".format(epoch))
                min_loss = loss
                patience = 0
            else:
                patience += 1
        stop = timeit.default_timer()
        epochs_time = stop-start
        print(f"Current epoch time: {epochs_time}")
        total_time += epochs_time
        if patience > args.patience:
            break

        if epoch:
            print("Last epoch before stopping:", epoch)
        
        test_acc = test(model, test_loader, args)
        print("Test acc: {}".format(test_acc))


           
if __name__ == "__main__":
    main()
