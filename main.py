import torch
import torch.nn as nn

from model.ProtoNet import ProtoNet
from networks.simple import Net
from dataloader import load_data
from utils import train, test, make_data_for_FSL, set_global_seed, create_log_folders


import logging

def main(args):

    root,checkpoint_path, CSV_path = create_log_folders(args)

    set_global_seed(args.seed)

    train_x, train_y = load_data('train')
    test_x, test_y = load_data('test')

    train_y = train_y[:, 0]
    test_y = test_y[:, 0]



    train_x, train_y, test_x, test_y = make_data_for_FSL(
        train_x, train_y, test_x, test_y, total_cls=40, train_category=30, num_points=args.num_points,
        train_classes = args.strain, test_classes = args.stest)


    logging.basicConfig(filename=root+"/"+args.logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt="%H:%M:%S",
                        level=logging.DEBUG)

    logging.info("Running GPr-Net")

    logger = logging.getLogger(f"{args.manifold} GPr-Net")

    max_epoch = args.epochs

    encoder = Net(args.LV, args.emb_dim, args.K)
    device  = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"
    model = ProtoNet(encoder=encoder, c=args.c, manifold=args.manifold).to(device)


    if args.manifold.lower() =="euclidean":
        print("Using SGD")
        optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr,
                                weight_decay=args.weight_decay, 
                                momentum=args.momentum)
    else:
        from geoopt.optim import RiemannianSGD
        print("Using RSGD")
        optimizer = RiemannianSGD(model.parameters(), 
                                lr=args.lr,
                                weight_decay=args.weight_decay, 
                                momentum=args.momentum)


    train(args, checkpoint_path, CSV_path, logger, model, optimizer, train_x, train_y, test_x,
          test_y, max_epoch)
    test(args, checkpoint_path,CSV_path, logger, model, test_x, test_y)


if __name__ == "__main__":

    from configs import CONFIGS

    args = CONFIGS()
    main(args)
