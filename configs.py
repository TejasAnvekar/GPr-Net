import argparse
import os 
root = os.path.abspath(os.path.dirname(__file__))

def CONFIGS():
    args = argparse.ArgumentParser("GPr-Net")

    args.add_argument("--strain",default=None,type=list)
    args.add_argument("--stest",default=None,type=list)
    args.add_argument("--num_points",default=512,type=int)
    args.add_argument("--c",default=1.0,type=float)
    args.add_argument("--manifold",default="Poincare",type=str)
    args.add_argument("--emb_dim",default=32,type=int)
    args.add_argument("--K",default=40,type=int)
    args.add_argument("--LV",action='store_true')

    args.add_argument("--kways",default=5,type=int)
    args.add_argument("--TrS",default=10,type=int)
    args.add_argument("--TS",default=10,type=int)
    args.add_argument("--TrQ",default=20,type=int)
    args.add_argument("--TQ",default=20,type=int)
    args.add_argument("--TrE",default=4,type=int)
    args.add_argument("--TE",default=300,type=int)

    args.add_argument("--seed",default=10,type=int)
    args.add_argument("--epochs",default=50,type=int)
    args.add_argument("--lr",default=0.1,type=float)
    args.add_argument("--weight_decay",default=0.0001,type=float)
    args.add_argument("--momentum",default=0.9,type=float)
    args.add_argument("--gpu",action='store_true')

    args.add_argument("--logname",default="Poincare_GPr-Net.log",type=str)
    args.add_argument("--logpath",default=f"{root}/logs",type=str)

    return args.parse_args()