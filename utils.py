import numpy as np
import torch
from tqdm import tqdm
from torch import optim
import random
import os
import time
import pandas as pd


def create_log_folders(args):

    folder = os.path.join(os.path.join(args.logpath,args.manifold),f"seed_{args.seed}")

    name = f"num_points_{args.num_points}_K_{args.K}_LV_{args.LV}_c_{args.c}_emb_dim_{args.emb_dim}_Epochs_{args.epochs}_ways_{args.kways}_TrainSupport_{args.TrS}_TestSupport_{args.TS}_TrainQuery_{args.TrQ}_TestQuery_{args.TQ}_TrainEpisodes_{args.TrE}_TestEpisodes_{args.TE}"
    folder = os.path.join(folder,name)

    if not os.path.exists(folder):
        os.makedirs(folder)

    checkpoint = os.path.join(folder,"checkpoints")
    CSV = os.path.join(folder,"CSV")

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    if not os.path.exists(CSV):
        os.makedirs(CSV)

    return folder,checkpoint,CSV


    
    






def log_and_print(logger,str):
    logger.debug(str)
    print(str)



def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v*self.n+x)/(self.n+1)
        self.n += 1

    def item(self):
        return self.v


class Timer:
    def __init__(self):
        self.o = time.time

    def measure(self, p=1):
        x = (time.time()-self.o)/p
        x = int(x)
        if x >= 3600:
            return "{:.1f}h".format(x/3600)
        if x >= 60:
            return "{}m".format(x/60)
        return "{}s".format(x)


def compute_confidence_interval(data):
    a = 1.0*np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96*(std/np.sqrt(len(a)))
    return m, pm


def make_data_for_FSL(train_x, train_y, test_x, test_y, total_cls=40, train_category=24, num_points=512, train_classes=None, test_classes=None):

    data_x = np.concatenate([train_x, test_x], axis=0)[:, :num_points, :]
    data_y = np.concatenate([train_y, test_y], axis=0)

    num_cls = list(range(0, total_cls))
    # num_cls = [1,2,8,12,14,22,23,30,33,35]

    if train_classes == None:
        train_classes = random.sample(num_cls, train_category)
    if test_classes == None:
        test_classes = [cls for cls in num_cls if cls not in train_classes]

    print(f" Train split class ids: {train_classes} \n \
             Test split class ids: {test_classes}")

    print(
        f"D_train intersection D_test = {set(train_classes).intersection(set(test_classes))}")

    ridx = np.random.randint(
        low=0, high=data_x.shape[0], size=(data_x.shape[0]))

    data_x = data_x[ridx]
    data_y = data_y[ridx]

    train_split_idx = []
    test_split_idx = []

    for i in range(len(data_y)):
        if data_y[i] in train_classes:
            train_split_idx.append(i)
        else:
            test_split_idx.append(i)

    train_split_idx = np.array(train_split_idx)
    test_split_idx = np.array(test_split_idx)

    return data_x[train_split_idx], data_y[train_split_idx], data_x[test_split_idx], data_y[test_split_idx]




def get_modelnet10(train_x, train_y, test_x, test_y):
    class_labels = [1,2,8,12,14,22,23,30,33,35]
    class_labels = np.array(class_labels)


    tr_idx = np.where(train_y==class_labels)[0]
    t_idx = np.where(test_y==class_labels)[0]
    return train_x[tr_idx], train_y[tr_idx], test_x[t_idx], test_y[t_idx]




def extract_sample(n_way, n_support, n_query, datax, datay):
    sample = []
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    for cls in K:
        datax_cls = datax[datay == cls]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support+n_query)]
        sample.append(sample_cls)
    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()
    return ({
        "pointcloud": sample,
        "n_way": n_way,
        "n_support": n_support,
        "n_query": n_query
    })


def save_ckpt(model, optimizer, scheduler, checkpoint_path):
    data = {
        'weights': model.state_dict(),
        'optim': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    torch.save(data, checkpoint_path+"\\data.pt")





def train(args, checkpoint_path, CSV_path, logger, model, optimizer, train_x, train_y, test_x, test_y, max_epoch):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epoch, eta_min=0.001, last_epoch=-1, verbose=False)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop

    best_acc = 0.0

    Train_CSV_data = {"Train_loss":[],"Train_acc":[]}
    Val_CSV_data = {"Val_loss":[],"Val_acc":[]}

    while epoch < max_epoch and not stop:
        model.train()
        loop = range(args.TrE)

        tl = Averager()
        ta = Averager()

        for episode in tqdm(loop, total=len(loop), colour='green', leave=False):
            sample = extract_sample(args.kways, args.TrS, args.TrQ, train_x, train_y)
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3, norm_type=2)
            optimizer.step()
            tl.add(output['loss'])
            ta.add(output['acc'])

     
        log_str = 'Train results -- Loss: {:.4f} Acc: {:.4f}'.format(tl.item(), ta.item())
        log_and_print(logger,log_str)

        Train_CSV_data['Train_loss'].append(tl.item())
        Train_CSV_data['Train_acc'].append(ta.item())
        df = pd.DataFrame(Train_CSV_data)

        df.to_csv(CSV_path+"\\train.csv")

        
        epoch += 1
        scheduler.step()

        vl = Averager()
        va = Averager()

        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                loop = range(args.TE)
                for _ in tqdm(loop, total=len(loop), colour='blue', leave=False):
                    sample = extract_sample(args.kways, args.TS, args.TQ, test_x, test_y)
                    _, output = model.set_forward_loss(sample)

                    vl.add(output['loss'])
                    va.add(output['acc'])

                if va.item() > best_acc:
                    best_acc = va.item()
                    log_str = '\n Best Val results -- Loss: {:.4f} Acc: {:.4f}'.format(vl.item(), va.item())
                    log_and_print(logger,log_str)
                    save_ckpt(model, optimizer, scheduler, checkpoint_path)

            
                log_str = 'Val results -- Loss: {:.4f} Acc: {:.4f}'.format(vl.item(), va.item())
                log_and_print(logger,log_str)

                Val_CSV_data['Val_loss'].append(vl.item())
                Val_CSV_data['Val_acc'].append(va.item())

                df = pd.DataFrame(Val_CSV_data)
                df.to_csv(CSV_path+"\\val.csv")


def test(args, checkpoint_path,CSV_path, logger, model, test_x, test_y):

    ave_acc = Averager()
    test_acc_record = np.zeros(args.TE,)

    print("loading Model")
    model.load_state_dict(torch.load(checkpoint_path+"\\data.pt")['weights'])
        
    with torch.no_grad():
        model.eval()
        loop = range(args.TE)
        for i in loop:
            sample = extract_sample(args.kways, args.TS, args.TQ, test_x, test_y)
            _, output = model.set_forward_loss(sample)

            ave_acc.add(output['acc'])
            test_acc_record[i-1] = output['acc']
            print("batch {}: {:.2f}({:.2f})".format(
                i, ave_acc.item()*100, output['acc']*100))

        m, pm = compute_confidence_interval(test_acc_record)
        log_str = "\n \n Test Acc {:.4f} + {:.4f}".format(m, pm)
        log_and_print(logger,log_str)

        df = pd.DataFrame({"ACC":[m]})
        df.to_csv(CSV_path+"\\test.csv")



