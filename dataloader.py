import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob
import h5py
import random


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        # print(f"h5_name: {h5_name}")
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(
        pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40_FSL(Dataset):

    def __init__(self, num_points, n_episodes=10, k_ways=3, m_shots=5, n_querys=15):
        super().__init__()

        self.data, self.label = load_data('train')
        # self.test_data, self.test_label = load_data('test')

        self.num_points = num_points

        self.n_episodes = n_episodes
        self.k_ways = k_ways

        self.m_shots = m_shots
        self.n_querys = n_querys

        self.label_idx = {}
        for key in range(np.max(self.label)+1):
            self.label_idx[key] = []
            for i, label in enumerate(self.label):
                if label == key:
                    self.label_idx[key].append(i)

        self.make_all_episodes()

    def augumentation(self, pc):
        pc = jitter_pointcloud(pc)
        pc = translate_pointcloud(pc)
        pc = random_point_dropout(pc)
        return pc

    def make_all_episodes(self):

        self.DATA_SUPPORT = []
        self.DATA_QUERY = []
        self.LABEL_SUPPORT = []
        self.LABEL_QUERY = []

        k_way = random.sample(range(np.max(self.label)+1), self.k_ways)
        for _ in range(self.n_episodes):

            data_support = []
            label_support = []
            data_query = []
            label_query = []

            for i, class_id in enumerate(k_way):

                support_id = random.sample(
                    self.label_idx[class_id], self.m_shots)
                query_id = random.sample(
                    list(set(self.label_idx[class_id]) - set(support_id)), self.n_querys)

                pc_support_id = self.data[support_id][:, :self.num_points, :]

                pc_query_id = self.data[query_id][:, :self.num_points, :]

                for j in range(pc_support_id.shape[0]):
                    # pc_support_id[j] = self.augumentation(pc_support_id[j])
                    np.random.shuffle(pc_support_id[j])

                for j in range(pc_query_id.shape[0]):
                    np.random.shuffle(pc_query_id[j])

                data_support.append(pc_support_id)
                label_support.append(i*np.ones(self.m_shots))

                data_query.append(pc_query_id)
                label_query.append(i*np.ones(self.n_querys))

            self.DATA_SUPPORT.append(np.concatenate(data_support))
            self.LABEL_SUPPORT.append(np.concatenate(label_support))
            self.DATA_QUERY.append(np.concatenate(data_query))
            self.LABEL_QUERY.append(np.concatenate(label_query))

        del data_support
        del data_query
        del label_support
        del label_query
        del self.label_idx
        del self.data
        del self.label

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, index):

        return self.DATA_SUPPORT[index], self.LABEL_SUPPORT[index], self.DATA_QUERY[index], self.LABEL_QUERY[index]


if __name__ == '__main__':

    # from render_pc import render_pc

    n_querys = 20
    m_shots = 10

    k_ways = 5
    n_cls = 40
    n_episodes = 2
    num_points = 1024

    dataset = ModelNet40_FSL(num_points=num_points, n_episodes=n_episodes,
                             k_ways=k_ways, m_shots=m_shots, n_querys=n_querys)
    loader = DataLoader(dataset=dataset, batch_size=1,
                        num_workers=4, shuffle=True, drop_last=False)

    for i, (data) in enumerate(loader):

        tr_s_pc, tr_s_l, tr_q_pc, tr_q_l = data

        print(f"{i} support stats---> data:{tr_s_pc.shape}, label:{tr_s_l.shape}")
        print(f"{i} query stats--->  data:{tr_q_pc.shape}, label:{tr_q_l.shape}")

        exit()
