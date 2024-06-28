import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
from openpyxl import load_workbook
import pickle
import pandas as pd


class PIPALFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num, sel_types=[2]):
        # dist_dict = {'trad': range(12), 'trad_SR': range(16), 'PSNR_SR': range(10), 'SR_mismatch': range(24),
        #              'GAN_SR': range(13), 'Denoising': range(14), 'SR_Denoising': range(27)}
        # dist_sub_type = {0: 'trad', 1: 'trad_SR', 2: 'PSNR_SR', 3: 'SR_mismatch', 4: 'GAN_SR', 5: 'Denoising',
        #                  6: 'SR_Denoising'}
        info_root = os.path.join(root, 'train', 'Train_Label')
        info_txt = [os.path.join(info_root, file) for file in sorted(os.listdir(info_root))]

        names = []
        scores = []
        for i in index:
            with open(info_txt[i], 'r') as f:
                content = f.readlines()
            for line in content:
                name, score = line.strip().split(',')
                _, dis_type, _ = name.split('_')
                if int(dis_type) in sel_types:
                    names.append(name)
                    scores.append(score)

        mos = np.array(scores).astype(np.float32)
        labels = normalize_labels(mos)

        sample = []
        for i, name in enumerate(names):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'train', 'Distortion', name), labels[i]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return path, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class KADIS_700kFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        data = pd.read_csv(os.path.join(root, 'kadis700k_ref_imgs.csv'))
        ref_im = data['ref_im'].tolist()
        dist_type = data['dist_type_1'].tolist()

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'ref_imgs', ref_im[item]), 0))
            for j in range(1, 6):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'dist_imgs',
                                                '%s_%02d_%02d.bmp' % (ref_im[item].split('.')[0], dist_type[item], j)),
                                   dist_type[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return path, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class SPAQFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        info = pd.read_excel(os.path.join(root, 'Annotations', 'MOS and Image attribute scores.xlsx'))
        imgname = info['Image name'].tolist()  # old version: tolist(); new version: to_list()
        mos = info['MOS'].values.astype(np.float32)

        labels = normalize_labels(mos)
        sample = []

        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'TestImage', imgname[item]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return path, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class LIVEChallengeFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169]
        info = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        mos = info['AllMOS_release'].astype(np.float32)
        mos = mos[0][7:1169]
        labels = normalize_labels(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'Images', imgpath[item][0][0]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return path, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class Koniq_10kFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        data = pd.read_csv(os.path.join(root, 'koniq10k_scores_and_distributions.csv'))
        imgname = data['image_name'].tolist()
        mos = data['MOS'].values.astype(np.float32)  # to_numpy()

        labels = normalize_labels(np.array(mos))
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, '512x384', imgname[item]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return path, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class BIDFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        info = pd.read_excel(os.path.join(root, 'DatabaseGrades.xlsx'))
        img_num = info['Image Number'].tolist()  # old version: tolist(); new version: to_list()
        imgname = ["DatabaseImage%04d.JPG" % (i) for i in img_num]

        mos = info['Average Subjective Grade'].values.astype(
            np.float32)  # old version: value(); new version: to_numpy()

        labels = normalize_labels(mos)
        sample = []

        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, imgname[item]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return path, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class KADID_10kFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num, sel_all=True):
        refname = ['I%02d.png' % i for i in range(1, 82)]

        data = pd.read_csv(os.path.join(root, 'dmos.csv'))
        if sel_all:
            imgnames = data['dist_img'].tolist()
            refnames_all = data['ref_img'].values
            mos = data['dmos'].values.astype(np.float32)
        else:
            dist_lvs = np.array(range(1, 6)).reshape([1, -1])
            sel_types = np.array([1, 3, 9, 18, 20, 25]).reshape([-1, 1])
            sel_imgs = np.array(range(0, 81)).reshape([-1, 1])
            sel_dists = dist_lvs + (sel_types - 1) * 5 - 1
            sel_idx = sel_dists.reshape([1, -1]) + sel_imgs * 125
            sel_idx = sel_idx.flatten().tolist()

            imgnames = data.loc[sel_idx, 'dist_img'].tolist()
            refnames_all = data.loc[sel_idx, 'ref_img'].values
            mos = data.loc[sel_idx, 'dmos'].values.astype(np.float32)

        labels = normalize_labels(mos)

        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append(
                        (os.path.join(root, 'images', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return path, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def normalize_labels(ys, flip=False):
    assert type(ys) == np.ndarray
    y_max = np.max(ys)
    y_min = np.min(ys)
    ys_norm = (ys - y_min) / (y_max - y_min)
    if flip:
        ys_norm = 1 - ys_norm
    return ys_norm * 10.
