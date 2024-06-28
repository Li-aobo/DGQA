import os
import pickle
import argparse
import random
import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import data_loader
from baseline import Baseline


def seed_torch(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_db_base_info(root, dataset):
    folder_path = {
        # 'kadis-700k': '/kadis700k/',
        # 'kadid-10k': '/KADID_10k/',
        'livec': '/ChallengeDB_release/',
        'koniq-10k': '/KonIQ_10k/',
        'bid': '/BID_512/',
        'pipal': '/PIPAL/',
        'spaq': '/SPAQ_512/',
    }
    folder_path = {key: root + val for key, val in folder_path.items()}[dataset]

    img_num = {
        # 'kadis-700k': list(range(0, 140000)),
        # 'kadid-10k': list(range(0, 81)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
        'pipal': list(range(0, 200)),
        'spaq': list(range(0, 11125)),
    }
    sel_num = img_num[dataset]
    return folder_path, sel_num


def main(config):
    # ----- 1. get distortion features -----
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.device)
    seed_torch(seed=config.seed)

    folder_path, sel_num = get_db_base_info(config.root, config.dataset)
    img_data = data_loader.DataLoader(config.dataset, folder_path, sel_num, config.patch_size,
                                      config.test_patch_num, batch_size=1, istrain=False).get_data()

    model = Baseline().cuda()
    model.load_state_dict(torch.load('./model/kadis-700k_model_best.pth.tar')['state_dict'])
    model.eval()

    preds = []
    names = []
    progress = tqdm.tqdm(img_data)
    for i, (imgname, img, _) in enumerate(progress):
        img = img.cuda()
        pred = model(img)

        names += list(imgname)
        preds.append(pred.cpu().detach().numpy())

    preds = torch.mean(torch.reshape(torch.tensor(preds), (-1, config.test_patch_num, pred.size(-1))), dim=1)

    res = {'names': names[0::config.test_patch_num], 'preds': preds}
    if not os.path.exists('pred_res'):
        os.makedirs('pred_res')
    with open('./pred_res/{}_dist_pred.pkl'.format(config.dataset), 'wb') as f:
        pickle.dump(res, f)

    # ----- 2. select distortion types -----
    ntop = 4
    _, top5 = preds.topk(ntop, 1, True, True)
    unique, counts = torch.unique(top5, return_counts=True)

    all_count = preds.size()[0] * (ntop - 1)
    for dist_type, count in zip(unique, counts):
        if round(count.item() / all_count, 2) >= 0.04:  #
            print(dist_type.item(), round(count.item() / all_count, 3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', dest='device', type=int, default=0, help='0, 1, 2, 3')
    parser.add_argument('--dataset', dest='dataset', type=str, default='livec',
                        help='Support datasets: livec|koniq-10k|bid|spaq|pipal')
    parser.add_argument('--seed', dest='seed', type=int, default=123, help='Random seed')
    parser.add_argument('--root', dest='root', type=str, default='', help='Root of datasets')

    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for testing image patches')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=5,
                        help='Number of sample patches from testing image')

    config = parser.parse_args()
    main(config)

