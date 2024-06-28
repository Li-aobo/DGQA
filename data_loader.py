import torch
import torchvision
import folders

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain

        if istrain:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])

        if dataset == 'kadis-700k':
            self.data = folders.KADIS_700kFolder(
                    root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'livec':
            self.data = folders.LIVEChallengeFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'koniq-10k':
            self.data = folders.Koniq_10kFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'bid':
            self.data = folders.BIDFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'pipal':
            self.data = folders.PIPALFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'spaq':
            self.data = folders.SPAQFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'kadid-10k':
            self.data = folders.KADID_10kFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False)
        return dataloader
