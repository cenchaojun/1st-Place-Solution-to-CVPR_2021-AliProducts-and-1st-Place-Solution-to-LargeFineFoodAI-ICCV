import os
import pandas as pd
import torch
import json
from transforms import transforms
from utils.autoaugment import ImageNetPolicy
import pdb

# pretrained model checkpoints
pretrained_model = {
    'resnet50': './models/pretrained/resnet50-19c8e357.pth',
}


# transforms dict
def load_data_transformers(resize_reso=512, crop_reso=448, swap_num=[7, 7]):
    center_resize = 600
    Normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    data_transforms = {
        'swap':
        transforms.Compose([
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
        'common_aug':
        transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
        ]),
        'train_totensor':
        transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            # ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val_totensor':
        transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test_totensor':
        transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            #transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.FiveCrop(384),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
            #transforms.ToTensor(),
            transforms.Normalize_Crop([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225]),
        ]),
        'None':
        None,
    }
    return data_transforms


class LoadConfig(object):
    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        else:
            raise Exception("train/val/test ???\n")

        ###############################
        #### add dataset info here ####
        ###############################

        # put image data in $PATH/data
        # put annotation txt file in $PATH/anno
        self.data_sample = False
        if args.dataset == 'clothes':
            self.dataset = args.dataset
            self.rawdata_root = '/data1/zengpeng/data/clothes'
            self.anno_root = '/data1/zengpeng/data/clothes'
            self.numcls = 151
        elif args.dataset == 'shopee':
            self.dataset = args.dataset
            self.rawdata_root = '/data1/zengpeng/shopee/input/shopee-product-matching'
            self.anno_root = self.rawdata_root
            self.numcls = 11013
        elif args.dataset == 'cvpr':
            self.dataset = args.dataset
            self.rawdata_root = '/data1/zengpeng/products'
            self.anno_root = '/data1/zengpeng/products'
            self.numcls = 50030

        elif args.dataset == 'style':
            self.dataset = args.dataset
            self.rawdata_root = '/data2/zengpeng/labels/style_data'
            self.anno_root = '/data2/zengpeng/labels/style_data'
            self.numcls = 77

        elif args.dataset == 'color':
            self.dataset = args.dataset
            self.rawdata_root = '/data3/zengpeng/labels'
            self.anno_root = '/data3/zengpeng/labels'
            self.numcls = 67

        elif args.dataset == 'food':
            self.dataset = args.dataset
            self.rawdata_root = '/data3/zengpeng/products/food/classify'
            self.anno_root = '/data3/zengpeng/products/food/classify'
            self.numcls = 1000

        elif args.dataset == 'CUB':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/CUB_200_2011/data'
            self.anno_root = './dataset/CUB_200_2011/anno'
            self.numcls = 200
        elif args.dataset == 'STCAR':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/st_car/data'
            self.anno_root = './dataset/st_car/anno'
            self.numcls = 196
        elif args.dataset == 'AIR':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/aircraft/data'
            self.anno_root = './dataset/aircraft/anno'
            self.numcls = 100
        else:
            raise Exception('dataset not defined ???')

        # annotation file organized as :
        # path/image_name cls_num\n
        if self.dataset == 'cvpr':
            if 'train' in get_list:
                with open(self.rawdata_root + "/train_clean_v1.json",
                          'r',
                          encoding='utf-8') as train_h:
                    train_data = {
                        'train/' + one['class_id'] + '/' + one['image_id']:
                        int(one['class_id'])
                        for one in json.load(train_h)['images']
                    }
                    self.train_anno = pd.DataFrame(
                        list(train_data.items()),
                        columns=['ImageName', 'label'])
                    if self.data_sample:
                        self.train_anno = self.train_anno.sample(
                            frac=1).groupby(['label']).head(self.data_sample)
            if 'val' in get_list:
                with open(self.rawdata_root + "/val.json",
                          'r',
                          encoding='utf-8') as val_h:
                    val_data = {
                        'val/' + one['class_id'] + '/' + one['image_id']:
                        int(one['class_id'])
                        for one in json.load(val_h)['images']
                    }
                    self.val_anno = pd.DataFrame(
                        list(val_data.items()), columns=['ImageName', 'label'])
            if 'test' in get_list:
                with open(self.rawdata_root + "/test.json",
                          'r',
                          encoding='utf-8') as test_h:
                    test_data = [
                        'test/' + one['image_id']
                        for one in json.load(test_h)['images']
                    ]
                    self.test_anno = pd.DataFrame(test_data,
                                                  columns=['ImageName'])
        elif self.dataset == 'clothes':
            if 'train' in get_list:
                self.train_anno = pd.read_csv(os.path.join(
                    self.anno_root, 'train.txt'),
                                              sep="\t",
                                              header=None,
                                              names=['ImageName', 'label'])

            if 'val' in get_list:
                self.val_anno = pd.read_csv(os.path.join(
                    self.anno_root, 'test.txt'),
                                            sep="\t",
                                            header=None,
                                            names=['ImageName', 'label'])

            if 'test' in get_list:
                self.test_anno = pd.read_csv(os.path.join(
                    self.anno_root, 'test.txt'),
                                             sep="\t",
                                             header=None,
                                             names=['ImageName', 'label'])

        elif self.dataset == 'style':
            if 'train' in get_list:
                self.train_anno = pd.read_csv(os.path.join(
                    self.anno_root, 'train.txt'),
                                              sep="\t",
                                              header=None,
                                              names=['ImageName', 'label'])

            if 'val' in get_list:
                self.val_anno = pd.read_csv(os.path.join(
                    self.anno_root, 'test.txt'),
                                            sep="\t",
                                            header=None,
                                            names=['ImageName', 'label'])

            if 'test' in get_list:
                self.test_anno = pd.read_csv(os.path.join(
                    self.anno_root, 'test.txt'),
                                             sep="\t",
                                             header=None,
                                             names=['ImageName', 'label'])

        elif self.dataset == 'color':
            if 'train' in get_list:
                self.train_anno = pd.read_csv(os.path.join(
                    self.anno_root, 'train.txt'),
                                              sep="\t",
                                              header=None,
                                              names=['ImageName', 'label'])

            if 'val' in get_list:
                self.val_anno = pd.read_csv(os.path.join(
                    self.anno_root, 'test.txt'),
                                            sep="\t",
                                            header=None,
                                            names=['ImageName', 'label'])

            if 'test' in get_list:
                self.test_anno = pd.read_csv(os.path.join(
                    self.anno_root, 'test.txt'),
                                             sep="\t",
                                             header=None,
                                             names=['ImageName'])

        elif self.dataset == 'food':
            if 'train' in get_list:
                self.train_anno = pd.read_csv(os.path.join(
                    self.anno_root, 'train.txt'),
                                              sep="\t",
                                              header=None,
                                              names=['ImageName'])
                self.train_anno['label'] = self.train_anno['ImageName'].apply(
                    lambda a: int(a.split('/')[-2]))

            if 'val' in get_list:
                self.val_anno = pd.read_csv(os.path.join(
                    self.anno_root, 'val.txt'),
                                            sep="\t",
                                            header=None,
                                            names=['ImageName'])
                self.val_anno['label'] = self.val_anno['ImageName'].apply(
                    lambda a: int(a.split('/')[-2]))

            if 'test' in get_list:
                self.test_anno = pd.read_csv(os.path.join(
                    self.anno_root, 'test.txt'),
                                             sep="\t",
                                             header=None,
                                             names=['ImageName'])

        self.swap_num = args.swap_num

        self.save_dir = './net_model'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.backbone = args.backbone

        self.use_dcl = False
        self.use_backbone = False if self.use_dcl else True
        self.use_Asoftmax = False
        self.use_focal_loss = False
        self.use_fpn = False
        self.use_hier = False

        self.weighted_sample = False
        self.cls_2 = True
        self.cls_2xmul = True

        self.log_folder = './logs'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)
