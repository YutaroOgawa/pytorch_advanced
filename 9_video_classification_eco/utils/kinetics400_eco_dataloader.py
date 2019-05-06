# 第9章 動画分類（ECO：Efficient 3DCNN）
# 9.4	Kinetics動画データセットからDataLoaderの作成

# 必要なパッケージのimport
import os
from PIL import Image
import csv
import numpy as np

import torch
import torch.utils.data
from torch import nn

import torchvision


def make_datapath_list(root_path):
    """
    動画を画像データにしたフォルダへのファイルパスリストを作成する。
    root_path : str、データフォルダへのrootパス
    Returns：ret : video_list、動画を画像データにしたフォルダへのファイルパスリスト
    """

    # 動画を画像データにしたフォルダへのファイルパスリスト
    video_list = list()

    # root_pathにある、クラスの種類とパスを取得
    class_list = os.listdir(path=root_path)

    # 各クラスの動画ファイルを画像化したフォルダへのパスを取得
    for class_list_i in (class_list):  # クラスごとのループ

        # クラスのフォルダへのパスを取得
        class_path = os.path.join(root_path, class_list_i)

        # 各クラスのフォルダ内の画像フォルダを取得するループ
        for file_name in os.listdir(class_path):

            # ファイル名と拡張子に分割
            name, ext = os.path.splitext(file_name)

            # フォルダでないmp4ファイルは無視
            if ext == '.mp4':
                continue

            # 動画ファイルを画像に分割して保存したフォルダのパスを取得
            video_img_directory_path = os.path.join(class_path, name)

            # vieo_listに追加
            video_list.append(video_img_directory_path)

    return video_list


class VideoTransform():
    """
    動画を画像にした画像ファイルの前処理クラス。学習時と推論時で異なる動作をします。
    動画を画像に分割しているため、分割された画像たちをまとめて前処理する点に注意してください。
    """

    def __init__(self, resize, crop_size, mean, std):
        self.data_transform = {
            'train': torchvision.transforms.Compose([
                # DataAugumentation()  # 今回は省略
                GroupResize(int(resize)),  # 画像をまとめてリサイズ　
                GroupCenterCrop(crop_size),  # 画像をまとめてセンタークロップ
                GroupToTensor(),  # データをPyTorchのテンソルに
                GroupImgNormalize(mean, std),  # データを標準化
                Stack()  # 複数画像をframes次元で結合させる
            ]),
            'val': torchvision.transforms.Compose([
                GroupResize(int(resize)),  # 画像をまとめてリサイズ　
                GroupCenterCrop(crop_size),  # 画像をまとめてセンタークロップ
                GroupToTensor(),  # データをPyTorchのテンソルに
                GroupImgNormalize(mean, std),  # データを標準化
                Stack()  # 複数画像をframes次元で結合させる
            ])
        }

    def __call__(self, img_group, phase):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img_group)


# 前処理で使用するクラスたちの定義


class GroupResize():
    ''' 画像をまとめてリスケールするクラス。
    画像の短い方の辺の長さがresizeに変換される。
    アスペクト比は保たれる。
    '''

    def __init__(self, resize, interpolation=Image.BILINEAR):
        '''リスケールする処理を用意'''
        self.rescaler = torchvision.transforms.Resize(resize, interpolation)

    def __call__(self, img_group):
        '''リスケールをimg_group(リスト)内の各imgに実施'''
        return [self.rescaler(img) for img in img_group]


class GroupCenterCrop():
    ''' 画像をまとめてセンタークロップするクラス。
        （crop_size, crop_size）の画像を切り出す。
    '''

    def __init__(self, crop_size):
        '''センタークロップする処理を用意'''
        self.ccrop = torchvision.transforms.CenterCrop(crop_size)

    def __call__(self, img_group):
        '''センタークロップをimg_group(リスト)内の各imgに実施'''
        return [self.ccrop(img) for img in img_group]


class GroupToTensor():
    ''' 画像をまとめてテンソル化するクラス。
    '''

    def __init__(self):
        '''テンソル化する処理を用意'''
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, img_group):
        '''テンソル化をimg_group(リスト)内の各imgに実施
        0から1ではなく、0から255で扱うため、255をかけ算する。
        0から255で扱うのは、学習済みデータの形式に合わせるため
        '''

        return [self.to_tensor(img)*255 for img in img_group]


class GroupImgNormalize():
    ''' 画像をまとめて標準化するクラス。
    '''

    def __init__(self, mean, std):
        '''標準化する処理を用意'''
        self.normlize = torchvision.transforms.Normalize(mean, std)

    def __call__(self, img_group):
        '''標準化をimg_group(リスト)内の各imgに実施'''
        return [self.normlize(img) for img in img_group]


class Stack():
    ''' 画像を一つのテンソルにまとめるクラス。
    '''

    def __call__(self, img_group):
        '''img_groupはtorch.Size([3, 224, 224])を要素とするリスト
        '''
        ret = torch.cat([(x.flip(dims=[0])).unsqueeze(dim=0)
                         for x in img_group], dim=0)  # frames次元で結合
        # x.flip(dims=[0])は色チャネルをRGBからBGRへと順番を変えています（元の学習データがBGRであったため）
        # unsqueeze(dim=0)はあらたにframes用の次元を作成しています

        return ret


def get_label_id_dictionary(label_dicitionary_path='./video_download/kinetics_400_label_dicitionary.csv'):
    label_id_dict = {}
    id_label_dict = {}

    # eoncodingはUbuntuもこれで良いのか、確認せねば
    with open(label_dicitionary_path, encoding="utf-8_sig") as f:

        # 読み込む
        reader = csv.DictReader(f, delimiter=",", quotechar='"')

        # 1行ずつ読み込み、辞書型変数に追加します
        for row in reader:
            label_id_dict.setdefault(
                row["class_label"], int(row["label_id"])-1)
            id_label_dict.setdefault(
                int(row["label_id"])-1, row["class_label"])

    return label_id_dict,  id_label_dict


class VideoDataset(torch.utils.data.Dataset):
    """
    動画のDataset
    """

    def __init__(self, video_list, label_id_dict, num_segments, phase, transform, img_tmpl='image_{:05d}.jpg'):
        self.video_list = video_list  # 動画画像のフォルダへのパスリスト
        self.label_id_dict = label_id_dict  # ラベル名をidに変換する辞書型変数
        self.num_segments = num_segments  # 動画を何分割して使用するのかを決める
        self.phase = phase  # train or val
        self.transform = transform  # 前処理
        self.img_tmpl = img_tmpl  # 読み込みたい画像のファイル名のテンプレート

    def __len__(self):
        '''動画の数を返す'''
        return len(self.video_list)

    def __getitem__(self, index):
        '''
        前処理をした画像たちのデータとラベル、ラベルIDを取得
        '''
        imgs_transformed, label, label_id, dir_path = self.pull_item(index)
        return imgs_transformed, label, label_id, dir_path

    def pull_item(self, index):
        '''前処理をした画像たちのデータとラベル、ラベルIDを取得'''

        # 1. 画像たちをリストに読み込む
        dir_path = self.video_list[index]  # 画像が格納されたフォルダ
        indices = self._get_indices(dir_path)  # 読み込む画像idxを求める
        img_group = self._load_imgs(
            dir_path, self.img_tmpl, indices)  # リストに読み込む

        # 2. ラベルの取得し、idに変換する
        label = (dir_path.split('/')[3].split('/')[0])  # 注意：windowsOSの場合
        label_id = self.label_id_dict[label]  # idを取得

        # 3. 前処理を実施
        imgs_transformed = self.transform(img_group, phase=self.phase)

        return imgs_transformed, label, label_id, dir_path

    def _load_imgs(self, dir_path, img_tmpl, indices):
        '''画像をまとめて読み込み、リスト化する関数'''
        img_group = []  # 画像を格納するリスト

        for idx in indices:
            # 画像のパスを取得
            file_path = os.path.join(dir_path, img_tmpl.format(idx))

            # 画像を読み込む
            img = Image.open(file_path).convert('RGB')

            # リストに追加
            img_group.append(img)
        return img_group

    def _get_indices(self, dir_path):
        """
        動画全体をself.num_segmentに分割した際に取得する動画のidxのリストを取得する
        """
        # 動画のフレーム数を求める
        file_list = os.listdir(path=dir_path)
        num_frames = len(file_list)

        # 動画の取得間隔幅を求める
        tick = (num_frames) / float(self.num_segments)
        # 250 / 16 = 15.625
        # 動画の取得間隔幅で取り出す際のidxをリストで求める
        indices = np.array([int(tick / 2.0 + tick * x)
                            for x in range(self.num_segments)])+1
        # 250frameで16frame抽出の場合
        # indices = [  8  24  40  55  71  86 102 118 133 149 165 180 196 211 227 243]

        return indices
