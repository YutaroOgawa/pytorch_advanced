# 第4章姿勢推定のネットワーク
# 実装参考に使用
# https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/


# 必要なパッケージのimport
import torch
import torch.nn as nn
from torch.nn import init
import torchvision


def make_stages(cfg_dict):
    """
    コンフィグレーション変数からサブネットワークを作成
    CPMはConvolutional Pose Machinesの略称。
    以前のCPM論文の構造を参考にしている。
    """
    layers = []

    # 1つ目から最後の1つ手前までの層
    for i in range(len(cfg_dict) - 1):
        one_ = cfg_dict[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]

    # 最後の畳み込み層
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]

    net = nn.Sequential(*layers)

    # 初期化関数の設定。畳み込み層を初期化
    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:  # mobilenet conv2d doesn't add bias
                    init.constant_(m.bias, 0.0)

    net.apply(_initialize_weights_norm)

    return net


class OpenPoseNet(nn.Module):
    def __init__(self):
        super(OpenPoseNet, self).__init__()

        # 1. Featureモジュールを作成
        # VGG19の最初10個の畳み込みを使用
        vgg19 = torchvision.models.vgg19(pretrained=True)
        models = {}
        models['block0'] = vgg19.features[0:23]  # VGG19の最初の10個の畳み込み層まで

        # 残りは新たな畳み込み層を2つ用意
        models['block0'].add_module("23", torch.nn.Conv2d(
            512, 256, kernel_size=3, stride=1, padding=1))
        models['block0'].add_module("24", torch.nn.ReLU(inplace=True))
        models['block0'].add_module("25", torch.nn.Conv2d(
            256, 128, kernel_size=3, stride=1, padding=1))
        models['block0'].add_module("26", torch.nn.ReLU(inplace=True))

        # 2.CPMモジュールを作成します
        # CPMはConvolutional Pose Machinesの略称。
        # コンフィグレーションの辞書変数blocksを作成し、ネットワークを生成させる
        blocks = {}

        # Stage 1
        blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                              {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                              {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                              {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                              {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

        blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                              {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                              {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                              {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                              {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = [
                {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
                {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
                {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
                {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
            ]

            blocks['block%d_2' % i] = [
                {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
                {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
                {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
                {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
            ]

        # コンフィグレーションファイルからサブネットワークを生成
        for k, v in blocks.items():
            models[k] = make_stages(list(v))

        # 3. クラスのメンバ変数にします
        self.model0 = models['block0']

        # PAFs（Part Affinity Fields）側
        self.model1_1 = models['block1_1']
        self.model2_1 = models['block2_1']
        self.model3_1 = models['block3_1']
        self.model4_1 = models['block4_1']
        self.model5_1 = models['block5_1']
        self.model6_1 = models['block6_1']

        # confidence heatmap側
        self.model1_2 = models['block1_2']
        self.model2_2 = models['block2_2']
        self.model3_2 = models['block3_2']
        self.model4_2 = models['block4_2']
        self.model5_2 = models['block5_2']
        self.model6_2 = models['block6_2']

    def forward(self, x):
        """順伝搬の定義"""

        # Featureモジュール
        out1 = self.model0(x)

        # CPMのステージ1
        out1_1 = self.model1_1(out1)  # PAFs側
        out1_2 = self.model1_2(out1)  # confidence heatmap側

        # CPMのステージ2
        out2 = torch.cat([out1_1, out1_2, out1], 1)  # 次元1のチャネルで結合
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)

        # CPMのステージ3
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)

        # CPMのステージ4
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)

        # CPMのステージ5
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)

        # CPMのステージ6
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        # 損失の計算用に各ステージの結果を格納
        saved_for_loss = []
        saved_for_loss.append(out1_1)  # PAFs側
        saved_for_loss.append(out1_2)  # confidence heatmap側
        saved_for_loss.append(out2_1)
        saved_for_loss.append(out2_2)
        saved_for_loss.append(out3_1)
        saved_for_loss.append(out3_2)
        saved_for_loss.append(out4_1)
        saved_for_loss.append(out4_2)
        saved_for_loss.append(out5_1)
        saved_for_loss.append(out5_2)
        saved_for_loss.append(out6_1)
        saved_for_loss.append(out6_2)

        # 最終的なPAFsのout6_1とconfidence heatmapのout6_2、そして
        # 損失計算用に各ステージでのPAFsとheatmapを格納したsaved_for_lossを出力
        # out6_1：torch.Size([2, 38, 46, 46])
        # out6_2：torch.Size([2, 19, 46, 46])
        # saved_for_loss:[out1_1, out_1_2, ・・・, out6_2]

        return (out6_1, out6_2), saved_for_loss
