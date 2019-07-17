# 第9章 自然言語処理による感情分析（BERT）
# 実装参考に使用
# https://github.com/huggingface/pytorch-pretrained-BERT

# Copyright (c) 2018 Hugging Face
# Released under the Apache License 2.0
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/LICENSE



# 必要なパッケージのimport

import copy
import math
import json
from attrdict import AttrDict
import collections

import torch
from torch import nn

from utils.tokenizer import BasicTokenizer, WordpieceTokenizer


def get_config(file_path):
    # 設定をconfig.jsonから読み込み、JSONの辞書変数をオブジェクト変数に変換
    config_file = file_path  # "./weights/bert_config.json"

    # ファイルを開き、JSONとして読み込む
    json_file = open(config_file, 'r')
    json_object = json.load(json_file)

    # 辞書変数をオブジェクト変数に
    config = AttrDict(json_object)

    return config


# BERT用にLayerNormalization層を定義します。
# 実装の細かな点をTensorFlowに合わせています。
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """LayerNormalization層です。
        学習済みモデルをそのままロードするため、学習済みモデルの変数名に変えています。
        オリジナルのGitHubの実装から変数名を変えています。
        weight→gamma、bias→beta
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # weightのこと
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # biasのこと
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


# BERTのEmbeddingsモジュールです


class BertEmbeddings(nn.Module):
    """文章の単語ID列と、1文目か2文目かの情報を、埋め込みベクトルに変換する
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        # 3つのベクトル表現の埋め込み

        # Token Embedding：単語IDを単語ベクトルに変換、
        # vocab_size = 30522でBERTの学習済みモデルで使用したボキャブラリーの量
        # hidden_size = 768 で特徴量ベクトルの長さは768
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        # （注釈）padding_idx=0はidx=0の単語のベクトルは0にする。BERTのボキャブラリーのidx=0が[PAD]である。

        # Transformer Positional Embedding：位置情報テンソルをベクトルに変換
        # Transformerの場合はsin、cosからなる固定値だったが、BERTは学習させる
        # max_position_embeddings = 512　で文の長さは512単語
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)

        # Sentence Embedding：文章の1文目、2文目の情報をベクトルに変換
        # type_vocab_size = 2
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        # 作成したLayerNormalization層
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # Dropout　'hidden_dropout_prob': 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        '''
        input_ids： [batch_size, seq_len]の文章の単語IDの羅列
        token_type_ids：[batch_size, seq_len]の各単語が1文目なのか、2文目なのかを示すid
        '''

        # 1. Token Embeddings
        # 単語IDを単語ベクトルに変換
        words_embeddings = self.word_embeddings(input_ids)

        # 2. Sentence Embedding
        # token_type_idsがない場合は文章の全単語を1文目として、0にする
        # そこで、input_idsと同じサイズのゼロテンソルを作成
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 3. Transformer Positional Embedding：
        # [0, 1, 2 ・・・]と文章の長さだけ、数字が1つずつ昇順に入った
        # [batch_size, seq_len]のテンソルposition_idsを作成
        # position_idsを入力して、position_embeddings層から768次元のテンソルを取り出す
        seq_length = input_ids.size(1)  # 文章の長さ
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # 3つの埋め込みテンソルを足し合わせる [batch_size, seq_len, hidden_size]
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        # LayerNormalizationとDropoutを実行
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertLayer(nn.Module):
    '''BERTのBertLayerモジュールです。Transformerになります'''

    def __init__(self, config):
        super(BertLayer, self).__init__()

        # Self-Attention部分
        self.attention = BertAttention(config)

        # Self-Attentionの出力を処理する全結合層
        self.intermediate = BertIntermediate(config)

        # Self-Attentionによる特徴量とBertLayerへの元の入力を足し算する層
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        '''
        hidden_states：Embedderモジュールの出力テンソル[batch_size, seq_len, hidden_size]
        attention_mask：Transformerのマスクと同じ働きのマスキング
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            attention_output, attention_probs = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output, attention_probs

        elif attention_show_flg == False:
            attention_output = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output  # [batch_size, seq_length, hidden_size]


class BertAttention(nn.Module):
    '''BertLayerモジュールのSelf-Attention部分です'''

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.selfattn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, attention_show_flg=False):
        '''
        input_tensor：Embeddingsモジュールもしくは前段のBertLayerからの出力
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            self_output, attention_probs = self.selfattn(
                input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output, attention_probs

        elif attention_show_flg == False:
            self_output = self.selfattn(
                input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output


class BertSelfAttention(nn.Module):
    '''BertAttentionのSelf-Attentionです'''

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        # num_attention_heads': 12

        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size  # = 'hidden_size': 768

        # Self-Attentionの特徴量を作成する全結合層
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        '''multi-head Attention用にテンソルの形を変換する
        [batch_size, seq_len, hidden] → [batch_size, 12, seq_len, hidden/12] 
        '''
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        '''
        hidden_states：Embeddingsモジュールもしくは前段のBertLayerからの出力
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''
        # 入力を全結合層で特徴量変換（注意、multi-head Attentionの全部をまとめて変換しています）
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # multi-head Attention用にテンソルの形を変換
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 特徴量同士を掛け算して似ている度合をAttention_scoresとして求める
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        # マスクがある部分にはマスクをかけます
        attention_scores = attention_scores + attention_mask
        # （備考）
        # マスクが掛け算でなく足し算なのが直感的でないですが、このあとSoftmaxで正規化するので、
        # マスクされた部分は-infにしたいです。 attention_maskには、0か-infが
        # もともと入っているので足し算にしています。

        # Attentionを正規化する
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # ドロップアウトします
        attention_probs = self.dropout(attention_probs)

        # Attention Mapを掛け算します
        context_layer = torch.matmul(attention_probs, value_layer)

        # multi-head Attentionのテンソルの形をもとに戻す
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # attention_showのときは、attention_probsもリターンする
        if attention_show_flg == True:
            return context_layer, attention_probs
        elif attention_show_flg == False:
            return context_layer


class BertSelfOutput(nn.Module):
    '''BertSelfAttentionの出力を処理する全結合層です'''

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 'hidden_dropout_prob': 0.1

    def forward(self, hidden_states, input_tensor):
        '''
        hidden_states：BertSelfAttentionの出力テンソル
        input_tensor：Embeddingsモジュールもしくは前段のBertLayerからの出力
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def gelu(x):
    '''Gaussian Error Linear Unitという活性化関数です。
    LeLUが0でカクっと不連続なので、そこを連続になるように滑らかにした形のLeLUです。
    '''
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertIntermediate(nn.Module):
    '''BERTのTransformerBlockモジュールのFeedForwardです'''

    def __init__(self, config):
        super(BertIntermediate, self).__init__()

        # 全結合層：'hidden_size': 768、'intermediate_size': 3072
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        # 活性化関数gelu
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        '''
        hidden_states： BertAttentionの出力テンソル
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)  # GELUによる活性化
        return hidden_states


class BertOutput(nn.Module):
    '''BERTのTransformerBlockモジュールのFeedForwardです'''

    def __init__(self, config):
        super(BertOutput, self).__init__()

        # 全結合層：'intermediate_size': 3072、'hidden_size': 768
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # 'hidden_dropout_prob': 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        '''
        hidden_states： BertIntermediateの出力テンソル
        input_tensor：BertAttentionの出力テンソル
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# BertLayerモジュールの繰り返し部分モジュールの繰り返し部分です


class BertEncoder(nn.Module):
    def __init__(self, config):
        '''BertLayerモジュールの繰り返し部分モジュールの繰り返し部分です'''
        super(BertEncoder, self).__init__()

        # config.num_hidden_layers の値、すなわち12 個のBertLayerモジュールを作ります
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        hidden_states：Embeddingsモジュールの出力
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：返り値を全TransformerBlockモジュールの出力にするか、
        それとも、最終層だけにするかのフラグ。
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        # 返り値として使うリスト
        all_encoder_layers = []

        # BertLayerモジュールの処理を繰り返す
        for layer_module in self.layer:

            if attention_show_flg == True:
                '''attention_showのときは、attention_probsもリターンする'''
                hidden_states, attention_probs = layer_module(
                    hidden_states, attention_mask, attention_show_flg)
            elif attention_show_flg == False:
                hidden_states = layer_module(
                    hidden_states, attention_mask, attention_show_flg)

            # 返り値にBertLayerから出力された特徴量を12層分、すべて使用する場合の処理
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        # 返り値に最後のBertLayerから出力された特徴量だけを使う場合の処理
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        # attention_showのときは、attention_probs（最後の12段目）もリターンする
        if attention_show_flg == True:
            return all_encoder_layers, attention_probs
        elif attention_show_flg == False:
            return all_encoder_layers


class BertPooler(nn.Module):
    '''入力文章の1単語目[cls]の特徴量を変換して保持するためのモジュール'''

    def __init__(self, config):
        super(BertPooler, self).__init__()

        # 全結合層、'hidden_size': 768
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 1単語目の特徴量を取得
        first_token_tensor = hidden_states[:, 0]

        # 全結合層で特徴量変換
        pooled_output = self.dense(first_token_tensor)

        # 活性化関数Tanhを計算
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertModel(nn.Module):
    '''モジュールを全部つなげたBERTモデル'''

    def __init__(self, config):
        super(BertModel, self).__init__()

        # 3つのモジュールを作成
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        token_type_ids： [batch_size, sequence_length]の、各単語が1文目なのか、2文目なのかを示すid
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：最終出力に12段のTransformerの全部をリストで返すか、最後だけかを指定
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        # Attentionのマスクと文の1文目、2文目のidが無ければ作成する
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # マスクの変形　[minibatch, 1, 1, seq_length]にする
        # 後ほどmulti-head Attentionで使用できる形にしたいので
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # マスクは0、1だがソフトマックスを計算したときにマスクになるように、0と-infにする
        # -infの代わりに-10000にしておく
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 順伝搬させる
        # BertEmbeddinsモジュール
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # BertLayerモジュール（Transformer）を繰り返すBertEncoderモジュール
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''

            encoded_layers, attention_probs = self.encoder(embedding_output,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers, attention_show_flg)

        elif attention_show_flg == False:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers, attention_show_flg)

        # BertPoolerモジュール
        # encoderの一番最後のBertLayerから出力された特徴量を使う
        pooled_output = self.pooler(encoded_layers[-1])

        # output_all_encoded_layersがFalseの場合はリストではなく、テンソルを返す
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # attention_showのときは、attention_probs（1番最後の）もリターンする
        if attention_show_flg == True:
            return encoded_layers, pooled_output, attention_probs
        elif attention_show_flg == False:
            return encoded_layers, pooled_output


# 言語モデル学習用のモジュール（推論時には使わない）
class BertPreTrainingHeads(nn.Module):
    '''BERTの事前学習課題を行うアダプターモジュール'''

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()

        # 事前学習課題：Masked Language Model用のモジュール
        self.predictions = MaskedWordPredictions(config)

        # 事前学習課題：Next Sentence Prediction用のモジュール
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        '''入力情報
        sequence_output:[batch_size, seq_len, hidden_size]
        pooled_output:[batch_size, hidden_size]
        '''
        # 入力のマスクされた各単語がどの単語かを判定
        # 出力 [minibatch, seq_len, vocab_size]
        prediction_scores = self.predictions(sequence_output)

        # 先頭単語の特徴量から1文目と2文目がつながっているかを判定
        seq_relationship_score = self.seq_relationship(
            pooled_output)  # 出力 [minibatch, 2]

        return prediction_scores, seq_relationship_score


# 事前学習課題：Masked Language Model用のモジュール


class MaskedWordPredictions(nn.Module):
    def __init__(self, config):
        '''事前学習課題：Masked Language Model用のモジュール
        元の[2]の実装では、BertLMPredictionHeadという名前です。
        '''
        super(MaskedWordPredictions, self).__init__()

        # BERTから出力された特徴量を変換するモジュール（入出力のサイズは同じ）
        self.transform = BertPredictionHeadTransform(config)

        # self.transformの出力から、各位置の単語がどれかを当てる全結合層
        self.decoder = nn.Linear(in_features=config.hidden_size,  # 'hidden_size': 768
                                 out_features=config.vocab_size,  # 'vocab_size': 30522
                                 bias=False)
        # バイアス項
        self.bias = nn.Parameter(torch.zeros(
            config.vocab_size))  # 'vocab_size': 30522

    def forward(self, hidden_states):
        '''
        hidden_states：BERTからの出力[batch_size, seq_len, hidden_size]
        '''
        # BERTから出力された特徴量を変換
        # 出力サイズ：[batch_size, seq_len, hidden_size]
        hidden_states = self.transform(hidden_states)

        # 各位置の単語がボキャブラリーのどの単語なのかをクラス分類で予測
        # 出力サイズ：[batch_size, seq_len, vocab_size]
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    '''MaskedWordPredictionsにて、BERTからの特徴量を変換するモジュール（入出力のサイズは同じ）'''

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()

        # 全結合層 'hidden_size': 768
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # 活性化関数gelu
        self.transform_act_fn = gelu

        # LayerNormalization
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        '''hidden_statesはsequence_output:[minibatch, seq_len, hidden_size]'''
        # 全結合層で特徴量変換し、活性化関数geluを計算したあと、LayerNormalizationする
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

# 事前学習課題：Next Sentence Prediction用のモジュール


class SeqRelationship(nn.Module):
    def __init__(self, config, out_features):
        '''事前学習課題：Next Sentence Prediction用のモジュール
        元の引用[2]の実装では、とくにクラスとして用意はしていない。
        ただの全結合層に、わざわざ名前をつけた。
        '''
        super(SeqRelationship, self).__init__()

        # 先頭単語の特徴量から1文目と2文目がつながっているかを判定するクラス分類の全結合層
        self.seq_relationship = nn.Linear(config.hidden_size, out_features)

    def forward(self, pooled_output):
        return self.seq_relationship(pooled_output)


class BertForMaskedLM(nn.Module):
    '''BERTモデルに、事前学習課題用のアダプターモジュール
    BertPreTrainingHeadsをつなげたモデル'''

    def __init__(self, config, net_bert):
        super(BertForMaskedLM, self).__init__()

        # BERTモジュール
        self.bert = net_bert  # BERTモデル

        # 事前学習課題用のアダプターモジュール
        self.cls = BertPreTrainingHeads(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        token_type_ids： [batch_size, sequence_length]の、各単語が1文目なのか、2文目なのかを示すid
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        '''

        # BERTの基本モデル部分の順伝搬
        encoded_layers, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=False)

        # 事前学習課題の推論を実施
        prediction_scores, seq_relationship_score = self.cls(
            encoded_layers, pooled_output)

        return prediction_scores, seq_relationship_score


# 学習済みモデルのロード
def set_learned_params(net, weights_path = "./weights/pytorch_model.bin"):

    # セットするパラメータを読み込む
    loaded_state_dict = torch.load(weights_path)

    # 現在のネットワークモデルのパラメータ名
    net.eval()
    param_names = []  # パラメータの名前を格納していく

    for name, param in net.named_parameters():
        param_names.append(name)

    # 現在のネットワークの情報をコピーして新たなstate_dictを作成
    new_state_dict = net.state_dict().copy()

    # 新たなstate_dictに学習済みの値を代入
    for index, (key_name, value) in enumerate(loaded_state_dict.items()):
        name = param_names[index]  # 現在のネットワークでのパラメータ名を取得
        new_state_dict[name] = value  # 値を入れる
        print(str(key_name)+"→"+str(name))  # 何から何に入ったかを表示

        # 現在のネットワークのパラメータを全部ロードしたら終える
        if (index+1 - len(param_names)) >= 0:
            break

    # 新たなstate_dictを構築したBERTモデルに与える
    net.load_state_dict(new_state_dict)

    return net


# BERT用のTokenizer
# vocabファイルを読み込み、


def load_vocab(vocab_file):
    """text形式のvocabファイルの内容を辞書に格納します"""
    vocab = collections.OrderedDict()  # (単語, id)の順番の辞書変数
    ids_to_tokens = collections.OrderedDict()  # (id, 単語)の順番の辞書変数
    index = 0

    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()

            # 格納
            vocab[token] = index
            ids_to_tokens[index] = token
            index += 1

    return vocab, ids_to_tokens


# BasicTokenizer, WordpieceTokenizerは、引用文献[2]そのままです
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py
# これらはsub-wordで単語分割を行うクラスになります。
class BertTokenizer(object):
    '''BERT用の文章の単語分割クラスを実装'''

    def __init__(self, vocab_file, do_lower_case=True):
        '''
        vocab_file：ボキャブラリーへのパス
        do_lower_case：前処理で単語を小文字化するかどうか
        '''

        # ボキャブラリーのロード
        self.vocab, self.ids_to_tokens = load_vocab(vocab_file)

        # 分割処理の関数をフォルダ「utils」からimoprt、sub-wordで単語分割を行う
        never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        # (注釈)上記の単語は途中で分割させない。これで一つの単語とみなす

        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                              never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        '''文章を単語に分割する関数'''
        split_tokens = []  # 分割後の単語たち
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """分割された単語リストをIDに変換する関数"""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])

        return ids

    def convert_ids_to_tokens(self, ids):
        """IDを単語に変換する関数"""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
