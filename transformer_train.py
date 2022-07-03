#coding=utf-8
import paddle.nn as nn
import paddle
import numpy as np
import paddle.nn.functional as F
from paddle.nn import Linear, Embedding

def scaled_dot_product_attention(q, k, v, mask):
    """计算注意力权重。
        q, k, v 必须具有匹配的前置维度。比如(batch_size, num_heads)
        k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
        虽然 mask 根据其类型（填充或前瞻）有不同的形状，
        但是 mask 必须能进行广播转换以便求和。

        好像大部分情况，q k v的最后一维也是相同的
        参数:
          q: 请求的形状 == (..., seq_len_q, depth)
             q to match others 是负责寻找这个字的于其他字的相关度（通过其它字的key）
          k: 主键的形状 == (..., seq_len_k, depth)
             k to be matched key向量就是用来于query向量作匹配，得到相关度评分的
          v: 数值的形状 == (..., seq_len_v, depth_v)
             v information to be extracted
             v就是原输入经过一层全连接编码后新的词向量
             是实际上的字的表示, 一旦我们得到了字的相关度评分，这些表示是用来加权求和的
          mask: Float 张量，其形状能转换成
                (..., seq_len_q, seq_len_k)。默认为None。

        实际的形状可能就是：
        (batch_size, head_nums, seq_len_q, depth)
        (128,        8,         50,        512)

        返回值:
          输出 shape == (..., seq_len_q, depth_v)
          注意力权重 shape == (..., seq_len_q, seq_len_v)
    """
    matmul_qk = paddle.matmul(q, k, transpose_y=True)
    dk = paddle.cast(paddle.shape(k)[-1], "float32")
    scaled_attention_logits = matmul_qk / paddle.sqrt(dk)
    # 将 mask 加入到缩放的张量上。
    # 加上一个很大的负数，使softmax对应单元变成0
    if mask is not  None:
        scaled_attention_logits += (mask * -1e9)
    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = F.softmax(scaled_attention_logits, axis=-1)
    output = paddle.matmul(attention_weights, v)

    # print("output_shape", output.shape)
    return output, attention_weights


# Multi-head attention
class MultiHeadAttention(nn.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model  # 我暂时把这个理解为词向量维度

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads  # 拆分多头后的维度

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).

        原本 (batch_size, seq_len, d_model) 实际上就是把d_model分成了8份
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """

        x = paddle.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return paddle.transpose(x, perm=[0, 2, 1, 3])


    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]

        # 这是需要学习的参数
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = paddle.transpose(scaled_attention,
                                                  perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, d_model)
        concat_attention = paddle.reshape(scaled_attention,
                                                (batch_size, -1, self.d_model))
        # 要学习的参数
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = paddle.uniform((1, 60, 512))
print("y_shape", y.shape)
# y = fluid.layers.uniform_random((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)
print(out.shape, attn.shape)



class PointWiseFeedForwardNetwork(nn.Layer):
    def __init__(self, d_model, dff):
        super(PointWiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out


sample_ffn = PointWiseFeedForwardNetwork(512, 2048)
x = paddle.uniform((64, 50, 512))  # batch_size, seq_len, d_model
tmp = sample_ffn(x)
print(type(tmp))
print(tmp.shape)



class EncoderLayer(nn.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model)  # epsilon 避免除0
        self.layernorm2 = nn.LayerNorm(d_model)
        self.rate = rate


    def forward(self, x, testing, mask):
        # training 表示是否在训练模式下执行dropout 多头注意力的第二个输出：注意力权重是被忽略的
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        # attn_output = F.dropout(attn_output, self.rate, training=testing)
        # attn_output = nn.dropout(attn_output, is_test=testing, dropout_prob=self.rate)

        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        # ffn_output = F.dropout(ffn_output, self.rate, training=testing)
        # ffn_output = fluid.layers.dropout(ffn_output, is_test=testing, dropout_prob=self.rate)

        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        # out2 作为decoder的v和k
        return out2

# sample_encoder_layer = EncoderLayer(512, 8, 2048)
#
# sample_encoder_layer_output = sample_encoder_layer(
# paddle.uniform((64, 43, 512)), testing=False, mask=None)
# print(type(sample_encoder_layer_output))
# print(sample_encoder_layer_output.shape)  #


class Model(nn.Layer):
    def __init__(self, item_count, cate_count, hidden_units):
        super(Model, self).__init__()

        self.item_count = item_count
        self.cate_count = cate_count

        self.hidden_units = hidden_units

        self.item_emb_layer = Embedding(self.item_count, self.hidden_units // 2)
        self.cate_emb_layer = Embedding(self.cate_count, self.hidden_units // 2)
        self.item_b_layer = Embedding(self.item_count, 1)

        self.sig = nn.Sigmoid()

        self.soft = nn.Softmax()

        self.atten_fc1 = nn.Linear(self.hidden_units * 4, self.hidden_units)
        self.atten_fc2 = nn.Linear(self.hidden_units, hidden_units // 2)
        self.atten_fc3 = nn.Linear(hidden_units // 2, 1)

        # self.atten_bn1 = nn.BatchNorm(self.hidden_units)
        self.out_fc = nn.Linear(self.hidden_units, self.hidden_units)

        self.trans = EncoderLayer(hidden_units, 4, hidden_units)
        ###din_fc
        # self.din_bn = nn.BatchNorm(2*self.hidden_units)

        self.din_fc1 = nn.Linear(2 * self.hidden_units, self.hidden_units)
        self.din_fc2 = nn.Linear(self.hidden_units, self.hidden_units // 2)
        self.din_fc3 = nn.Linear(self.hidden_units // 2, 1)

        ###LOSS
        self.loss_func = paddle.nn.BCELoss()

    def forward(self, user_his, user_cate, target_item, target_cate, sl, label):
        target_item_bias_emb = self.item_b_layer(target_item)
        target_item_bias_emb = target_item_bias_emb.reshape([-1, 1])

        # print("target_item_shape:", target_item.shape)
        # print("target_cate_shape:", target_cate.shape)
        target_item_emb = self.item_emb_layer(target_item)  ###batch*1*hid/2
        target_cate_emb = self.cate_emb_layer(target_cate)  ###batch*1*hid/2
        # print("taegrt_item_meb_shape:", target_item_emb.shape)

        i_emb = paddle.concat([target_item_emb, target_cate_emb], axis=-1)
        i_emb = paddle.squeeze(i_emb, axis=1)
        # print("i_emb_shape:", i_emb.shape)
        ###history
        his_item_emb = self.item_emb_layer(user_his)
        his_cate_emb = self.cate_emb_layer(user_cate)

        his_emb = paddle.concat([his_item_emb, his_cate_emb], axis=-1)  ##batch*max_len*hidden

        ####添加transformer的encoder层
        his_emb = self.trans(his_emb, testing=False, mask=None)


        din_attention = self.attention(i_emb, his_emb, sl)
        din_attention = din_attention.reshape([-1, self.hidden_units])

        # din_attention = paddle.mean(his_emb, axis=1)

        out_fc = self.out_fc(din_attention)
        embedding_concat = paddle.concat([out_fc, i_emb], axis=1)
        # print("embedding_concat_shape", embedding_concat.shape)
        fc1 = self.sig(self.din_fc1(embedding_concat))
        fc2 = self.sig(self.din_fc2(fc1))
        fc3 = self.din_fc3(fc2)

        logits = fc3 + target_item_bias_emb
        input_logits = F.sigmoid(logits).squeeze(axis=-1)
        # print("input_lgotis", input_logits)
        # print("input_logits_shape", input_logits.shape)
        # print("logits_shape:", self.logits.shape)
        # print("slef_lable_shape", label.shape)
        # loss = self.loss_func(input_logits, label)

        # print("loss", loss)
        # ave_loss = paddle.mean(loss)

        # return ave_loss, input_logits

        return input_logits

    def attention(self, querys, keys, keys_length):
        """desc"""
        """
        keys_len batch*1
        """
        querys = paddle.tile(querys, [1, keys.shape[1]])
        querys = querys.reshape([-1, keys.shape[1], self.hidden_units])
        queries_hidden_units = querys.shape[-1]
        # print("querys_shape:", querys.shape)
        # print("keys_shape:", keys.shape)
        concat_vec = paddle.concat([querys, keys, querys - keys, querys * keys], axis=2)
        ###
        # print("concat_vec.shape", concat_vec.shape)

        d_layer_1_all = self.sig(self.atten_fc1(concat_vec))
        d_layer_2_all = self.sig(self.atten_fc2(d_layer_1_all))
        d_layer_3_all = self.sig(self.atten_fc3(d_layer_2_all))
        outputs = d_layer_3_all.reshape([-1, 1, keys.shape[1]])

        max_len = keys.shape[1]

        # print("outputs_shape:", outputs.shape)
        # print("keys_length_shape", keys_length.shape)
        ####sequence_mask操作
        aa = paddle.expand(keys_length, [keys_length.shape[0], max_len])
        cc = np.arange(max_len).reshape([1, -1])
        bb = paddle.expand(paddle.to_tensor(cc), [keys_length.shape[0], max_len])
        attention_mask = (aa > bb)
        attention_mask = attention_mask.reshape([keys_length.shape[0], 1, max_len])
        # print("attention_mask_shape:", attention_mask.shape)

        paddings = paddle.ones(outputs.shape) * (-2 ** 32 + 1)

        #
        outputs = paddle.where(attention_mask, outputs, paddings)  # [B, 1, T]
        # # Scale
        outputs = outputs / (keys.shape[-1] ** 0.5)
        #
        # # Activation
        outputs = F.softmax(outputs)  # [B, 1, T]

        # print("outptus_shape:", outputs.shape)
        # # outputs = paddle.multiply(outputs.squeeze(1), attention_mask.squeeze(1)).reshape([-1, 1, max_len])
        # outputs = paddle.multiply(outputs, attention_mask)
        # Weighted sum
        outputs = paddle.matmul(outputs, keys)  # [B, 1, H]

        # print("outptus_shape:", outputs.shape)
        return outputs


import os
import time
import pickle
import random

import sys
import paddle.nn.functional as F

# random.seed(1234)
# np.random.seed(1234)
# tf.set_random_seed(1234)


train_batch_size = 16
test_batch_size = 512
predict_batch_size = 32
predict_users_num = 1000
predict_ads_num = 100
hidden_units = 80

# with open('../dataset.pkl', 'rb') as f:
#   train_set = pickle.load(f)
#   test_set = pickle.load(f)
#   cate_list = pickle.load(f)
#   user_count, item_count, cate_count = pickle.load(f)

user_count = 47840
item_count = 2771
cate_count = 6

best_auc = 0.0


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d: d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None


def _auc_arr(score):
    score_p = score[:, 0]
    score_n = score[:, 1]
    # print "============== p ============="
    # print score_p
    # print "============== n ============="
    # print score_n
    score_arr = []
    for s in score_p.tolist():
        score_arr.append([0, 1, s])
    for s in score_n.tolist():
        score_arr.append([1, 0, s])
    return score_arr


# def _eval(sess, model):
#   auc_sum = 0.0
#   score_arr = []
#   for _, uij in DataInputTest(test_set, test_batch_size):
#     auc_, score_ = model.eval(sess, uij)
#     score_arr += _auc_arr(score_)
#     auc_sum += auc_ * len(uij[0])
#   test_gauc = auc_sum / len(test_set)
#   Auc = calc_auc(score_arr)
#   global best_auc
#   if best_auc < Auc:
#     best_auc = Auc
#     model.save(sess, 'save_path/ckpt')
#   return test_gauc, Auc

# def _test(sess, model):
#   auc_sum = 0.0
#   score_arr = []
#   predicted_users_num = 0
#   print "test sub items"
#   for _, uij in DataInputTest(test_set, predict_batch_size):
#     if predicted_users_num >= predict_users_num:
#         break
#     score_ = model.test(sess, uij)
#     score_arr.append(score_)
#     predicted_users_num += predict_batch_size
#   return score_[0]

model = Model(item_count, cate_count, hidden_units)

sys.stdout.flush()
lr = 0.1
start_time = time.time()
# 获得数据读取器
# data_loader = model.train_loader
# 使用adam优化器，学习率使用0.01
# opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
opt = paddle.optimizer.SGD(learning_rate=lr, parameters=model.parameters())

train_set = []
with open("../raw_data/paddle_train.txt") as f:
    for line in f:
        toks = line.strip("\n").split(";")
        hist = toks[0].split(" ")  # 商品历史点击序列
        cate = toks[1].split(" ")  # 商品历史点击对应的类别序列
        # max_len = max(max_len, len(hist))   # 序列最大长度
        click_next_item = toks[2]
        click_next_item_cate = toks[3]
        label = toks[4]
        # print(toks[2], toks[3])
        train_set.append([hist, cate, click_next_item, click_next_item_cate, float(label)])
        # res.append([hist, cate, click_next_item, click_next_item_cate, float(label)])

for epoch in range(30):

    # random.shuffle(train_set)

    epoch_size = round(len(train_set) / train_batch_size)
    # loss_sum = 0.0
    loss_sum = paddle.zeros([1])
    idx = 0
    data = np.array(train_set, dtype="object")
    batch_size = train_batch_size

    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    data_size = len(train_set)
    shuffle = True
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        ts = shuffled_data[start_index:end_index]

        hist_item, hist_cat, target_item, target_cate, hist_len, b_label = [], [], [], [], [], []

        for t in ts:
            item = t[0]
            cate = t[1]
            target_i = t[2]
            target_c = t[3]
            len_ = len(t[0])
            label = float(t[4])
            hist_item.append(item)
            hist_cat.append(cate)
            target_item.append(target_i)
            target_cate.append(target_c)
            hist_len.append(len_)
            b_label.append(label)

        max_sl = max(hist_len)
        hist_i = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[0])):
                hist_i[k][l] = t[0][l]
            k += 1

        hist_item = hist_i

        cate_i = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                cate_i[k][l] = t[1][l]
            k += 1

        hist_cate = cate_i

        ###转成paddle的tensor格式
        hist_item = paddle.to_tensor(hist_item)
        hist_cate = paddle.to_tensor(hist_cate)
        target_item = paddle.to_tensor(np.array(target_item).astype(np.int32)).reshape([-1, 1])
        target_cate = paddle.to_tensor(np.array(target_cate).astype(np.int32)).reshape([-1, 1])

        # print("hist_item", hist_item)
        # print("hist_cat", hist_cate)
        # print("target_item", target_item)
        # print("target_cate", target_cate)
        # print("hist_len", hist_len)
        # print("label", b_label)
        hist_len = paddle.to_tensor(np.array(hist_len).astype(np.int32)).reshape([-1, 1])
        b_label = paddle.to_tensor(np.array(b_label).astype(np.float32))

        p_rate = model(hist_item, hist_cate, target_item, target_cate, hist_len, b_label)
        loss = F.binary_cross_entropy(p_rate, b_label)

        # loss_sum += loss.numpy()
        loss_sum += loss
        idx += 1
        # if idx % 150 == 0:
        #     print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, idx, loss_sum / idx))

        # 损失函数下降，并清除梯度
        loss.backward()
        opt.step()
        opt.clear_grad()
        # break

    print("epoch: {}, loss is: {}".format(epoch, loss_sum.numpy() / idx))
    ###每一轮保存一个模型
    paddle.save(model.state_dict(), './checkpoint/epoch' + str(epoch) + '.pdparams')
    #     # 每个epoch 保存一次模型

    sys.stdout.flush()
    # break
