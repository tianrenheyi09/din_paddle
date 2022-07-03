import os

import paddle
from paddle.nn import Linear, Embedding
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F


import random
import math

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

        ###din_fc
        # self.din_bn = nn.BatchNorm(2*self.hidden_units)

        self.din_fc1 = nn.Linear(2*self.hidden_units, self.hidden_units)
        self.din_fc2 = nn.Linear(self.hidden_units, self.hidden_units // 2)
        self.din_fc3 = nn.Linear(self.hidden_units // 2, 1)   

        ###LOSS
        self.loss_func = paddle.nn.BCELoss()

    def forward(self, user_his, user_cate, target_item,  target_cate, sl, label):
        
        
        target_item_bias_emb = self.item_b_layer(target_item)
        target_item_bias_emb = target_item_bias_emb.reshape([-1,1])

        # print("target_item_shape:", target_item.shape)
        # print("target_cate_shape:", target_cate.shape)
        target_item_emb = self.item_emb_layer(target_item) ###batch*1*hid/2
        target_cate_emb = self.cate_emb_layer(target_cate) ###batch*1*hid/2
        # print("taegrt_item_meb_shape:", target_item_emb.shape)

        i_emb = paddle.concat([target_item_emb, target_cate_emb], axis=-1)
        i_emb = paddle.squeeze(i_emb, axis =1)
        # print("i_emb_shape:", i_emb.shape)
        ###history
        his_item_emb = self.item_emb_layer(user_his)
        his_cate_emb = self.cate_emb_layer(user_cate)

        his_emb = paddle.concat([his_item_emb, his_cate_emb], axis=-1)##batch*max_len*hidden

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
        concat_vec = paddle.concat([querys, keys, querys - keys, querys * keys], axis = 2)
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
        aa  = paddle.expand(keys_length, [keys_length.shape[0], max_len])
        cc = np.arange(max_len).reshape([1,-1])
        bb = paddle.expand(paddle.to_tensor(cc), [keys_length.shape[0], max_len])
        attention_mask = (aa > bb)
        attention_mask = attention_mask.reshape([keys_length.shape[0], 1, max_len])
        # print("attention_mask_shape:", attention_mask.shape)


        paddings = paddle.ones(outputs.shape) * (-2 ** 32 + 1)

        #
        outputs =  paddle.where(attention_mask, outputs, paddings)  # [B, 1, T]
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
    arr = sorted(raw_arr, key=lambda d:d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
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
  score_p = score[:,0]
  score_n = score[:,1]
  #print "============== p ============="
  #print score_p
  #print "============== n ============="
  #print score_n
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
        hist = toks[0].split(" ")   # 商品历史点击序列
        cate = toks[1].split(" ")   # 商品历史点击对应的类别序列
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

    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
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
        target_item = paddle.to_tensor(np.array(target_item).astype(np.int32)).reshape([-1,1])
        target_cate = paddle.to_tensor(np.array(target_cate).astype(np.int32)).reshape([-1,1])

        # print("hist_item", hist_item)
        # print("hist_cat", hist_cate)
        # print("target_item", target_item)
        # print("target_cate", target_cate)
        # print("hist_len", hist_len)
        # print("label", b_label)
        hist_len = paddle.to_tensor(np.array(hist_len).astype(np.int32)).reshape([-1,1])
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
    paddle.save(model.state_dict(), './checkpoint/epoch'+str(epoch)+'.pdparams')
    #     # 每个epoch 保存一次模型

    sys.stdout.flush()
    # break

