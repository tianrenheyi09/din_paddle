import os


import numpy as np
import torch
import torch.nn as nn
import random
import math
# user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_

class Model(nn.Module):
    def __init__(self, item_count, cate_count, hidden_units):
        super(Model, self).__init__()
        self.item_count = item_count
        self.cate_count = cate_count
       
        self.hidden_units = hidden_units
       
        self.item_emb_layer = nn.Embedding(self.item_count, self.hidden_units // 2)
        self.cate_emb_layer = nn.Embedding(self.cate_count, self.hidden_units // 2)
        
        self.item_b_layer = nn.Embedding(self.item_count, 1)
        # self.embed.weight.data.normal_(0., 0.0001)
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
        
        self.loss_func = nn.BCELoss()

    def forward(self, user_his, user_cate, target_item,  target_cate, sl, label):
        
        target_item_bias_emb = self.item_b_layer(target_item)
        target_item_bias_emb = target_item_bias_emb.reshape([-1,1])

        # print("target_item_shape:", target_item.shape)
        # print("target_cate_shape:", target_cate.shape)
        target_item_emb = self.item_emb_layer(target_item) ###batch*1*hid/2
        target_cate_emb = self.cate_emb_layer(target_cate) ###batch*1*hid/2
        # print("taegrt_item_meb_shape:", target_item_emb.shape)

        i_emb = torch.cat([target_item_emb, target_cate_emb], dim=-1)
        i_emb = torch.squeeze(i_emb, 1)
        # print("i_emb_shape:", i_emb.shape)
        ###history
        his_item_emb = self.item_emb_layer(user_his)
        his_cate_emb = self.cate_emb_layer(user_cate)

        his_emb = torch.cat([his_item_emb, his_cate_emb], -1)##batch*max_len*hidden

        din_attention = self.attention(i_emb, his_emb, sl)
        din_attention = din_attention.reshape([-1, self.hidden_units])

        out_fc = self.out_fc(din_attention)
        embedding_concat = torch.cat([out_fc, i_emb], 1)
        # print("embedding_concat_shape", embedding_concat.shape)
        fc1 = self.sig(self.din_fc1(embedding_concat))
        fc2 = self.sig(self.din_fc2(fc1))
        fc3 = self.din_fc3(fc2)


        logits = fc3 + target_item_bias_emb
        input_logits = self.sig(logits).squeeze(axis=-1)
        loss = self.loss_func(input_logits, label)
        ave_loss = torch.mean(loss)
        return ave_loss, input_logits


    def attention(self, querys, keys, keys_length):
        """desc"""
        """
        keys_len batch*1
        
        """
        max_len = keys.shape[1]
        querys = querys.reshape([querys.shape[0], 1, querys.shape[-1]])
        # user_behavior_len = user_behavior.size(1)
        # querys = torch.cat([querys for _ in range(max_len)], dim=1)
        # print("querys_shape:", querys.shape)
        # print("keys_shape:", keys.shape)

        querys = querys.repeat(1, max_len, 1)
        # querys = paddle.tile(querys, [1, keys.shape[1]])
        # querys = querys.reshape([-1, keys.shape[1], self.hidden_units])
        # queries_hidden_units = querys.shape[-1]
        # print("querys_shape:", querys.shape)
        # # print("keys_shape:", keys.shape)
        concat_vec = torch.cat([querys, keys, querys - keys, querys * keys], 2)
        # ###
        # print("concat_vec.shape", concat_vec.shape)

        d_layer_1_all = self.sig(self.atten_fc1(concat_vec))
        d_layer_2_all = self.sig(self.atten_fc2(d_layer_1_all))
        d_layer_3_all = self.sig(self.atten_fc3(d_layer_2_all))
        outputs = d_layer_3_all.reshape([-1, 1, keys.shape[1]])



        # attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
        # print(attention_score.size())

        # define mask by length
        # user_behavior_length = user_behavior_length.type(torch.LongTensor)
        mask = torch.arange(max_len)[None, :] < keys_length[:, None]
        # print("mask_shape", mask.shape)
        # print("outouts.shape", outputs.shape)
        # mask
        # print("outputs_shape", outputs.shape)
        # print("mask_shape", mask.shape)
        output = torch.mul(outputs, mask)  # batch_size *

        ####测试缩小
        # output = output / (keys.shape[-1] ** 0.5)
        #
        # # Activation
        # output = self.soft(output)  # [B, 1, T]

        # print("keys.shape", keys.shape)
        # print("output.shape", output.shape)
        # multiply weight
        output = torch.matmul(output, keys)
        # print("outptus_shape:", output.shape)

        return output




import os
import time
import pickle
import random
import numpy as np
import sys
from input import DataInput, DataInputTest


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# random.seed(1234)
# np.random.seed(1234)
# tf.set_random_seed(1234)


train_batch_size = 32
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


def val(model, data_set):
    """
    计算模型在验证集上的信息
    """
    model.eval()########固定

    acc_val = 0
    total = 0
    loss_val = 0
    val_iteration = 0
    loss_sum = 0.0
    idx = 0


    data = np.array(data_set, dtype="object")
    batch_size = test_batch_size

    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    data_size = len(data_set)

    shuffled_data = data

    score = []
    index = 0
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
        hist_item = torch.from_numpy(hist_item).type(torch.LongTensor)
        hist_cate = torch.from_numpy(hist_cate).type(torch.LongTensor)
        target_item = torch.from_numpy(np.array(target_item).astype(np.int32)).reshape([-1, 1]).type(torch.LongTensor)
        target_cate = torch.from_numpy(np.array(target_cate).astype(np.int32)).reshape([-1, 1]).type(torch.LongTensor)

        hist_len = torch.from_numpy(np.array(hist_len).astype(np.int32)).reshape([-1, 1]).type(torch.LongTensor)
        b_label = torch.from_numpy(np.array(b_label).astype(np.float32))

        loss, p_rate = model(hist_item, hist_cate, target_item, target_cate, hist_len, b_label)
        loss_val += loss.item()

        index += 1

        for i in range(len(b_label)):
            if b_label[i] > 0.5:
                score.append([0, 1, p_rate[i]])
            else:
                score.append([1, 0, p_rate[i]])

    ###计算asuc
    auc = calc_auc(score)
    print("test--->auc:{}, val_loss:{}".format(auc, float(loss_val)/index))

    model.train()  ####重启




model = Model(item_count, cate_count, hidden_units)


sys.stdout.flush()
lr = 0.1
start_time = time.time()
 # 获得数据读取器



optim = torch.optim.SGD(model.parameters(), lr = 0.1) # 定义优化器

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


test_set = []

with open("../raw_data/paddle_test.txt") as f:
    for line in f:
        toks = line.strip("\n").split(";")
        hist = toks[0].split(" ")   # 商品历史点击序列
        cate = toks[1].split(" ")   # 商品历史点击对应的类别序列
        # max_len = max(max_len, len(hist))   # 序列最大长度
        click_next_item = toks[2]
        click_next_item_cate = toks[3]
        label = toks[4]
        test_set.append([hist, cate, click_next_item, click_next_item_cate, float(label)])


for epoch in range(30):

    # random.shuffle(train_set)
    
    epoch_size = round(len(train_set) / train_batch_size)
    loss_sum = 0.0
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

            # if len(hist_item) == self.batch_size:
            #     max_len = max(hist_len)
            #     hist_item = pad_sequences(hist_item, max_len, padding="post")
            #     hist_cat = pad_sequences(hist_cat, max_len, padding="post")

            #     yield [np.array(hist_item), np.array(hist_cat), np.array(target_item), np.array(target_cate), np.array(hist_len), np.array(b_label)], None

            #     hist_item, hist_cat, target_item, target_cate, hist_len, b_label = [], [], [], [], [], []
        # print(user_his.shape)
        # target_item = [paddle.to_tensor(var).astype('int64') for var in uij[:1]]
        # sl = [paddle.to_tensor(var).astype('int64') for var in uij[:4]]
        # label = [paddle.to_tensor(var).astype('int64') for var in uij[:2]]
        ###转成paddle的tensor格式
        hist_item = torch.from_numpy(hist_item).type(torch.LongTensor)
        hist_cate = torch.from_numpy(hist_cate).type(torch.LongTensor)
        target_item = torch.from_numpy(np.array(target_item).astype(np.int32)).reshape([-1,1]).type(torch.LongTensor)
        target_cate = torch.from_numpy(np.array(target_cate).astype(np.int32)).reshape([-1,1]).type(torch.LongTensor)

        # print("hist_item", hist_item)
        # print("hist_cat", hist_cate)
        # print("target_item", target_item)
        # print("target_cate", target_cate)
        # print("hist_len", hist_len)
        # print("label", b_label)
        hist_len = torch.from_numpy(np.array(hist_len).astype(np.int32)).reshape([-1,1]).type(torch.LongTensor)
        b_label = torch.from_numpy(np.array(b_label).astype(np.float32))


        loss, p_rate = model(hist_item, hist_cate, target_item, target_cate, hist_len, b_label)

        # loss_sum += loss.detach().numpy()
        loss_sum += loss.item()
        idx += 1
        # if idx % 150 == 0:
        #     print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, idx, loss_sum / idx))
                
        # 损失函数下降，并清除梯度
        loss.backward()
        optim.step()  # 更新权重，即向梯度方向走一步
        optim.zero_grad()  # 清空梯度

        # break
    print("epoch: {}, loss is: {}".format(epoch, loss_sum / idx))

    val(model, test_set)
    ###每一轮保存一个模型
    # paddle.save(model.state_dict(), './checkpoint/epoch'+str(epoch)+'.pdparams')
    #     # 每个epoch 保存一次模型
    
    # test_gauc, Auc = _eval(sess, model)
    # print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
    #         (model.global_epoch_step.eval(), model.global_step.eval(),
    #         loss_sum / 1000, test_gauc, Auc))
    # sys.stdout.flush()
    # loss_sum = 0.0

    # print('best test_auc:', best_auc)
    sys.stdout.flush()
    # break

