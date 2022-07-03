import os
import paddle
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import random
import math


import os
import time
import pickle
import random
import numpy as np

import sys

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

import logging
import networkMy
import network
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser("din")
    parser.add_argument(
        '--config_path', type=str, default='../raw_data/config.txt', help='dir of config')
    parser.add_argument(
        '--train_dir', type=str, default='../raw_data/paddle_train.txt', help='dir of train file')
    parser.add_argument(
        '--model_dir', type=str, default='din_amazon', help='dir of saved model')
    parser.add_argument(
        '--batch_size', type=int, default=16, help='number of batch size')
    parser.add_argument(
        '--epoch_num', type=int, default=200, help='number of epoch')
    parser.add_argument(
        '--use_cuda', type=int, default=0, help='whether to use gpu')
    parser.add_argument(
        '--parallel', type=int, default=0, help='whether to use parallel executor')
    parser.add_argument(
        '--base_lr', type=float, default=0.85, help='based learning rate')
    parser.add_argument(
        '--num_devices', type=int, default=1, help='Number of GPU devices')
    parser.add_argument(
        '--enable_ce', action='store_true', help='If set, run the task with continuous evaluation logs.')
    parser.add_argument(
        '--batch_num', type=int, help="batch num for ce")
    args = parser.parse_args()
    return args

def train():
    args = parse_args()

    if args.enable_ce:
        SEED = 102
        fluid.default_main_program().random_seed = SEED
        fluid.default_startup_program().random_seed = SEED

    config_path = args.config_path
    train_path = args.train_dir
    epoch_num = args.epoch_num
    use_cuda = True if args.use_cuda else False
    use_parallel = True if args.parallel else False


    logger.info("reading data begins")
    user_count = 47840
    item_count = 2771
    cate_count = 6
    hidden_units = 80

    train_set = []
    max_len = 0
    with open("../raw_data/paddle_train.txt") as f:
        for line in f:
            toks = line.strip("\n").split(";")
            hist = toks[0].split(" ")  # 商品历史点击序列
            cate = toks[1].split(" ")  # 商品历史点击对应的类别序列
            max_len = max(max_len, len(hist))   # 序列最大长度
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
            hist = toks[0].split(" ")  # 商品历史点击序列
            cate = toks[1].split(" ")  # 商品历史点击对应的类别序列
            # max_len = max(max_len, len(hist))   # 序列最大长度
            click_next_item = toks[2]
            click_next_item_cate = toks[3]
            label = toks[4]
            test_set.append([hist, cate, click_next_item, click_next_item_cate, float(label)])

    # data_reader, max_len = reader.prepare_reader(train_path, args.batch_size *
    #                                              args.num_devices)
    logger.info("reading data completes")
    # print("max_len: item_count, cat_count", (max_len, item_count, cat_count))

    avg_cost, pred = networkMy.networkMy(item_count, cate_count, max_len, hidden_units)
    # avg_cost, pred = network.network(item_count, cate_count, max_len)


    # avg_cost, pred = network.network(item_count, cat_count, max_len)
    fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(
        clip_norm=5.0))
    base_lr = 0.1
    boundaries = [410000]
    values = [base_lr, 0.2]
    # sgd_optimizer = fluid.optimizer.SGD(
    #     learning_rate=fluid.layers.piecewise_decay(
    #         boundaries=boundaries, values=values))
    # sgd_optimizer.minimize(avg_cost)


    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)

    sgd_optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    test_program = fluid.default_main_program().clone(for_test=True)
    # feeder = fluid.DataFeeder(
    #     feed_list=[
    #         "hist_item_seq", "hist_cat_seq", "target_item", "target_cat",
    #         "label"
    #     ],
    #     place=place)
    if use_parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=use_cuda, loss_name=avg_cost.name)
    else:
        train_exe = exe

    ####读取数据

    logger.info("train begins")

    global_step = 0
    PRINT_STEP = 1000

    total_time = []
    ce_info = []
    start_time = time.time()

    for id in range(30):
        epoch = id + 1
        epoch_size = round(len(train_set) / train_batch_size)
        loss_sum = 0.0
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
            max_sl = max_len
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
            # hist_item = paddle.to_tensor(hist_item)
            # hist_cate = paddle.to_tensor(hist_cate)
            hist_item = hist_item.reshape([-1, max_len, 1]).astype(np.int64)
            hist_cate = hist_cate.reshape([-1, max_len, 1]).astype(np.int64)
            target_item =np.array(target_item).astype(np.int64).reshape([-1, 1, 1])
            target_cate = np.array(target_cate).astype(np.int64).reshape([-1,1,  1])

            mask = np.array(
                [[0] * x + [-1e9] * (max_len - x) for x in hist_len]).reshape(
                [-1, max_len, 1]).astype(np.float32)
            target_item_seq = target_item.repeat(max_len, axis=1).astype(np.int64)
            target_cat_seq = target_cate.repeat(max_len, axis=1).astype(np.int64)
            hist_len = np.array(hist_len).astype(np.int64).reshape([-1, 1])
            b_label = np.array(b_label).astype(np.float32).reshape([-1,1])

            # target_item = target_item.reshape([-1,1])
            # target_cate = target_cate.reshape([-1,1])


            # print("hist_item", hist_item.shape)
            # print("hist_cat", hist_cate.shape)
            # print("target_item", target_item.shape)
            # print("target_cate", target_cate.shape)
            # print("hist_len", hist_len.shape)
            # print("label", b_label.shape)
            # print("target_item_seq_shape", target_item_seq.shape)
            # print("target_cate_seq_shape", target_cat_seq.shape)
            #
            # print("mask", mask.shape)
            # print("mask", mask)
            results = train_exe.run(
                program=fluid.default_main_program(),
                feed={
                    "hist_item_seq":hist_item,
                    "hist_cat_seq":hist_cate,
                    "target_item":target_item,
                    "target_cat":target_cate,
                    "hist_len":hist_len,
                    "label":b_label,
                },
                fetch_list=[avg_cost.name, pred.name],
                return_numpy=True)

            loss_sum += results[0].mean()
            idx += 1



        logger.info(
            "epoch: %d\tglobal_step: %d\ttrain_loss: %.4f\t\ttime: %.2f"
            % (epoch, global_step, loss_sum / idx,
               time.time() - start_time))
        start_time = time.time()

        #####每一轮都拿验证集合进行测试

        data = np.array(test_set, dtype="object")
        batch_size = test_batch_size

        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        data_size = len(test_set)

        shuffled_data = data
        loss_sum_test = 0

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            ts = shuffled_data[start_index:end_index]

            hist_item, hist_cat, target_item, target_cate, hist_len, b_label = [], [], [], [], [], []

            max_sl = 0
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
                # max_sl = max(max_sl, len_)


            max_sl = max_len
            hist_i = np.zeros([len(ts), max_sl], np.int64)
            k = 0
            for t in ts:
                for l in range(min(len(t[0]), max_sl)):
                    hist_i[k][l] = t[0][l]
                k += 1

            hist_item = hist_i

            cate_i = np.zeros([len(ts), max_sl], np.int64)
            k = 0
            for t in ts:
                for l in range(min(len(t[1]), max_sl)):
                    cate_i[k][l] = t[1][l]
                k += 1

            hist_cate = cate_i

            ###转成paddle的tensor格式
            hist_item = hist_item.reshape([-1, max_len, 1]).astype(np.int64)
            hist_cate = hist_cate.reshape([-1, max_len, 1]).astype(np.int64)
            target_item = np.array(target_item).astype(np.int64).reshape([-1, 1, 1])
            target_cate = np.array(target_cate).astype(np.int64).reshape([-1, 1, 1])


            hist_len = np.array(hist_len).astype(np.int64).reshape([-1, 1])
            b_label = np.array(b_label).astype(np.float32).reshape([-1, 1])


            results = train_exe.run(
                program=test_program,
                feed={
                    "hist_item_seq": hist_item,
                    "hist_cat_seq": hist_cate,
                    "target_item": target_item,
                    "target_cat": target_cate,
                    "hist_len": hist_len,
                    "label": b_label,
                },
                fetch_list=[avg_cost.name, pred.name],
                return_numpy=True)

            loss_sum_test += results[0].mean()

        logger.info(
            "epoch: %d\tglobal_step: %d\ttest_loss: %.4f\t\ttime: %.2f"
            % (epoch, global_step, loss_sum_test / num_batches_per_epoch,
               time.time() - start_time))

        # save_dir ="/global_step_" + str(
        #     global_step)
        # feed_var_name = [
        #     "hist_item_seq", "hist_cat_seq", "target_item",
        #     "target_cat", "hist_len", "label"
        # ]
        # fetch_vars = [avg_cost, pred]
        # fluid.io.save_inference_model(save_dir, feed_var_name,
        #                               fetch_vars, exe)
        # logger.info("model saved in " + save_dir)

def get_cards(args):
    if args.enable_ce:
        cards = os.environ.get('CUDA_VISIBLE_DEVICES')
        num = len(cards.split(","))
        return num
    else:
        return args.num_devices


if __name__ == '__main__':
    train()