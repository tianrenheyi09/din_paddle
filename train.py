#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys
import logging
import time
import numpy as np
import argparse
import paddle.fluid as fluid
import paddle
import time
import network
import reader
import random

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
    user_count, item_count, cat_count = reader.config_read(config_path)
    data_reader, max_len = reader.prepare_reader(train_path, args.batch_size *
                                                 args.num_devices)
    logger.info("reading data completes")
    print("max_len: item_count, cat_count", (max_len, item_count, cat_count))
    avg_cost, pred = network.network(item_count, cat_count, max_len)
    fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(
        clip_norm=5.0))
    base_lr = args.base_lr
    boundaries = [410000]
    values = [base_lr, 0.2]
    # sgd_optimizer = fluid.optimizer.SGD(
    #     learning_rate=fluid.layers.piecewise_decay(
    #         boundaries=boundaries, values=values))

    # sgd_optimizer = fluid.optimizer.Adam(learning_rate=0.1)
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)
    sgd_optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    feeder = fluid.DataFeeder(
        feed_list=[
            "hist_item_seq", "hist_cat_seq", "target_item", "target_cat",
            "label", "mask", "target_item_seq", "target_cat_seq"
        ],
        place=place)
    if use_parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=use_cuda, loss_name=avg_cost.name)
    else:
        train_exe = exe

    logger.info("train begins")

    global_step = 0
    PRINT_STEP = 1000

    total_time = []
    ce_info = []
    start_time = time.time()
    loss_sum = 0.0

    train_set = []
    max_len = 0
    with open("../raw_data/paddle_train.txt") as f:
        for line in f:
            toks = line.strip("\n").split(";")
            hist = toks[0].split(" ")  # 商品历史点击序列
            cate = toks[1].split(" ")  # 商品历史点击对应的类别序列
            max_len = max(max_len, len(hist))  # 序列最大长度
            click_next_item = toks[2]
            click_next_item_cate = toks[3]
            label = toks[4]
            # print(toks[2], toks[3])
            train_set.append([hist, cate, click_next_item, click_next_item_cate, float(label)])
            # res.append([hist, cate, click_next_item, click_next_item_cate, float(label)])

    for id in range(epoch_num):
        epoch = id + 1
        train_batch_size = 16
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

            global_step += 1

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
            target_item = np.array(target_item).astype(np.int64).reshape([-1, 1, 1])
            target_cate = np.array(target_cate).astype(np.int64).reshape([-1, 1, 1])

            mask = np.array(
                [[0] * x + [-1e9] * (max_len - x) for x in hist_len]).reshape(
                [-1, max_len, 1]).astype(np.float32)
            target_item_seq = target_item.repeat(max_len, axis=1).astype(np.int64)
            target_cat_seq = target_cate.repeat(max_len, axis=1).astype(np.int64)
            hist_len = np.array(hist_len).astype(np.int64).reshape([-1, 1])
            b_label = np.array(b_label).astype(np.float32).reshape([-1, 1])

            target_item = target_item.reshape([-1, 1])
            target_cate = target_cate.reshape([-1, 1])
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
            # results = train_exe.run(
            #     fluid.default_main_program(),
            #     feed={
            #         "hist_item_seq":hist_item,
            #         "hist_cat_seq":hist_cate,
            #         "target_item":target_item,
            #         "target_cat":target_cate,
            #         "hist_len":hist_len,
            #         "label":b_label,
            #     },
            #     fetch_list=[avg_cost.name, pred.name],
            #     return_numpy=True)

            results = train_exe.run(
                fluid.default_main_program(),
                feed={
                    "hist_item_seq": hist_item,
                    "hist_cat_seq": hist_cate,
                    "target_item": target_item,
                    "target_cat": target_cate,
                    "label": b_label,
                    "mask": mask,
                    "target_item_seq": target_item_seq,
                    "target_cat_seq": target_cat_seq
                },
                fetch_list=[avg_cost.name, pred.name],
                return_numpy=True)

            loss_sum += results[0].mean()
            idx += 1

        # for data in data_reader():
        #     global_step += 1
        #     # print("data_shape", len(data))
        #     # print("data_####", data)
        #     # for i in range(len(data)):
        #     #     print("data_"+str(i), data[i])
        #     results = train_exe.run(feed=feeder.feed(data),
        #                             fetch_list=[avg_cost.name, pred.name],
        #                             return_numpy=True)
        #     loss_sum += results[0].mean()

            # if global_step % PRINT_STEP == 0:
            #     ce_info.append(loss_sum / PRINT_STEP)
            #     total_time.append(time.time() - start_time)
            #     logger.info(
            #         "epoch: %d\tglobal_step: %d\ttrain_loss: %.4f\t\ttime: %.2f"
            #         % (epoch, global_step, loss_sum / PRINT_STEP,
            #            time.time() - start_time))
            #     start_time = time.time()
            #     loss_sum = 0.0
            #
            #     if (global_step > 400000 and global_step % PRINT_STEP == 0) or (
            #             global_step <= 400000 and global_step % 50000 == 0):
            #         save_dir = args.model_dir + "/global_step_" + str(
            #             global_step)
            #         feed_var_name = [
            #             "hist_item_seq", "hist_cat_seq", "target_item",
            #             "target_cat", "label", "mask", "target_item_seq",
            #             "target_cat_seq"
            #         ]
            #         fetch_vars = [avg_cost, pred]
            #         fluid.io.save_inference_model(save_dir, feed_var_name,
            #                                       fetch_vars, exe)
            #         logger.info("model saved in " + save_dir)
            #
            # # break
            # if args.enable_ce and global_step >= args.batch_num:
            #     break
        logger.info(
            "epoch: %d\tglobal_step: %d\ttrain_loss: %.4f\t\ttime: %.2f"
            % (epoch, global_step, loss_sum / idx,
               time.time() - start_time))
        start_time = time.time()
        loss_sum = 0.0
    # only for ce
    #     break
    # if args.enable_ce:
    #     gpu_num = get_cards(args)
    #     ce_loss = 0
    #     ce_time = 0
    #     try:
    #         ce_loss = ce_info[-1]
    #         ce_time = total_time[-1]
    #     except:
    #         print("ce info error")
    #     print("kpis\teach_pass_duration_card%s\t%s" %
    #                 (gpu_num, ce_time))
    #     print("kpis\ttrain_loss_card%s\t%s" %
    #                 (gpu_num, ce_loss))


def get_cards(args):
    if args.enable_ce:
        cards = os.environ.get('CUDA_VISIBLE_DEVICES')
        num = len(cards.split(","))
        return num
    else:
        return args.num_devices


if __name__ == "__main__":
    train()
