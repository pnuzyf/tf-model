# coding=utf-8
# @Time     : 2021/10/25 20:34
# @Auther   : cristoli@tencent.com
import sys
sys.path.append('..')
from abc import ABC
import json
import os

from pkgs.tf.feature_util import ModelInputConfig
from models.data_helper import create_dataset
import tensorflow as tf
import os
import json
import argparse
from collections import OrderedDict


import json
import os
from pkgs.tf.feature_util import ModelInputConfig
from models.dtower import DTowerModel
from models.data_helper import create_dataset
from pkgs.tf.helperfuncs import create_loss, create_metric
import tensorflow as tf

class TFDistContext(object):
    """
    TF分布式上下文管理器，可以自适应单机，多机
    """

    __inst = None

    def __init__(self):
        self._tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
        self._dist_strategy = self._create_strategy()
        print("TFDistContext._dist_strategy={}".format(self._dist_strategy))

    def _create_strategy(self):
        if self.is_multi_worker():
            return tf.distribute.experimental.MultiWorkerMirroredStrategy()
        has_gpu = tf.test.is_gpu_available()
        if has_gpu:
            return tf.distribute.MirroredStrategy()
        return tf.distribute.get_strategy()

    @property
    def strategy(self):
        return self._dist_strategy

    def is_multi_worker(self):
        if not self._tf_config:
            return False
        cluster_info = self._tf_config.get('cluster', {})
        workers = cluster_info.get('worker', [])
        chiefs = cluster_info.get('chief', [])
        masters = cluster_info.get('master', [])

        return len(workers) + len(chiefs) + len(masters) > 1

    def get_role_in_cluster(self):
        if not self._tf_config:
            return None, None
        task_info = self._tf_config.get('task', {})
        role, index = task_info.get('type'), task_info.get('index')
        return role, index

    def is_chief(self):
        if not self._tf_config:
            return True
        role, index = self.get_role_in_cluster()
        if role in ['chief', 'master']:
            return True
        cluster_info = self._tf_config.get('cluster', {})
        if 'chief' in cluster_info or 'master' in cluster_info:
            return False
        return role == 'worker' and index == 0

    def global_batch_size(self, per_replica_batch_size):
        return self._dist_strategy.num_replicas_in_sync * per_replica_batch_size

    def normalize_save_path(self, worker_save_path):
        if self.is_chief():
            return worker_save_path
        role, index = self.get_role_in_cluster()
        return os.path.join(worker_save_path, ".{}-{}_tmp".format(role, index))

    @classmethod
    def inst(cls):
        if cls.__inst is None:
            cls.__inst = TFDistContext()
        return cls.__inst

if __name__ == '__main__':
    # 以下参数列表只是示例，实际使用时请按需自己增删改
    arg_parser = argparse.ArgumentParser("机器学习平台plain方式训练demo")
    arg_parser.add_argument('--runs-path', type=str, required=True, help="runs路径")
    arg_parser.add_argument('--pack-path', type=str, required=True, help="pack路径")
    arg_parser.add_argument('--train-conf-path', type=str, required=True, help="训练配置路径")
    arg_parser.add_argument('--test-conf-path', type=str, required=True, help="测试配置路径")
    arg_parser.add_argument('--train-path', type=str, required=True, help="训练路径")
    arg_parser.add_argument('--val-path', type=str, required=True, help="验证路径")
    arg_parser.add_argument('--test-path', type=str, required=True, help="测试路径")
    arg_parser.add_argument('--batch-size', type=int, default=1024)
    arg_parser.add_argument('--epochs', type=int, default=10)
    arg_parser.add_argument('--lr', type=float, default=0.001)
    arg_parser.add_argument('--num-train-samples', type=int, default=22000000, help="样本数，分布式方式下必传，单机模式下不用")
    arg_parser.add_argument('--num-val-samples', type=int, default=2000000, help="样本数，分布式方式下必传，单机模式下不用")
    arg_parser.add_argument('--model-name', type=str, required=True, help="模型名字")
    arg_parser.add_argument('--model-save-path', type=str, required=True, help="模型保存路径")

    args = arg_parser.parse_args()
    print("args={}".format(args))

    ctxt = TFDistContext.inst()
    global_batch_size = ctxt.global_batch_size(args.batch_size)
    print("global_batch_size={}".format(global_batch_size))
    print("train_path={}".format(args.train_path))
    print("train_conf_path={}".format(args.train_conf_path))
    # train_ds = create_train_dataset(args.train_file, global_batch_size, args.field_delim,
    #                                 ctxt.is_multi_worker())

    parserDict = {"field_delim": ",", "with_headers": False,
                  "headers": "ftime,uin, u_click_seq, u_age_level, u_sex, u_degree, u_income_level, u_city_level, u_os_type, u_log_days, u_net_day,u_expnum_30d ,u_clicknum_30d ,u_clickzhubonum_30d ,u_valid_viewcnt_30d_flow,u_valid_viewcnt_30d_unflow ,u_valid_viewzhubocnt_30d_flow ,u_valid_viewzhubocnt_30d_unflow,u_short_viewcnt_30d_flow ,u_short_viewcnt_30d_unflow ,u_short_viewzhubocnt_30d_flow,u_short_viewzhubocnt_30d_unflow  ,u_long_viewcnt_30d_flow ,u_long_viewcnt_30d_unflow,u_long_viewzhubocnt_30d_flow ,u_long_viewzhubocnt_30d_unflow ,u_followcnt_30d,u_unfollow_30d ,u_giftcnt_30d ,u_gift_totcnt_30d ,u_giftzhubocnt_30d ,u_chargecnt_30d,u_chargemoney_30d ,u_danmunum_30d ,u_danmuzhubocnt_30d,u_sharecnt_30d, anchor_id, a_age, a_sex,a_province, a_kaibo_type, watch_score, finance_score,  a_kaibo_period,a_kaibo_period_weight,imp_label, clk_label, play_time, u_scenario",
                  "fake_label": 1}

    test_parserDict = {"field_delim": ",", "with_headers": False,
                  "headers": "ftime,uin, u_click_seq, u_age_level, u_sex, u_degree, u_income_level, u_city_level, u_os_type, u_log_days, u_net_day,u_expnum_30d ,u_clicknum_30d ,u_clickzhubonum_30d ,u_valid_viewcnt_30d_flow,u_valid_viewcnt_30d_unflow ,u_valid_viewzhubocnt_30d_flow ,u_valid_viewzhubocnt_30d_unflow,u_short_viewcnt_30d_flow ,u_short_viewcnt_30d_unflow ,u_short_viewzhubocnt_30d_flow,u_short_viewzhubocnt_30d_unflow  ,u_long_viewcnt_30d_flow ,u_long_viewcnt_30d_unflow,u_long_viewzhubocnt_30d_flow ,u_long_viewzhubocnt_30d_unflow ,u_followcnt_30d,u_unfollow_30d ,u_giftcnt_30d ,u_gift_totcnt_30d ,u_giftzhubocnt_30d ,u_chargecnt_30d,u_chargemoney_30d ,u_danmunum_30d ,u_danmuzhubocnt_30d,u_sharecnt_30d, anchor_id, a_age, a_sex,a_province, a_kaibo_type, watch_score, finance_score,  a_kaibo_period,a_kaibo_period_weight,imp_label, clk_label, play_time, u_scenario"}
    ds = create_dataset(args.train_conf_path, args.train_path, file_type='csv', batch_size=args.batch_size,  **parserDict)
    val_ds = create_dataset(args.train_conf_path, args.val_path, file_type='csv', batch_size=args.batch_size,  **test_parserDict)
    test_ds = create_dataset(args.test_conf_path, args.test_path, file_type='csv', batch_size=args.batch_size,  **test_parserDict)
    # i = 0
    # for r in ds:
    #     if i >= 10:
    #         i = 0
    #         break
    #     print(r)
    #     i += 1

    steps_per_train_epoch = None
    steps_per_test_epoch = None
    if ctxt.is_multi_worker():
        if not args.num_train_samples:
            raise RuntimeError("--num-samples must be set under distributed environment")

        steps_per_train_epoch = args.num_train_samples // global_batch_size
        steps_per_test_epoch = args.num_val_samples // global_batch_size
        print("steps_per_train_epoch={}".format(steps_per_train_epoch))

    with ctxt.strategy.scope():
        # model，metric，loss，optimizer，callback都需要在scope下创建，
        # compile，fit或者自定义training loop都需要定义在scope下
        input = ModelInputConfig.parse(args.train_conf_path, args.pack_path, args.runs_path)

        user_tower_units = [
            128,
            64,
            32
        ]

        item_tower_units = [
            128,
            64,
            32
        ]

        user_tower_hidden_act = "relu"
        item_tower_hideen_act = "relu"
        user_tower_output_act = "relu"
        item_tower_output_act = "relu"

        use_cosine = True
        model_type = "constrative"
        temperature = 0.05

        model = DTowerModel(input, user_tower_units=user_tower_units, user_tower_hidden_act=user_tower_hidden_act,
                            item_tower_units=item_tower_units, item_tower_hidden_act=item_tower_hideen_act,
                            user_tower_output_act=user_tower_output_act, item_tower_output_act=item_tower_output_act,
                            model_type=model_type, use_cosine=use_cosine, temperature=temperature)

        infonce_loss = create_loss('infonce')

        # po_acc_metric = create_metric('po_acc', name_prefix='')

        softmax_acc_metric = create_metric('softmax_acc', name_prefix='')
        optimizer = tf.keras.optimizers.Adam(args.lr)
        #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_po_acc', min_delta=0.002,patience=3)]

        model.compile(optimizer=optimizer, loss=infonce_loss, metrics=softmax_acc_metric)

        model.fit(ds, epochs=args.epochs, steps_per_epoch=steps_per_train_epoch, validation_data=val_ds.repeat(), validation_steps=steps_per_test_epoch)

       # model.fit(ds, epochs=args.epochs, callbacks=callbacks, steps_per_epoch=steps_per_train_epoch, validation_data=val_ds, validation_steps=steps_per_test_epoch)
        eval_result = model.evaluate(test_ds, batch_size=1024, steps=steps_per_test_epoch)

        eval_result_path = os.path.join(args.runs_path, "model_eval_result.json")

        eval_str = {}
        eval_map = {}
        for i in range(len(model.metrics_names)):
            eval_map[model.metrics_names[i]] = eval_result[i]
        eval_list = [args.model_name]
        eval_list.append(args.model_save_path)
        eval_list.append(eval_map)
        eval_str['eval_results'] = [eval_list]

        with open(eval_result_path, 'w') as fw:
            json.dump(eval_str, fw)
    model_save_path = os.path.join(args.runs_path, 'saved_model')
    save_path = ctxt.normalize_save_path(model_save_path)
    print("save_path='{}', is_chief={}".format(save_path, ctxt.is_chief()))
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
        print("created model saving dir '{}'".format(save_path))
    model.save(save_path)
