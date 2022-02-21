# coding=utf-8
# @Time     : 2021/10/25 20:34
# @Auther   : cristoli@tencent.com
import sys
sys.path.append('..')
from abc import ABC
import json
import os

from pkgs.tf.feature_util import ModelInputConfig
from models.ple import PLEModel
from models.data_helper import create_dataset
import tensorflow as tf
import os
import json
import argparse
from collections import OrderedDict
from pkgs.tf.helperfuncs import create_loss


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
    arg_parser.add_argument('--custom-path', type=str, required=True, help="定制路径")
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
    print("train_custom_path={}".format(args.custom_path))

    # train_ds = create_train_dataset(args.train_file, global_batch_size, args.field_delim,
    #                                 ctxt.is_multi_worker())

    ordered_task_names =  ["clk_label", "play_time"]
    layer_number = 1 #cgc层数
    ple_dict = {
        "clk_label": {
            "0": "PersonalRadioExpertV1",
            "1": "PersonalRadioExpertV1"
        },
        "play_time": {
            "0": "PersonalRadioExpertV1",
            "1": "PersonalRadioExpertV1"
        },
        "shared_experts": {
            "0": "PersonalRadioExpertV1",
            "1": "PersonalRadioExpertV1"
        }
    }
    tower_dict = {
        "clk_label": "PersonalLiveTowerV1",
        "play_time": "PersonalLiveTowerV1"
    }

    task_output_act = ["sigmoid", "none"]
    cin_layer = [2, 2]

    shallow_tower_dict = {0: [32, 1], 1: [32, 1]}



    parserDict = {"field_delim": ",", "with_headers": False, "headers": "ftime,uin,u_click_seq,u_pre_singer,u_pre_singer_weight,u_age_level,u_sex,u_degree,u_income_level,u_city_level,u_os_type,u_log_days,u_net_day,u_expnum_1d,u_clicknum_1d,u_clickzhubonum_1d,u_valid_viewcnt_1d_flow,u_valid_viewcnt_1d_unflow,u_valid_viewzhubocnt_1d_flow,u_valid_viewzhubocnt_1d_unflow,u_short_viewcnt_1d_flow,u_short_viewcnt_1d_unflow,u_short_viewzhubocnt_1d_flow,u_short_viewzhubocnt_1d_unflow,u_long_viewcnt_1d_flow,u_long_viewcnt_1d_unflow,u_long_viewzhubocnt_1d_flow,u_long_viewzhubocnt_1d_unflow,u_followcnt_1d,u_unfollow_1d,u_giftcnt_1d,u_gift_totcnt_1d,u_giftzhubocnt_1d,u_chargecnt_1d,u_chargemoney_1d,u_danmunum_1d,u_danmuzhubocnt_1d,u_expnum_7d,u_clicknum_7d,u_clickzhubonum_7d,u_valid_viewcnt_7d_flow,u_valid_viewcnt_7d_unflow,u_valid_viewzhubocnt_7d_flow,u_valid_viewzhubocnt_7d_unflow,u_short_viewcnt_7d_flow,u_short_viewcnt_7d_unflow,u_short_viewzhubocnt_7d_flow,u_short_viewzhubocnt_7d_unflow,u_long_viewcnt_7d_flow,u_long_viewcnt_7d_unflow,u_long_viewzhubocnt_7d_flow,u_long_viewzhubocnt_7d_unflow,u_followcnt_7d,u_unfollow_7d,u_giftcnt_7d,u_gift_totcnt_7d,u_giftzhubocnt_7d,u_chargecnt_7d,u_chargemoney_7d,u_danmunum_7d,u_danmuzhubocnt_7d,u_expnum_15d,u_clicknum_15d,u_clickzhubonum_15d,u_valid_viewcnt_15d_flow,u_valid_viewcnt_15d_unflow,u_valid_viewzhubocnt_15d_flow,u_valid_viewzhubocnt_15d_unflow,u_short_viewcnt_15d_flow,u_short_viewcnt_15d_unflow,u_short_viewzhubocnt_15d_flow,u_short_viewzhubocnt_15d_unflow,u_long_viewcnt_15d_flow,u_long_viewcnt_15d_unflow,u_long_viewzhubocnt_15d_flow,u_long_viewzhubocnt_15d_unflow,u_followcnt_15d,u_unfollow_15d,u_giftcnt_15d,u_gift_totcnt_15d,u_giftzhubocnt_15d,u_chargecnt_15d,u_chargemoney_15d,u_danmunum_15d,u_danmuzhubocnt_15d,u_expnum_30d,u_clicknum_30d,u_clickzhubonum_30d,u_valid_viewcnt_30d_flow,u_valid_viewcnt_30d_unflow,u_valid_viewzhubocnt_30d_flow,u_valid_viewzhubocnt_30d_unflow,u_short_viewcnt_30d_flow,u_short_viewcnt_30d_unflow,u_short_viewzhubocnt_30d_flow,u_short_viewzhubocnt_30d_unflow,u_long_viewcnt_30d_flow,u_long_viewcnt_30d_unflow,u_long_viewzhubocnt_30d_flow,u_long_viewzhubocnt_30d_unflow,u_followcnt_30d,u_unfollow_30d,u_giftcnt_30d,u_gift_totcnt_30d,u_giftzhubocnt_30d,u_chargecnt_30d,u_chargemoney_30d,u_danmunum_30d,u_danmuzhubocnt_30d,u_sharecnt_1d,u_sharecnt_7d,u_sharecnt_15d,u_sharecnt_30d,anchor_id,a_age,a_sex,a_province,a_kaibo_type,a_first_launch_time,a_kaibo_period,a_kaibo_period_weight,a_kaibo_days_1d,a_kaibo_time_1d,a_kaibo_cnt_1d,a_expnum_1d,a_clicknum_1d,a_clickusernum_1d,a_valid_viewcnt_1d_flow,a_valid_viewcnt_1d_unflow,a_valid_viewuser_1d_flow,a_valid_viewuser_1d_unflow,a_short_viewcnt_1d_flow,a_short_viewcnt_1d_unflow,a_short_viewuser_1d_flow,a_short_viewuser_1d_unflow,a_long_viewcnt_1d_flow,a_long_viewcnt_1d_unflow,a_long_viewuser_1d_flow,a_long_viewuser_1d_unflow,a_followusercnt_1d,a_followcnt_1d,a_unfollowcnt_1d,a_giftcnt_1d,a_gift_totcnt_1d,a_giftuser_1d,a_danmunum_1d,a_danmuuser_1d,a_kaibo_days_7d,a_kaibo_time_7d,a_kaibo_cnt_7d,a_expnum_7d,a_clicknum_7d,a_clickusernum_7d,a_valid_viewcnt_7d_flow,a_valid_viewcnt_7d_unflow,a_valid_viewuser_7d_flow,a_valid_viewuser_7d_unflow,a_short_viewcnt_7d_flow,a_short_viewcnt_7d_unflow,a_short_viewuser_7d_flow,a_short_viewuser_7d_unflow,a_long_viewcnt_7d_flow,a_long_viewcnt_7d_unflow,a_long_viewuser_7d_flow,a_long_viewuser_7d_unflow,a_followusercnt_7d,a_followcnt_7d,a_unfollowcnt_7d,a_giftcnt_7d,a_gift_totcnt_7d,a_giftuser_7d,a_danmunum_7d,a_danmuuser_7d,a_kaibo_days_15d,a_kaibo_time_15d,a_kaibo_cnt_15d,a_expnum_15d,a_clicknum_15d,a_clickusernum_15d,a_valid_viewcnt_15d_flow,a_valid_viewcnt_15d_unflow,a_valid_viewuser_15d_flow,a_valid_viewuser_15d_unflow,a_short_viewcnt_15d_flow,a_short_viewcnt_15d_unflow,a_short_viewuser_15d_flow,a_short_viewuser_15d_unflow,a_long_viewcnt_15d_flow,a_long_viewcnt_15d_unflow,a_long_viewuser_15d_flow,a_long_viewuser_15d_unflow,a_followusercnt_15d,a_followcnt_15d,a_unfollowcnt_15d,a_giftcnt_15d,a_gift_totcnt_15d,a_giftuser_15d,a_danmunum_15d,a_danmuuser_15d,a_kaibo_days_30d,a_kaibo_time_30d,a_kaibo_cnt_30d,a_expnum_30d,a_clicknum_30d,a_clickusernum_30d,a_valid_viewcnt_30d_flow,a_valid_viewcnt_30d_unflow,a_valid_viewuser_30d_flow,a_valid_viewuser_30d_unflow,a_short_viewcnt_30d_flow,a_short_viewcnt_30d_unflow,a_short_viewuser_30d_flow,a_short_viewuser_30d_unflow,a_long_viewcnt_30d_flow,a_long_viewcnt_30d_unflow,a_long_viewuser_30d_flow,a_long_viewuser_30d_unflow,a_followusercnt_30d,a_followcnt_30d,a_unfollowcnt_30d,a_giftcnt_30d,a_gift_totcnt_30d,a_giftuser_30d,a_danmunum_30d,a_danmuuser_30d,a_sharecnt_1d,a_shareusercnt_1d,a_sharecnt_7d,a_shareusercnt_7d,a_sharecnt_15d,a_shareusercnt_15d,a_sharecnt_30d,a_shareusercnt_30d,imp_label,clk_label,play_time,u_scenario"}
    ds = create_dataset(args.train_conf_path, args.train_path, file_type='csv', batch_size=args.batch_size,  **parserDict)
    val_ds = create_dataset(args.train_conf_path, args.val_path, file_type='csv', batch_size=args.batch_size,  **parserDict)
    test_ds = create_dataset(args.test_conf_path, args.test_path, file_type='csv', batch_size=args.batch_size,  **parserDict)

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
        # dnn_l2_reg =  0.01,
        # embedding_l2_reg =  0.01

        model = PLEModel(input, ordered_task_names, layer_number, ple_dict, tower_dict, task_output_act,
                         tower_dependencies_dict={}, dropout_layer="PersonalRadioInputDropoutV1",
                         custom_layer_file_path=args.custom_path,
                         is_concat_gate_input=True, use_inputs_dropout=True,
                         use_shallow=True, shallow_tower_dict=shallow_tower_dict, name='ple')
        optimizer = tf.keras.optimizers.Adam(args.lr)

        loss1 = create_loss('binary_cross_entropy')
        loss2 = create_loss('tmse')
        auc = tf.keras.metrics.AUC()
        bce = tf.keras.metrics.BinaryCrossentropy()
        model.compile(optimizer=optimizer, loss=[loss1, loss2], loss_weights=[1, 1000],
                      metrics={'clk_label': [auc, bce], 'play_time': [
                tf.keras.metrics.MeanSquaredError()]})
        print("loss weight={}".format(100))


        # callbacks = [tf.keras.callbacks.TensorBoard(os.path.join(args.runs_path, "tblogs"))]
        # model.fit(ds, epochs=args.epochs, callbacks=callbacks, steps_per_epoch=steps_per_epoch)
        # results = model.evaluate(ds, batch_size=128)
        # model.fit(train_ds, epochs=args.epochs, callbacks=callbacks, steps_per_epoch=steps_per_epoch)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_clk_label_auc', min_delta=0.002,patience=3)]
        model.fit(ds.repeat(), epochs=args.epochs, callbacks=callbacks, steps_per_epoch=steps_per_train_epoch, validation_data=val_ds.repeat(), validation_steps=steps_per_test_epoch)
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
