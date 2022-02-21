import tensorflow
from tensorflow.keras.utils import plot_model

import json
import os
from pkgs.tf.feature_util import ModelInputConfig
from models.msmt import MsMtModel
from models.data_helper import create_dataset
from pkgs.tf.helperfuncs import create_loss
import tensorflow as tf
from tensorflow.keras.utils import plot_model


if __name__ == "__main__":
    import tensorflow as tf

    # tf.config.run_functions_eagerly(True)
    config_file = '/Users/fabriszhou/PycharmProjects/tf-model/test/msmt/msmt_memory.json'
    export_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/msmt/runs'
    pack_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/msmt'
    data_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/msmt/part-00007'


    input = ModelInputConfig.parse(config_file, pack_path, export_path)

    ordered_task_names =  ["clk_label", "play_time"]
    layer_number = 1 #cgc层数
    ple_dict = {
        "clk_label": {
            "0": "PersonalLiveExpertV1",
            "1": "PersonalLiveExpertV1"
        },
        "play_time": {
            "0": "PersonalLiveExpertV1",
            "1": "PersonalLiveExpertV1"
        },
        "shared_experts": {
            "0": "PersonalLiveExpertV1",
            "1": "PersonalLiveExpertV1"
        }
    }
    tower_dict = {
        "clk_label": "PersonalLiveTowerV1",
        "play_time": "PersonalLiveTowerV1"
    }

    task_output_act = ["sigmoid", "none"]
    num_scenarios = 3
    moe_num_experts = 3
    moe_expert_layers = [128, 64]

    cut_tower_dict = [1024, 512]

    is_memory = True
    controller_layers = [256, 64]
    controller_output_act = 'tanh'
    n_clusters = 100
    key_dims = 64
    long_dims = 32
    temperature = 0.1
    alpha = 0.1

    model = MsMtModel(input, num_scenarios, moe_num_experts, moe_expert_layers, ordered_task_names, layer_number, ple_dict, tower_dict, task_output_act,
                     tower_dependencies_dict={},
                     custom_layer_file_path="/Users/fabriszhou/PycharmProjects/tf-model/test/msmt/custom/custom_layers.py", is_parallel=False, short_cut=True,
                     cut_tower=cut_tower_dict, is_memory=True, controller_layers=controller_layers, controller_output_act=controller_output_act, key_dims=key_dims,
                     long_dims=long_dims,
                     name='msmt')

    #parserDict = {"field_delim": ",", "with_headers": False,"headers":"ftime,uin,u_click_seq,u_age_level,u_log_days,u_clicknum_1d,anchor_id,a_age,a_kaibo_period, a_kaibo_period_weight,a_clicknum_1d,clk_label,play_time"}

    parserDict = {"field_delim": ",", "with_headers": False,
                  "headers": "ftime,uin,u_click_seq_new,u_pre_singer,u_pre_singer_weight,u_age_level,u_sex,u_degree,u_income_level,u_city_level,u_os_type,u_log_days,u_net_day,u_expnum_1d,u_clicknum_1d,u_clickzhubonum_1d,u_valid_viewcnt_1d_flow,u_valid_viewcnt_1d_unflow,u_valid_viewzhubocnt_1d_flow,u_valid_viewzhubocnt_1d_unflow,u_short_viewcnt_1d_flow,u_short_viewcnt_1d_unflow,u_short_viewzhubocnt_1d_flow,u_short_viewzhubocnt_1d_unflow,u_long_viewcnt_1d_flow,u_long_viewcnt_1d_unflow,u_long_viewzhubocnt_1d_flow,u_long_viewzhubocnt_1d_unflow,u_followcnt_1d,u_unfollow_1d,u_giftcnt_1d,u_gift_totcnt_1d,u_giftzhubocnt_1d,u_chargecnt_1d,u_chargemoney_1d,u_danmunum_1d,u_danmuzhubocnt_1d,u_expnum_7d,u_clicknum_7d,u_clickzhubonum_7d,u_valid_viewcnt_7d_flow,u_valid_viewcnt_7d_unflow,u_valid_viewzhubocnt_7d_flow,u_valid_viewzhubocnt_7d_unflow,u_short_viewcnt_7d_flow,u_short_viewcnt_7d_unflow,u_short_viewzhubocnt_7d_flow,u_short_viewzhubocnt_7d_unflow,u_long_viewcnt_7d_flow,u_long_viewcnt_7d_unflow,u_long_viewzhubocnt_7d_flow,u_long_viewzhubocnt_7d_unflow,u_followcnt_7d,u_unfollow_7d,u_giftcnt_7d,u_gift_totcnt_7d,u_giftzhubocnt_7d,u_chargecnt_7d,u_chargemoney_7d,u_danmunum_7d,u_danmuzhubocnt_7d,u_expnum_15d,u_clicknum_15d,u_clickzhubonum_15d,u_valid_viewcnt_15d_flow,u_valid_viewcnt_15d_unflow,u_valid_viewzhubocnt_15d_flow,u_valid_viewzhubocnt_15d_unflow,u_short_viewcnt_15d_flow,u_short_viewcnt_15d_unflow,u_short_viewzhubocnt_15d_flow,u_short_viewzhubocnt_15d_unflow,u_long_viewcnt_15d_flow,u_long_viewcnt_15d_unflow,u_long_viewzhubocnt_15d_flow,u_long_viewzhubocnt_15d_unflow,u_followcnt_15d,u_unfollow_15d,u_giftcnt_15d,u_gift_totcnt_15d,u_giftzhubocnt_15d,u_chargecnt_15d,u_chargemoney_15d,u_danmunum_15d,u_danmuzhubocnt_15d,u_expnum_30d,u_clicknum_30d,u_clickzhubonum_30d,u_valid_viewcnt_30d_flow,u_valid_viewcnt_30d_unflow,u_valid_viewzhubocnt_30d_flow,u_valid_viewzhubocnt_30d_unflow,u_short_viewcnt_30d_flow,u_short_viewcnt_30d_unflow,u_short_viewzhubocnt_30d_flow,u_short_viewzhubocnt_30d_unflow,u_long_viewcnt_30d_flow,u_long_viewcnt_30d_unflow,u_long_viewzhubocnt_30d_flow,u_long_viewzhubocnt_30d_unflow,u_followcnt_30d,u_unfollow_30d,u_giftcnt_30d,u_gift_totcnt_30d,u_giftzhubocnt_30d,u_chargecnt_30d,u_chargemoney_30d,u_danmunum_30d,u_danmuzhubocnt_30d,u_sharecnt_1d,u_sharecnt_7d,u_sharecnt_15d,u_sharecnt_30d,anchor_id,a_age,a_sex,a_province,a_kaibo_type,a_first_launch_time,a_kaibo_period,a_kaibo_period_weight,a_kaibo_days_1d,a_kaibo_time_1d,a_kaibo_cnt_1d,a_expnum_1d,a_clicknum_1d,a_clickusernum_1d,a_valid_viewcnt_1d_flow,a_valid_viewcnt_1d_unflow,a_valid_viewuser_1d_flow,a_valid_viewuser_1d_unflow,a_short_viewcnt_1d_flow,a_short_viewcnt_1d_unflow,a_short_viewuser_1d_flow,a_short_viewuser_1d_unflow,a_long_viewcnt_1d_flow,a_long_viewcnt_1d_unflow,a_long_viewuser_1d_flow,a_long_viewuser_1d_unflow,a_followusercnt_1d,a_followcnt_1d,a_unfollowcnt_1d,a_giftcnt_1d,a_gift_totcnt_1d,a_giftuser_1d,a_danmunum_1d,a_danmuuser_1d,a_kaibo_days_7d,a_kaibo_time_7d,a_kaibo_cnt_7d,a_expnum_7d,a_clicknum_7d,a_clickusernum_7d,a_valid_viewcnt_7d_flow,a_valid_viewcnt_7d_unflow,a_valid_viewuser_7d_flow,a_valid_viewuser_7d_unflow,a_short_viewcnt_7d_flow,a_short_viewcnt_7d_unflow,a_short_viewuser_7d_flow,a_short_viewuser_7d_unflow,a_long_viewcnt_7d_flow,a_long_viewcnt_7d_unflow,a_long_viewuser_7d_flow,a_long_viewuser_7d_unflow,a_followusercnt_7d,a_followcnt_7d,a_unfollowcnt_7d,a_giftcnt_7d,a_gift_totcnt_7d,a_giftuser_7d,a_danmunum_7d,a_danmuuser_7d,a_kaibo_days_15d,a_kaibo_time_15d,a_kaibo_cnt_15d,a_expnum_15d,a_clicknum_15d,a_clickusernum_15d,a_valid_viewcnt_15d_flow,a_valid_viewcnt_15d_unflow,a_valid_viewuser_15d_flow,a_valid_viewuser_15d_unflow,a_short_viewcnt_15d_flow,a_short_viewcnt_15d_unflow,a_short_viewuser_15d_flow,a_short_viewuser_15d_unflow,a_long_viewcnt_15d_flow,a_long_viewcnt_15d_unflow,a_long_viewuser_15d_flow,a_long_viewuser_15d_unflow,a_followusercnt_15d,a_followcnt_15d,a_unfollowcnt_15d,a_giftcnt_15d,a_gift_totcnt_15d,a_giftuser_15d,a_danmunum_15d,a_danmuuser_15d,a_kaibo_days_30d,a_kaibo_time_30d,a_kaibo_cnt_30d,a_expnum_30d,a_clicknum_30d,a_clickusernum_30d,a_valid_viewcnt_30d_flow,a_valid_viewcnt_30d_unflow,a_valid_viewuser_30d_flow,a_valid_viewuser_30d_unflow,a_short_viewcnt_30d_flow,a_short_viewcnt_30d_unflow,a_short_viewuser_30d_flow,a_short_viewuser_30d_unflow,a_long_viewcnt_30d_flow,a_long_viewcnt_30d_unflow,a_long_viewuser_30d_flow,a_long_viewuser_30d_unflow,a_followusercnt_30d,a_followcnt_30d,a_unfollowcnt_30d,a_giftcnt_30d,a_gift_totcnt_30d,a_giftuser_30d,a_danmunum_30d,a_danmuuser_30d,a_sharecnt_1d,a_shareusercnt_1d,a_sharecnt_7d,a_shareusercnt_7d,a_sharecnt_15d,a_shareusercnt_15d,a_sharecnt_30d,a_shareusercnt_30d,imp_label,clk_label,play_time,u_scenario, u_weekday, u_cur_hour, u_scenario_type, u_cur_period, u_click_type_seq, isActive"}

    ds = create_dataset(config_file, data_path, file_type='csv', batch_size=1024, **parserDict)
    i = 0
    # for r in ds:
    #     if i >= 10:
    #         i = 0
    #         break
    #     print(r)
    #     i += 1

    loss1 = create_loss('binary_cross_entropy')
    loss2 = create_loss('truncated_mse')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=[loss1, loss2], metrics={'clk_label': [
        tf.keras.metrics.AUC(),
        tf.keras.metrics.BinaryCrossentropy()],
    'play_time': [
        tf.keras.metrics.MeanSquaredError()]}
    )
    model.fit(ds, epochs=1)
    # model.evaluate(ds)
    mymodel = model.get_train_model()
    mymodel.save('ple-din')

    # plot_model(model, to_file='model.png')