import json
import os
from pkgs.tf.feature_util import ModelInputConfig
from models.mmoe import MMoEModel
from models.data_helper import create_dataset
from pkgs.tf.extend_layers import DNNLayer
from pkgs.tf.helperfuncs import create_loss
import tensorflow as tf


if __name__ == "__main__":
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)
    config_file = '/Users/fabriszhou/PycharmProjects/tf-model/test/mmoe/deepfm_live.json'
    export_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/mmoe/runs'
    pack_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/mmoe'
    data_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/mmoe/part-00000'


    input = ModelInputConfig.parse(config_file, pack_path, export_path)

    task_structs = [[64, 32, 1]]
    num_experts = 3
    expert_layers = [128, 64]
    task_hidden_act = "relu",
    task_output_act = "sigmoid"
    cin_layer = [2]

    model = MMoEModel(input, task_structs, num_experts, expert_layers, task_hidden_act=task_hidden_act, task_output_act=task_output_act, use_wide=True, common_wide=True, cin_layer=cin_layer)

    #model = MMoEModel(input, task_structs, num_experts, expert_layers, task_hidden_act=task_hidden_act, use_wide=True)
    #parserDict = {"field_delim": ",", "with_headers": False, "headers": "ftime,logid,uin,item_id,moment_id,is_click,weight,log_days,net_day,square_clicknum_3d,square_clicknum_15d,square_fig_clicknum_7d,square_fig_clicknum_15d,square_video_clicknum_15d,square_fig_viewtime_3d,square_fig_viewtime_7d,square_fig_viewtime_15d,total_viewcnt_1d,total_viewcnt_3d,total_viewcnt_7d,total_viewcnt_15d,total_viewtime_1d,total_viewtime_3d,total_viewtime_7d,total_viewtime_15d,total_favorcnt_15d,click_num_3d,click_num_15d,favor_num_1d,favor_num_15d,share_num_7d,share_num_15d,comment_num_1d,comment_num_3d,comment_num_7d,comment_num_15d,feed_page_time_3d,feed_page_time_7d,feed_page_time_15d,detail_page_time_7d,detail_page_time_15d,video_playtime_1d,video_playtime_7d,video_playtime_15d,video_duration,freshness,validity_level,example_age,age_level,sex,degree,income_level,city_level,is_kg,is_dapan,os_type,moment_type,is_weixin,cur_hour,pics_num,group_id,publisher_id,singer_profile_sim_avg_7d,singer_profile_sim_max_7d,singer_profile_sim_avg_long,singer_profile_sim_max_long,user_op_avg_sim,user_op_max_sim,ctr_current,ctr_short,ctr_near,playtime_score_short,playtime_score_near,read_score_current,read_score_short,cate_first,cate_second,cate_third,cate_1_7d,cate_1_15d,cate_1_30d,cate_1_90d,cate_2_7d,cate_2_15d,cate_2_30d,cate_2_90d,cate_3_7d,cate_3_15d,cate_3_30d,cate_3_90d,group_90d,publisher_90d,cate_first_weight,cate_second_weight,cate_third_weight,cate_1_7d_weight,cate_1_15d_weight,cate_1_30d_weight,cate_1_90d_weight,cate_2_7d_weight,cate_2_15d_weight,cate_2_30d_weight,cate_2_90d_weight,cate_3_7d_weight,cate_3_15d_weight,cate_3_30d_weight,cate_3_90d_weight,group_90d_weight,publisher_90d_weight"}

    parserDict = {"field_delim": ",", "with_headers": False, "headers":"ftime,uin,u_click_seq,u_age_level,u_log_days,u_clicknum_1d,anchor_id,a_age,a_kaibo_period, a_kaibo_period_weight, a_clicknum_1d,clk_label,play_time"}

    ds = create_dataset(config_file, data_path, file_type='csv', batch_size=64, **parserDict)
    i = 0
    for r in ds:
        if i >= 10:
            i = 0
            break
        print(r)
        i += 1

    loss = create_loss('binary_cross_entropy')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=loss, metrics={'clk_label': [tf.keras.metrics.AUC(), tf.keras.metrics.BinaryCrossentropy()]})
    model.fit(ds, epochs=10)
    model.save('mmoe-cin')



