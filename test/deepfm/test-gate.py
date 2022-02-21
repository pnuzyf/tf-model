import json
import os
from pkgs.tf.feature_util import ModelInputConfig
from models.deepfm import DeepFMModel
from models.data_helper import create_dataset
import tensorflow as tf


if __name__ == "__main__":
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)
    config_file = '/Users/fabriszhou/PycharmProjects/tf-model/test/deepfm/model_input_low.json'
    export_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/deepfm/runs'
    pack_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/deepfm'
    data_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/deepfm/part-00011'


    input = ModelInputConfig.parse(config_file, pack_path, export_path)
    k = 8
    dnn_hidden_widthes = [64, 48]
    dnn_hidden_active_fn = 'lrelu'

    model = DeepFMModel(input, k, dnn_hidden_widthes, dnn_hidden_active_fn, use_gate=True, gate_act='tanh')

    #parserDict = {"field_delim": ",", "with_headers": False, "headers": "ftime,logid,uin,item_id,moment_id,is_click,weight,log_days,net_day,square_clicknum_3d,square_clicknum_15d,square_fig_clicknum_7d,square_fig_clicknum_15d,square_video_clicknum_15d,square_fig_viewtime_3d,square_fig_viewtime_7d,square_fig_viewtime_15d,total_viewcnt_1d,total_viewcnt_3d,total_viewcnt_7d,total_viewcnt_15d,total_viewtime_1d,total_viewtime_3d,total_viewtime_7d,total_viewtime_15d,total_favorcnt_15d,click_num_3d,click_num_15d,favor_num_1d,favor_num_15d,share_num_7d,share_num_15d,comment_num_1d,comment_num_3d,comment_num_7d,comment_num_15d,feed_page_time_3d,feed_page_time_7d,feed_page_time_15d,detail_page_time_7d,detail_page_time_15d,video_playtime_1d,video_playtime_7d,video_playtime_15d,video_duration,freshness,validity_level,example_age,age_level,sex,degree,income_level,city_level,is_kg,is_dapan,os_type,moment_type,is_weixin,cur_hour,pics_num,group_id,publisher_id,singer_profile_sim_avg_7d,singer_profile_sim_max_7d,singer_profile_sim_avg_long,singer_profile_sim_max_long,user_op_avg_sim,user_op_max_sim,ctr_current,ctr_short,ctr_near,playtime_score_short,playtime_score_near,read_score_current,read_score_short,cate_first,cate_second,cate_third,cate_1_7d,cate_1_15d,cate_1_30d,cate_1_90d,cate_2_7d,cate_2_15d,cate_2_30d,cate_2_90d,cate_3_7d,cate_3_15d,cate_3_30d,cate_3_90d,group_90d,publisher_90d,cate_first_weight,cate_second_weight,cate_third_weight,cate_1_7d_weight,cate_1_15d_weight,cate_1_30d_weight,cate_1_90d_weight,cate_2_7d_weight,cate_2_15d_weight,cate_2_30d_weight,cate_2_90d_weight,cate_3_7d_weight,cate_3_15d_weight,cate_3_30d_weight,cate_3_90d_weight,group_90d_weight,publisher_90d_weight"}

    parserDict = {"field_delim": ",", "with_headers": False, "headers": "ftime,uin, u_click_seq, u_pre_singer,u_pre_singer_weight,u_age_level, u_sex, u_degree, u_income_level, u_city_level, u_os_type, u_log_days, u_net_day,u_expnum_30d ,u_clicknum_30d ,u_clickzhubonum_30d ,u_valid_viewcnt_30d_flow,u_valid_viewcnt_30d_unflow ,u_valid_viewzhubocnt_30d_flow ,u_valid_viewzhubocnt_30d_unflow,u_short_viewcnt_30d_flow ,u_short_viewcnt_30d_unflow ,u_short_viewzhubocnt_30d_flow,u_short_viewzhubocnt_30d_unflow  ,u_long_viewcnt_30d_flow ,u_long_viewcnt_30d_unflow,u_long_viewzhubocnt_30d_flow ,u_long_viewzhubocnt_30d_unflow ,u_followcnt_30d,u_unfollow_30d ,u_giftcnt_30d ,u_gift_totcnt_30d ,u_giftzhubocnt_30d ,u_chargecnt_30d,u_chargemoney_30d ,u_danmunum_30d ,u_danmuzhubocnt_30d,u_sharecnt_30d, anchor_id, a_age, a_sex,a_province, a_kaibo_type, watch_score, finance_score,  a_kaibo_period,a_kaibo_period_weight,a_kaibo_days_15d,a_kaibo_time_15d,a_kaibo_cnt_15d,a_expnum_15d,a_clicknum_15d,a_clickusernum_15d,a_valid_viewcnt_15d_flow,a_valid_viewcnt_15d_unflow,a_valid_viewuser_15d_flow,a_valid_viewuser_15d_unflow,a_short_viewcnt_15d_flow,a_short_viewcnt_15d_unflow,a_short_viewuser_15d_flow,a_short_viewuser_15d_unflow,a_long_viewcnt_15d_flow,a_long_viewcnt_15d_unflow,a_long_viewuser_15d_flow,a_long_viewuser_15d_unflow,a_followusercnt_15d,a_followcnt_15d,a_unfollowcnt_15d,a_giftcnt_15d,a_gift_totcnt_15d,a_giftuser_15d,a_danmunum_15d,a_danmuuser_15d,a_sharecnt_15d,a_shareusercnt_15d,,imp_label,clk_label,play_time,u_scenario, u_weekday,u_cur_hour"}
    ds = create_dataset(config_file, data_path, file_type='csv', batch_size=64, **parserDict)
    i = 0
    for r in ds:
        if i >= 10:
            i = 0
            break
        print(r)
        i += 1
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss='binary_crossentropy', metrics=[[tf.keras.metrics.AUC(), tf.keras.metrics.BinaryCrossentropy()]])
    model.fit(ds, epochs=10)



