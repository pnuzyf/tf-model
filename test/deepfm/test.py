import json
import os
from pkgs.tf.feature_util import ModelInputConfig
from models.deepfm import DeepFMModel
from models.data_helper import create_dataset
import tensorflow as tf


if __name__ == "__main__":
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)
    config_file = '/Users/fabriszhou/PycharmProjects/tf-model/test/deepfm/deepfm_live.json'
    export_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/deepfm/runs'
    pack_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/deepfm'
    data_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/deepfm/part-00000'


    input = ModelInputConfig.parse(config_file, pack_path, export_path)
    k = 8
    dnn_hidden_widthes = [64, 48]
    dnn_hidden_active_fn = 'lrelu'

    model = DeepFMModel(input, k, dnn_hidden_widthes, dnn_hidden_active_fn, use_gate=True, gate_act='tanh')

    #parserDict = {"field_delim": ",", "with_headers": False, "headers": "ftime,logid,uin,item_id,moment_id,is_click,weight,log_days,net_day,square_clicknum_3d,square_clicknum_15d,square_fig_clicknum_7d,square_fig_clicknum_15d,square_video_clicknum_15d,square_fig_viewtime_3d,square_fig_viewtime_7d,square_fig_viewtime_15d,total_viewcnt_1d,total_viewcnt_3d,total_viewcnt_7d,total_viewcnt_15d,total_viewtime_1d,total_viewtime_3d,total_viewtime_7d,total_viewtime_15d,total_favorcnt_15d,click_num_3d,click_num_15d,favor_num_1d,favor_num_15d,share_num_7d,share_num_15d,comment_num_1d,comment_num_3d,comment_num_7d,comment_num_15d,feed_page_time_3d,feed_page_time_7d,feed_page_time_15d,detail_page_time_7d,detail_page_time_15d,video_playtime_1d,video_playtime_7d,video_playtime_15d,video_duration,freshness,validity_level,example_age,age_level,sex,degree,income_level,city_level,is_kg,is_dapan,os_type,moment_type,is_weixin,cur_hour,pics_num,group_id,publisher_id,singer_profile_sim_avg_7d,singer_profile_sim_max_7d,singer_profile_sim_avg_long,singer_profile_sim_max_long,user_op_avg_sim,user_op_max_sim,ctr_current,ctr_short,ctr_near,playtime_score_short,playtime_score_near,read_score_current,read_score_short,cate_first,cate_second,cate_third,cate_1_7d,cate_1_15d,cate_1_30d,cate_1_90d,cate_2_7d,cate_2_15d,cate_2_30d,cate_2_90d,cate_3_7d,cate_3_15d,cate_3_30d,cate_3_90d,group_90d,publisher_90d,cate_first_weight,cate_second_weight,cate_third_weight,cate_1_7d_weight,cate_1_15d_weight,cate_1_30d_weight,cate_1_90d_weight,cate_2_7d_weight,cate_2_15d_weight,cate_2_30d_weight,cate_2_90d_weight,cate_3_7d_weight,cate_3_15d_weight,cate_3_30d_weight,cate_3_90d_weight,group_90d_weight,publisher_90d_weight"}

    parserDict = {"field_delim": ",", "with_headers": False, "headers": "ftime,uin,u_click_seq,u_age_level,u_sex,u_degree,u_income_level,u_city_level,u_os_type,u_log_days,u_net_day,user_type,live_active_label,u_lastclick_time,u_lastvalidview_time,u_lastshortview_time,u_lastlongview_time,u_lastgift_time,u_last_danmu_time,u_expnum_1d,u_clicknum_1d,u_clickzhubonum_1d,u_ctr_1d,u_valid_viewcnt_1d_flow,u_valid_viewcnt_1d_unflow,u_valid_viewzhubocnt_1d_flow,u_valid_viewzhubocnt_1d_unflow,u_valid_viewrate_1d,u_short_viewcnt_1d_flow,u_short_viewcnt_1d_unflow,u_short_viewzhubocnt_1d_flow,u_short_viewzhubocnt_1d_unflow,u_short_viewrate_1d,u_long_viewcnt_1d_flow,u_long_viewcnt_1d_unflow,u_long_viewzhubocnt_1d_flow,u_long_viewzhubocnt_1d_unflow,u_long_viewrate_1d,u_followcnt_1d,u_unfollow_1d,u_giftcnt_1d,u_gift_totcnt_1d,u_giftzhubocnt_1d,u_chargecnt_1d,u_chargemoney_1d,u_danmunum_1d,u_danmuzhubocnt_1d,u_expnum_7d,u_clicknum_7d,u_clickzhubonum_7d,u_ctr_7d,u_valid_viewcnt_7d_flow,u_valid_viewcnt_7d_unflow,u_valid_viewzhubocnt_7d_flow,u_valid_viewzhubocnt_7d_unflow,u_valid_viewrate_7d,u_short_viewcnt_7d_flow,u_short_viewcnt_7d_unflow,u_short_viewzhubocnt_7d_flow,u_short_viewzhubocnt_7d_unflow,u_short_viewrate_7d,u_long_viewcnt_7d_flow,u_long_viewcnt_7d_unflow,u_long_viewzhubocnt_7d_flow,u_long_viewzhubocnt_7d_unflow,u_long_viewrate_7d,u_followcnt_7d,u_unfollow_7d,u_giftcnt_7d,u_gift_totcnt_7d,u_giftzhubocnt_7d,u_chargecnt_7d,u_chargemoney_7d,u_danmunum_7d,u_danmuzhubocnt_7d,u_expnum_15d,u_clicknum_15d,u_clickzhubonum_15d,u_ctr_15d,u_valid_viewcnt_15d_flow,u_valid_viewcnt_15d_unflow,u_valid_viewzhubocnt_15d_flow,u_valid_viewzhubocnt_15d_unflow,u_valid_viewrate_15d,u_short_viewcnt_15d_flow,u_short_viewcnt_15d_unflow,u_short_viewzhubocnt_15d_flow,u_short_viewzhubocnt_15d_unflow,u_short_viewrate_15d,u_long_viewcnt_15d_flow,u_long_viewcnt_15d_unflow,u_long_viewzhubocnt_15d_flow,u_long_viewzhubocnt_15d_unflow,u_long_viewrate_15d,u_followcnt_15d,u_unfollow_15d,u_giftcnt_15d,u_gift_totcnt_15d,u_giftzhubocnt_15d,u_chargecnt_15d,u_chargemoney_15d,u_danmunum_15d,u_danmuzhubocnt_15d,u_expnum_30d,u_clicknum_30d,u_clickzhubonum_30d,u_ctr_30d,u_valid_viewcnt_30d_flow,u_valid_viewcnt_30d_unflow,u_valid_viewzhubocnt_30d_flow,u_valid_viewzhubocnt_30d_unflow,u_valid_viewrate_30d,u_short_viewcnt_30d_flow,u_short_viewcnt_30d_unflow,u_short_viewzhubocnt_30d_flow,u_short_viewzhubocnt_30d_unflow,u_short_viewrate_30d,u_long_viewcnt_30d_flow,u_long_viewcnt_30d_unflow,u_long_viewzhubocnt_30d_flow,u_long_viewzhubocnt_30d_unflow,u_long_viewrate_30d,u_followcnt_30d,u_unfollow_30d,u_giftcnt_30d,u_gift_totcnt_30d,u_giftzhubocnt_30d,u_chargecnt_30d,u_chargemoney_30d,u_danmunum_30d,u_danmuzhubocnt_30d,u_sharecnt_1d,u_sharecnt_7d,u_sharecnt_15d,u_sharecnt_30d,anchor_id,a_age,a_sex,a_province,a_kaibo_type,rent_level,finance_level,a_first_launch_time,a_kaibo_period,a_kaibo_period_weight,a_kaibo_days_1d,a_kaibo_time_1d,a_kaibo_cnt_1d,a_expnum_1d,a_clicknum_1d,a_clickusernum_1d,a_ctr_1d,a_valid_viewcnt_1d_flow,a_valid_viewcnt_1d_unflow,a_valid_viewuser_1d_flow,a_valid_viewuser_1d_unflow,a_valid_viewrate_1d,a_short_viewcnt_1d_flow,a_short_viewcnt_1d_unflow,a_short_viewuser_1d_flow,a_short_viewuser_1d_unflow,a_short_viewrate_1d,a_long_viewcnt_1d_flow,a_long_viewcnt_1d_unflow,a_long_viewuser_1d_flow,a_long_viewuser_1d_unflow,a_long_viewrate_1d,a_followusercnt_1d,a_followcnt_1d,a_unfollowcnt_1d,a_giftcnt_1d,a_gift_totcnt_1d,a_giftuser_1d,a_danmunum_1d,a_danmuuser_1d,a_kaibo_days_7d,a_kaibo_time_7d,a_kaibo_cnt_7d,a_expnum_7d,a_clicknum_7d,a_clickusernum_7d,a_ctr_7d,a_valid_viewcnt_7d_flow,a_valid_viewcnt_7d_unflow,a_valid_viewuser_7d_flow,a_valid_viewuser_7d_unflow,a_valid_viewrate_7d,a_short_viewcnt_7d_flow,a_short_viewcnt_7d_unflow,a_short_viewuser_7d_flow,a_short_viewuser_7d_unflow,a_short_viewrate_7d,a_long_viewcnt_7d_flow,a_long_viewcnt_7d_unflow,a_long_viewuser_7d_flow,a_long_viewuser_7d_unflow,a_long_viewrate_7d,a_followusercnt_7d,a_followcnt_7d,a_unfollowcnt_7d,a_giftcnt_7d,a_gift_totcnt_7d,a_giftuser_7d,a_danmunum_7d,a_danmuuser_7d,a_kaibo_days_15d,a_kaibo_time_15d,a_kaibo_cnt_15d,a_expnum_15d,a_clicknum_15d,a_clickusernum_15d,a_ctr_15d,a_valid_viewcnt_15d_flow,a_valid_viewcnt_15d_unflow,a_valid_viewuser_15d_flow,a_valid_viewuser_15d_unflow,a_valid_viewrate_15d,a_short_viewcnt_15d_flow,a_short_viewcnt_15d_unflow,a_short_viewuser_15d_flow,a_short_viewuser_15d_unflow,a_short_viewrate_15d,a_long_viewcnt_15d_flow,a_long_viewcnt_15d_unflow,a_long_viewuser_15d_flow,a_long_viewuser_15d_unflow,a_long_viewrate_15d,a_followusercnt_15d,a_followcnt_15d,a_unfollowcnt_15d,a_giftcnt_15d,a_gift_totcnt_15d,a_giftuser_15d,a_danmunum_15d,a_danmuuser_15d,a_kaibo_days_30d,a_kaibo_time_30d,a_kaibo_cnt_30d,a_expnum_30d,a_clicknum_30d,a_clickusernum_30d,a_ctr_30d,a_valid_viewcnt_30d_flow,a_valid_viewcnt_30d_unflow,a_valid_viewuser_30d_flow,a_valid_viewuser_30d_unflow,a_valid_viewrate_30d,a_short_viewcnt_30d_flow,a_short_viewcnt_30d_unflow,a_short_viewuser_30d_flow,a_short_viewuser_30d_unflow,a_short_viewrate_30d,a_long_viewcnt_30d_flow,a_long_viewcnt_30d_unflow,a_long_viewuser_30d_flow,a_long_viewuser_30d_unflow,a_long_viewrate_30d,a_followusercnt_30d,a_followcnt_30d,a_unfollowcnt_30d,a_giftcnt_30d,a_gift_totcnt_30d,a_giftuser_30d,a_danmunum_30d,a_danmuuser_30d,a_sharecnt_1d,a_shareusercnt_1d,a_sharecnt_7d,a_shareusercnt_7d,a_sharecnt_15d,a_shareusercnt_15d,a_sharecnt_30d,a_shareusercnt_30d,imp_label,clk_label,play_time"}
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



