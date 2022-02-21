import tensorflow
from tensorflow.keras.utils import plot_model

import json
import os
from pkgs.tf.feature_util import ModelInputConfig
from models.ple import PLEModel
from models.data_helper import create_dataset
from pkgs.tf.helperfuncs import create_loss
import tensorflow as tf
from tensorflow.keras.utils import plot_model


if __name__ == "__main__":
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)
    config_file = '/Users/fabriszhou/PycharmProjects/tf-model/test/ple/ple_cin_input.json'
    export_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/ple/runs'
    pack_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/ple'
    data_path = '/Users/fabriszhou/PycharmProjects/tf-model/test/ple/part-00004'


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
    cin_layer = [2, 2]

    model = PLEModel(input, ordered_task_names, layer_number, ple_dict, tower_dict, task_output_act, tower_dependencies_dict={}, dropout_layer="PersonalRadioInputDropoutV1", custom_layer_file_path="/Users/fabriszhou/PycharmProjects/tf-model/test/ple/custom/custom_layers.py", is_concat_gate_input=False, use_inputs_dropout=True,
                     use_wide=True, wide_groups=["clk_label", "play_time",  "global"], cin_layer=cin_layer, dim=16, name='ple')

    parserDict = {"field_delim": ",", "with_headers": False,"headers":"ftime,uin,u_click_seq,u_age_level,u_log_days,u_clicknum_1d,anchor_id,a_age,a_kaibo_period, a_kaibo_period_weight,a_clicknum_1d,clk_label,play_time"}
    ds = create_dataset(config_file, data_path, file_type='csv', batch_size=1024, **parserDict)
    i = 0
    for r in ds:
        if i >= 10:
            i = 0
            break
        print(r)
        i += 1

    loss1 = create_loss('binary_cross_entropy')
    loss2 = create_loss('truncated_mse')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=[loss1, loss2], metrics={'clk_label': [
        tf.keras.metrics.AUC(),
        tf.keras.metrics.BinaryCrossentropy()],
    'play_time': [
        tf.keras.metrics.MeanSquaredError()]}
    )
    model.fit(ds, epochs=1)
    # model.save('ple-din')

    plot_model(model, to_file='model.png')