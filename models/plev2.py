'''
Author: jurluo
'''
from abc import ABC
from pkgs.tf.extend_layers import ModelInputLayer, LocalActivationUnit, AttentionLayer, CINLayer, DNNLayer, PLELayer
import tensorflow as tf
from pkgs.tf.feature_util import *
from pkgs.utils import dynamic_load_class
from pkgs.tf.helperfuncs import create_activate_func


#share_type: 0 global 1 独立  2 混合. wide_group 来控制 cin层的类型，有共享，有独立，也可以共享+独立, 如果设置use_shallow，需要shallow_tower的任务按顺序设置
#ple输入分为 dnn侧输入、wide侧输入（根据组名）、 shallow输入
class PLEModel(tf.keras.Model, ABC):
    def __init__(self, model_input_config: ModelInputConfig, ordered_task_names, layer_number, ple_dict, tower_dict, task_output_act, tower_dependencies_dict, dropout_layer, custom_layer_file_path, is_concat_gate_input, use_inputs_dropout,
                 named_outputs=None, use_wide=False, wide_groups=None, cin_layer=None, dim=8, wide_output_logit=True, use_shallow=False, shallow_tower_dict=None, shallow_tower_act='relu',
                 use_gate=False, common_gate=False, gate_act='relu', name='ple'):
        super(PLEModel, self).__init__(name=name)
        self.is_first = True
        self.ple_layer = PLELayer(custom_layer_file_path, ordered_task_names, layer_number, ple_dict, use_inputs_dropout, dropout_layer, is_concat_gate_input)
        self.input_layer = ModelInputLayer(model_input_config, auto_create_embedding=True)
        # Attention层的核心使用的是LocalActivationUnit-- 这里可以通过配置
        self.attention_layer = {}  # 这里可能存在多个
        self.gates = {} #gate net

        # common_gate: 单一gate 还是 多个gate is_filter： 用来控制是否该特征需要经过gate-net
        if use_gate:
            if common_gate:
                self.gates["common"] = tf.keras.layers.Dense(
                    1, activation=gate_act, use_bias=False,
                    kernel_regularizer=None,
                    name=self.name + '/gate_common')
        for input_config in model_input_config.all_inputs:
            name = input_config.name
            if input_config.query:
                self.attention_layer[name] = AttentionLayer(LocalActivationUnit(hidden_units=[36, 1], activations=['dice'], mode='SUM', normalization=True))
            if use_gate and not common_gate:
                if input_config.is_filter:
                    self.gates[name] = tf.keras.layers.Dense(
                        1, activation=gate_act, use_bias=False,
                        kernel_regularizer=None,
                        name=self.name + '/gate_{}'.format(name)
                    )

        # tower layer
        self.towers = dict()  # 塔
        for task_name, layer_class in tower_dict.items():
            if len(layer_class) == 0:
                raise RuntimeError("layer_class is empty, layer_class is: {}".format(layer_class))
            Tower = dynamic_load_class(custom_layer_file_path, layer_class)
            self.towers[task_name] = Tower(task_name)


        self.label_names = [i.name for i in model_input_config.all_inputs if i.is_label]
        print("label_names={}".format(self.label_names))
        if named_outputs is None:
            named_outputs = len(self.label_names) > 1
        print("named_outputs={}".format(named_outputs))
        self.model_input_config = model_input_config
        self.is_concat_gate_input = is_concat_gate_input
        self.custom_layer_file_path = custom_layer_file_path
        self.ordered_task_names = ordered_task_names
        self.layer_number = layer_number
        self.ple_dict = ple_dict
        self.tower_dict = tower_dict
        self.tower_dependencies_dict = tower_dependencies_dict
        self.dropout_layer = dropout_layer
        self.use_inputs_dropout = use_inputs_dropout
        self.named_outputs = named_outputs
        self.mixed_precision = tf.keras.mixed_precision.experimental.get_layer_policy(self).loss_scale is not None
        print("{}: mixed_precision={}".format(self.name, self.mixed_precision))


        self.dim = dim
        self.wide_feats = {}

        self.cin_layer = {}
        self.use_wide = use_wide
        self.wide_output_logit = wide_output_logit
        self.use_shallow = use_shallow
        self.shallow_tower_dict = shallow_tower_dict
        self.shallow_tower_act = shallow_tower_act
        self.wide_groups = wide_groups
        self.use_gate = use_gate
        self.common_gate = common_gate
        self.gate_act = gate_act

        # 是否存在wide侧
        if use_wide:
            if not wide_groups or not isinstance(wide_groups, (list, tuple)):
                raise RuntimeError("'wide_groups' should be a non-emplty list, got '{}' : {}".format(type(wide_groups), wide_groups))

            #配置每个wide侧的 特征以及构建对应CINlayer
            for group in wide_groups:
                self.cin_layer[group] = CINLayer(cin_layer, dim, activate='relu', use_bias=True)
                wide_feat = []
                for input_config in model_input_config.get_inputs_by_group(group):
                    wide_feat.append(input_config.name) #List
                self.wide_feats[group] = wide_feat

        #是否存在shallow_tower
        self.shallow_tower = {}
        if use_shallow:
            for task_index, tower_struct in shallow_tower_dict.items():
                # self.shallow_tower.append(DNNLayer(tower_struct, shallow_tower_act, None, name='task_{}'.format(task_index)))
                self.shallow_tower[task_index] = DNNLayer(tower_struct, shallow_tower_act, None, name='task_{}'.format(task_index))

        #输出层单独
        self.task_active_layers = []
        for i in range(len(ordered_task_names)):
            self.task_active_layers.append(
                tf.keras.layers.Activation(
                    create_activate_func(task_output_act[i]),
                    name=self.name + "/output_act_{}_task{}".format(task_output_act[i], i)
                ))  # 输出层单独

    @tf.function
    def call(self, inputs, training=None, mask=None):
        feature_groups = ['dnn']
        if self.wide_groups:
            feature_groups = feature_groups + self.wide_groups
        feat_vals = defaultdict(list) #模型的不同侧的特征输入

        if self.use_gate:
            # 门控的输入
            gate_vals = []
            transformed_inputs = self.input_layer(inputs, groups="gate")  # 获取指定组的特征
            for input_config in self.model_input_config.get_inputs_by_group("gate"):
                if input_config.is_gate:
                    name = input_config.name
                    m = transformed_inputs[name]  # 特征
                    val = tf.keras.layers.Flatten()(m)
                    gate_vals.append(val)
            gate_concat_vals = tf.concat(gate_vals, axis=-1, name='concat_gate_features')


        for group in feature_groups:
            transformed_inputs = self.input_layer(inputs, groups=group) #获取指定组的特征

            for input_config in self.model_input_config.get_inputs_by_group(group):
                if input_config.is_weight_col or input_config.is_label:
                    continue
                name = input_config.name
                m = transformed_inputs[name]  # 特征
                if input_config.query:
                    q = transformed_inputs[input_config.query] #query
                    # 拿到attention score
                    hist_len = self.input_layer.compute_mask_histlen(inputs, name,
                                                                     return_seqlist=True)
                    m = self.attention_layer[name]([q, m, m, hist_len])  # [batch_size, 1, dim]
                    m = tf.squeeze(m, axis=1)

                if self.use_gate and input_config.is_filter:
                        if self.common_gate:
                            gate_output = self.gates["common"](gate_concat_vals)
                        else:
                            gate_output = self.gates[name](gate_concat_vals)
                        m = gate_output * m
                val = tf.keras.layers.Flatten()(m)
                if not val.dtype.is_integer and not val.dtype.is_floating and not val.dtype.is_bool:
                    raise RuntimeError("dtype of input '{}' is {}, only float/int/bool are allowed"
                                       .format(name, val.dtype))
                elif val.dtype != self._dtype_policy.compute_dtype:
                    val = tf.cast(val, self._dtype_policy.compute_dtype,
                                  name=name + '_cast2' + self._dtype_policy.compute_dtype)

                feat_vals[group].append(val) #插入特征进入对应的侧


        wide_logits = {}  # 多个输出
        wide_cross_feats_dict = {}
        if self.use_wide:
            for group in self.wide_groups:
                value = feat_vals[group]
                stacked_wide_vals = tf.stack(value, axis=1) #[B,m,D]
                wide_logit, wide_cross_feats = self.cin_layer[group](stacked_wide_vals)
                wide_logits[group] = wide_logit
                wide_cross_feats_dict[group] = wide_cross_feats


        shallow_logits = [] #多个输出

        if self.use_shallow:
            for key in self.shallow_tower_dict.keys():
                shallow_vals = []
                model_input = self.input_layer(inputs, groups='shallow_{}'.format(key))
                for name, val in model_input.items():
                    val = tf.keras.layers.Flatten()(val)
                    shallow_vals.append(val)
                shallow_feats = tf.concat(shallow_vals, axis=-1, name='concat_shallow_features_{}'.format(key))
                # shallow_logits.append(self.shallow_tower[int(key)](shallow_feats))
                shallow_logits.append(self.shallow_tower[key](shallow_feats))


        concat_vals = tf.concat(feat_vals['dnn'], axis=-1, name='concat_features')
        ple_outputs = self.ple_layer(concat_vals, training=training)

        tower_result = dict()
        # 是否拼接moe的输出
        for task_name in [tn for tn in self.ordered_task_names]:
            if self.wide_output_logit:
                concat_vals = ple_outputs[task_name]
            else:
                concat_vals = tf.concat([wide_cross_feats_dict['global'], ple_outputs[task_name]], axis=-1,
                                        name='concat_task_{}_features'.format(task_name))
            tower_result[task_name] = self.towers[task_name](concat_vals)

        for task_main, task_parent in self.tower_dependencies_dict.items():  # task_main = task_main * task_parent，当两者都在0～1之间的时候，task_main必然小于task_parent
            tower_result[task_main] *= tower_result[task_parent]

        task_outputs = []
        out_i = 0
        for task_name, task_output in tower_result.items(): # label_name need to be equal to task name
            if self.mixed_precision and task_output.dtype != tf.float32:
                task_output = tf.cast(task_output, tf.float32, self.name+"_mp_output_{}_cast2float32".format(out_i))
            task_outputs.append(task_output)
            out_i += 1

        idx = 0
        if self.named_outputs:
            output_dict = {}
            for name, output, task_act_layer in zip(self.label_names, task_outputs, self.task_active_layers):
                if wide_logits is not None and self.use_wide:
                    if "global" in self.wide_groups:
                        #需要输出logit
                        if self.wide_output_logit:
                            output = output + wide_logits["global"]
                    if name in self.wide_groups:
                        if self.wide_output_logit:
                            output = output + wide_logits[name]
                if self.use_shallow and idx < len(self.shallow_tower_dict):
                    output = output + shallow_logits[idx]
                if task_act_layer:
                    output_dict[name] = task_act_layer(output)
                else:
                    output_dict[name] = tf.nn.sigmoid(output)

                idx += 1
            if self.is_first:
                self.is_first = False
                self.summary()

            return output_dict
        else:
            outputs = []
            for output, task_act_layer in zip(task_outputs, self.task_active_layers):
                # 如果设置output_cross为0, 则将mmoe的task_layer的输出和wide侧模型的输出进行加和, 再经过sigmoid得到最终输出
                if wide_logits is not None and self.use_wide:  # 0: add; 1: concat;
                    if "global" in self.wide_groups:
                        output = output + wide_logits["global"]
                    if name in self.wide_groups:
                        output = output + wide_logits[name]
                    if self.use_shallow and idx < len(self.shallow_tower_dict):
                        output = output + shallow_logits[idx]
                if task_act_layer:
                    outputs.append(task_act_layer(output))
                else:
                    outputs.append(tf.nn.sigmoid(output))
                idx += 1

            if self.is_first:
                self.is_first = False
                self.summary()

            return tuple(outputs)

    def get_config(self):
        config = {
            'model_input_config': self.model_input_config,
            'custom_layer_file_path': self.custom_layer_file_path,
            'ordered_task_names': self.ordered_task_names,
            'layer_number': self.layer_number,
            'ple_dict': self.ple_dict,
            'tower_dict': self.tower_dict,
            'tower_dependencies_dict': self.tower_dependencies_dict,
            'dropout_layer': self.dropout_layer,
            'use_inputs_dropout': self.use_inputs_dropout,
            'is_concat_gate_input': self.is_concat_gate_input,
            'namded_outputs': self.named_outputs,
            'use_wide': self.use_wide,
            'wide_output_logit': self.wide_output_logit,
            'wide_groups': self.wide_groups,
            'cin_layer': self.cin_layer,
            'dim': self.dim,
            'use_shallow': self.use_shallow,
            'shallow_tower_dict': self.shallow_tower_dict,
            'shallow_tower_act': self.shallow_tower_act,
            'use_gate': self.use_gate,
            'common_gate': self.common_gate,
            'gate_act': self.gate_act,
            'name': self.name
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_save_signatures(self):
        call_fn_specs = self.input_layer.get_tensor_specs()
        sigs = {
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: self.call.get_concrete_function(call_fn_specs)
        }
        return sigs

'''
ple_dict = dict(
    size: (task_number + 1(shared_experts)),
    key: task_name or shared_experts_block_name,
    value: expert_dict
)
expert_dict = dict(
    size: expert_number,
    key: expert_id,
    value: layer_class
)
tower_dict = dict(
    size: task_number,
    key: task_name,
    value: layer_class
)
'''