'''
Author: jurluo
'''
from abc import ABC
from pkgs.tf.extend_layers import ModelInputLayer, LocalActivationUnit, AttentionLayer, CINLayer, DNNLayer
import tensorflow as tf
from pkgs.tf.feature_util import *
from pkgs.utils import dynamic_load_class
from pkgs.tf.helperfuncs import create_activate_func


#share_type: 0 global 1 独立  2 混合. wide_group 来控制 cin层的类型，有共享，有独立，也可以共享+独立, 如果设置use_shallow，需要shallow_tower的任务按顺序设置
#ple输入分为 dnn侧输入、wide侧输入（根据组名）、 shallow输入
class PLEModel(tf.keras.Model, ABC):
    def __init__(self, model_input_config: ModelInputConfig, ordered_task_names, layer_number, ple_dict, tower_dict, task_output_act, tower_dependencies_dict, dropout_layer, custom_layer_file_path, is_concat_gate_input, use_inputs_dropout,
                 named_outputs=None, use_wide=False, wide_groups=None, cin_layer=None, dim=8, use_shallow=False, shallow_tower_dict=None, shallow_tower_act='relu', use_gate=False, gate_act='relu', name='ple'):
        super(PLEModel, self).__init__(name=name)
        self.is_first = True
        self.ple_layer = PLELayer(custom_layer_file_path, ordered_task_names, layer_number, ple_dict, tower_dict, tower_dependencies_dict, use_inputs_dropout, dropout_layer, is_concat_gate_input)
        self.input_layer = ModelInputLayer(model_input_config, auto_create_embedding=True)
        # Attention层的核心使用的是LocalActivationUnit-- 这里可以通过配置
        self.attention_layer = {}  # 这里可能存在多个
        self.gate = {} #gate net
        for input_config in model_input_config.all_inputs:
            name = input_config.name
            if input_config.query:
                self.attention_layer[name] = AttentionLayer(
                    LocalActivationUnit(hidden_units=[36, 1], activations=['dice'], mode='SUM',
                                        normalization=True)
                )
            # 是否 存在gate-net来控制输入
            if use_gate:
                if input_config.is_label or input_config.is_weight_col or input_config.is_gate:
                    continue
                self.gates[name] = tf.keras.layers.Dense(
                    1, activation=gate_act, use_bias=False,
                    kernel_regularizer=None,
                    name=self.name + '/gate_{}'.format(name)
                )
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
        self.use_shallow = use_shallow
        self.shallow_tower_dict = shallow_tower_dict
        self.shallow_tower_act = shallow_tower_act
        self.wide_groups = wide_groups
        self.use_gate = use_gate
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

                if self.use_gate:
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
        if self.use_wide:
            for group in self.wide_groups:
                value = feat_vals[group]
                stacked_wide_vals = tf.stack(value, axis=1) #[B,m,D]
                wide_logit, wide_cross_feats = self.cin_layer[group](stacked_wide_vals)
                wide_logits[group] = wide_logit

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
        task_outputs = []
        out_i = 0
        for task_name, task_output in ple_outputs.items(): # label_name need to be equal to task name
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
                        output = output + wide_logits["global"]
                    if name in self.wide_groups:
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
            'wide_groups': self.wide_groups,
            'cin_layer': self.cin_layer,
            'dim': self.dim,
            'use_shallow': self.use_shallow,
            'shallow_tower_dict': self.shallow_tower_dict,
            'shallow_tower_act': self.shallow_tower_act,
            'use_gate': self.use_gate,
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
class PLELayer(tf.keras.layers.Layer):
    def __init__(self, custom_layer_file_path, ordered_task_names, layer_number, ple_dict, tower_dict, tower_dependencies_dict, use_inputs_dropout, dropout_layer, is_concat_gate_input, name='ple_layer', trainable=True, **kwargs):
        super(PLELayer, self).__init__(name=name, trainable=trainable, **kwargs)
        assert layer_number > 0, "'layer_number' should be a positive integer, got '{}': {}"\
            .format(type(layer_number), layer_number)
        if not ple_dict or type(ple_dict) != type(dict()):
            raise RuntimeError("'ple_dict' should not be none and should be dict type")
        if 'shared_experts' not in ple_dict: # 共享task用一个固定的字符串吧
            raise RuntimeError("'shared_experts' must in ple_dict for ple_layer")
        assert len(ple_dict) >= 3, "'ple_dict' should have not less than 2 tasks, got {} tasks" \
            .format(len(ple_dict))
        if not tower_dict or type(tower_dict) != type(dict()) or len(tower_dict) + 1 != len(ple_dict):
            raise RuntimeError("'tower_dict' should not be none and should be dict type and towers size is tasks number")
        assert len(tower_dict) >= 2, "'tower_dict' should have not less than 2 tasks, got {} tasks" \
            .format(len(tower_dict))
        tns = [k for k, _ in ple_dict.items()]
        if len(ordered_task_names) + 1 != len(ple_dict) or any([task_name not in tns for task_name in ordered_task_names]):
            raise RuntimeError("'ordered_task_names' should has equal task names as 'ple_dict' except 'shared_experts'")
        tns2 = [k for k, _ in tower_dict.items()]
        if any([task_name not in tns2 for task_name in ordered_task_names]):
            raise RuntimeError("'ordered_task_names' should has equal task names as 'tower_dict'")
        for task_main, task_parent in tower_dependencies_dict.items():
            if task_main not in ordered_task_names:
                raise RuntimeError("task_main {} is not in rodered_task_names from tower_dependencies_dict".format(task_main))
            if task_parent not in ordered_task_names:
                raise RuntimeError("task_parent {} is not in rodered_task_names from tower_dependencies_dict".format(task_parent))
        self.custom_layer_file_path = custom_layer_file_path
        self.is_concat_gate_input = is_concat_gate_input
        self.ordered_task_names = ordered_task_names + ['shared_experts'] # task的顺序，固定计算图
        self.layer_number = layer_number
        self.ple_dict = ple_dict
        self.tower_dict = tower_dict
        self.tower_dependencies_dict = tower_dependencies_dict
        self.use_inputs_dropout = use_inputs_dropout
        self.dropout_layer = dropout_layer
        if use_inputs_dropout:
            PersonalRadioInputDropoutV1 = dynamic_load_class(self.custom_layer_file_path, dropout_layer)
            self.inp_dr = PersonalRadioInputDropoutV1()
        else:
            self.inp_dr = None
        self.layerid2block = dict() # 专家网络
        self.layerid2gate = dict() # 门控
        self.shared_experts_number = len(ple_dict['shared_experts'])
        self.total_experts_number = 0
        if self.shared_experts_number == 0:
            raise RuntimeError("'shared_experts' should has not less than 1 expert")
        for task_name, expert_dict in ple_dict.items():
            if not expert_dict or len(expert_dict) == 0:
                raise RuntimeError("expert_dict should not be none and the size should not be 0, task name is: {}".format(task_name))
            self.total_experts_number += len(expert_dict)
        for layer_id in range(layer_number):
            layer_id_str = str(layer_id)
            self.layerid2block[layer_id_str] = dict()
            self.layerid2gate[layer_id_str] = dict()
            for task_name, expert_dict in ple_dict.items():
                self.layerid2block[layer_id_str][task_name] = dict()
                if 'shared_experts' == task_name:
                    if layer_id < (layer_number - 1) :
                        self.layerid2gate[layer_id_str][task_name] = PLEGate(layer_id, task_name, self.total_experts_number)
                else:
                    self.layerid2gate[layer_id_str][task_name] = PLEGate(layer_id, task_name, self.shared_experts_number + len(expert_dict))
                for expert_id_str, layer_class in expert_dict.items():
                    if not expert_id_str.isdigit() or len(layer_class) == 0 or int(expert_id_str) >= len(expert_dict):
                        raise RuntimeError("expert_id is not digit or layer_class is empty or expert_id is out of range, expert_id is: {}, layer_class is: {}, expert_dict size: {}".format(expert_id_str, layer_class, len(expert_dict)))
                    Expert = dynamic_load_class(self.custom_layer_file_path, layer_class)
                    self.layerid2block[layer_id_str][task_name][expert_id_str] = Expert(layer_id, task_name, int(expert_id_str))
        self.towers = dict() # 塔
        for task_name, layer_class in tower_dict.items():
            if len(layer_class) == 0:
                raise RuntimeError("layer_class is empty, layer_class is: {}".format(layer_class))
            Tower = dynamic_load_class(self.custom_layer_file_path, layer_class)
            self.towers[task_name] = Tower(task_name)

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        if self.inp_dr:
            dropped_inputs = self.inp_dr(inputs, training=training)
        else:
            dropped_inputs = inputs
        task_tensors = dict() # 每个task有一个list的expert
        task_gate = dict() # task_name -> tensor; tensor的每个元素都对应一个expert的门
        task_gated_tensor = dict() # task_name -> tensor; reduce sum of all experts go through their gates
        for layer_id in self.layerid2block:
            if layer_id == '0': # 第一层独立处理逻辑
                #task_tensors: 每个任务自己的专家的输出， task_gate: 每个任务自己的门控的输出,
                task_tensors, task_gate = self._calculate_expert_and_gate(layer_id, task_tensors, task_gate, task_gated_tensor, expert_input=dropped_inputs, gate_input=inputs, is_first_layer=True, is_last_layer=False, is_concat_gate_input=self.is_concat_gate_input)
                for task_name in self.ordered_task_names: # 第二次遍历，把门和每个expert融合
                    #task_gated_tensor，计算每个任务的每一层的最终输出
                    if 'shared_experts' == task_name:
                        if len(layer_id) < self.layer_number - 1:
                            task_gated_tensor = self._shared_experts_merge(layer_id, task_tensors, task_gate, task_gated_tensor)
                    else:
                        task_gated_tensor = self._experts_merge(layer_id, task_name, task_tensors, task_gate, task_gated_tensor)
            elif layer_id == str(len(self.layerid2block) - 1): # 最后一层shared gate是废弃的
                task_tensors, task_gate = self._calculate_expert_and_gate(layer_id, task_tensors, task_gate, task_gated_tensor, is_first_layer=False, is_last_layer=True, is_concat_gate_input=self.is_concat_gate_input)
                for task_name in self.ordered_task_names: # 第二次遍历，把门和每个expert融合
                    if 'shared_experts' != task_name:
                        task_gated_tensor = self._experts_merge(layer_id, task_name, task_tensors, task_gate, task_gated_tensor)
            else:
                task_tensors, task_gate = self._calculate_expert_and_gate(layer_id, task_tensors, task_gate, task_gated_tensor, is_concat_gate_input=self.is_concat_gate_input)
                for task_name in self.ordered_task_names: # 第二次遍历，把门和每个expert融合
                    if 'shared_experts' == task_name:
                        task_gated_tensor = self._shared_experts_merge(layer_id, task_tensors, task_gate, task_gated_tensor)
                    else:
                        task_gated_tensor = self._experts_merge(layer_id, task_name, task_tensors, task_gate, task_gated_tensor)
        tower_result = dict()
        for task_name in [tn for tn in self.ordered_task_names if tn != 'shared_experts']:
            tower_result[task_name] = self.towers[task_name](task_gated_tensor[task_name])
        for task_main, task_parent in self.tower_dependencies_dict.items(): # task_main = task_main * task_parent，当两者都在0～1之间的时候，task_main必然小于task_parent
            tower_result[task_main] *= tower_result[task_parent]
        return tower_result

    #计算专家和门, task_tensors: 每个task的专家网络的输出（因为可能存在多个expert）， task_gate: 计算每个任务的gate的输出, is_concat_gate_input: 非第一个cgc层 的门控网络输入（是否拼接所有的任务的expert输入）
    def _calculate_expert_and_gate(self, layer_id_str, task_tensors, task_gate, task_gated_tensor, expert_input=None, gate_input=None, is_first_layer=False, is_last_layer=False, is_concat_gate_input=True):
        for task_name in self.ordered_task_names: # 固定顺序，为门跟expert的固定对应做准备，首次遍历把门和expert都算好
            if not is_first_layer: # 第一层直接用输入，其他层用上一层的输出作为输入
                expert_input = task_gated_tensor[task_name] #每个任务自己的expert的输入
                if is_concat_gate_input: #所有的任务的expert的输入拼接起来作为该任务门的输入（非第一层的情况）
                    gated_tensors = [task_gated_tensor[order_tn] for order_tn in self.ordered_task_names] #全部task的上层输入都参与每个门的计算
                    gate_input = tf.concat(gated_tensors, axis=-1) #全部task的上层输入都参与每个门的计算
                else:
                    gate_input = task_gated_tensor[task_name] #分开task作为gate的输入
            for expert_id in range(len(self.layerid2block[layer_id_str][task_name])): # 计算每个专家网络的输出
                expert_id_str = str(expert_id)
                if task_name not in task_tensors:
                    task_tensors[task_name] = [self.layerid2block[layer_id_str][task_name][expert_id_str](expert_input)]
                else:
                    task_tensors[task_name].append(self.layerid2block[layer_id_str][task_name][expert_id_str](expert_input))
            if is_last_layer: # 最后一层的情况
                if 'shared_experts' != task_name:
                    task_gate[task_name] = self.layerid2gate[layer_id_str][task_name](gate_input) # 计算每个task的门
            else: # 非最后一层的情况
                layer_id = int(layer_id_str)
                gate_support = True
                if 'shared_experts' == task_name and layer_id == (self.layer_number - 1):
                    gate_support = False
                if gate_support:
                    task_gate[task_name] = self.layerid2gate[layer_id_str][task_name](gate_input) # 计算每个task的门
        return task_tensors, task_gate

    def _experts_merge(self, layer_id_str, task_name, task_tensors, task_gate, task_gated_tensor):
        gate_idx = 0
        # 先处理当前的task
        for expert_id in range(len(self.layerid2block[layer_id_str][task_name])): # 固定顺序
            if task_name not in task_gated_tensor: # 第一个直接赋值
                task_gated_tensor[task_name] = task_gate[task_name][:, gate_idx: gate_idx + 1] * task_tensors[task_name][expert_id]
            else: # 后面的直接累加进去
                task_gated_tensor[task_name] += task_gate[task_name][:, gate_idx: gate_idx + 1] * task_tensors[task_name][expert_id]
            gate_idx += 1
        # 后处理shared block
        for expert_id in range(len(self.layerid2block[layer_id_str]['shared_experts'])): # 固定顺序 这里肯定不是第一个了，所以直接进行累加
            task_gated_tensor[task_name] += task_gate[task_name][:, gate_idx: gate_idx + 1] * task_tensors['shared_experts'][expert_id]
            gate_idx += 1
        return task_gated_tensor

    def _shared_experts_merge(self, layer_id_str, task_tensors, task_gate, task_gated_tensor):
        gate_idx = 0
        for task_name1 in self.ordered_task_names: # shared block需要取全部的experts作为输入，并且固定顺序
            for expert_id in range(len(self.layerid2block[layer_id_str][task_name1])): # 固定顺序
                if 'shared_experts' not in task_gated_tensor: # 第一个直接赋值
                    task_gated_tensor['shared_experts'] = task_gate['shared_experts'][:, gate_idx: gate_idx + 1] * task_tensors[task_name1][expert_id]
                else: # 后面的直接累加进去
                    task_gated_tensor['shared_experts'] += task_gate['shared_experts'][:, gate_idx: gate_idx + 1] * task_tensors[task_name1][expert_id]
                gate_idx += 1
        return task_gated_tensor

    def get_config(self):
        config = super(PLELayer, self).get_config()
        config.update({
            'custom_layer_file_path': self.custom_layer_file_path,
            'ordered_task_names': self.ordered_task_names,
            'layer_number': self.layer_number,
            'ple_dict': self.ple_dict,
            'tower_dict': self.tower_dict,
            'tower_dependencies_dict': self.tower_dependencies_dict,
            'use_inputs_dropout': self.use_inputs_dropout,
            'dropout_layer': self.dropout_layer,
            'is_concat_gate_input': self.is_concat_gate_input,
            'name': self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PLEGate(tf.keras.layers.Layer):
    def __init__(self, layer_id, task_name, selected_vectors_number, name="ple_gate", **kwargs):
        super(PLEGate, self).__init__(name=name, trainable=True, **kwargs)
        self.layer_id = layer_id
        self.task_name = task_name
        self.selected_vectors_number = selected_vectors_number
        self.seq1 = tf.keras.layers.Dense(selected_vectors_number, name='L%d_T%s_gate_dense' % (layer_id, task_name)) # 每个专家就是一个vector，第一层的gate的输入是模型输入，第二层开始后面的gate的输入是前一条链路的输出结果（不要融合全部链路）
        self.seq2 = tf.keras.layers.Softmax(name='L%d_T%s_gate_softmax' % (layer_id, task_name))

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        y = inputs
        y = self.seq1(y)
        y = self.seq2(y)
        return y

    def get_config(self):
        config = super(PLEGate, self).get_config()
        config.update({
            'layer_id': self.layer_id,
            'task_name': self.task_name,
            'selected_vectors_number': self.selected_vectors_number,
            'name': self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)