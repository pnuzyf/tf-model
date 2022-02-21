'''
Author: fabriszhou
des: multi-scenario multi-task model
'''

from abc import ABC

from pkgs.tf.feature_util import *
from pkgs.tf.extend_layers import ModelInputLayer, MMoELayerV2, LocalActivationUnit, AttentionLayer, PLELayer, DNNLayer, MemoryLayer
from pkgs.tf.helperfuncs import TF_REF_VERSION
from pkgs.tf.helperfuncs import create_activate_func
from pkgs.utils import dynamic_load_class
from collections import defaultdict


#num_scenarios: moe层expert数量 share_gates: 控制moe层是否共享gates  is_parallel: True 并行结构（moe同plce是并行的）， 否则为串行结构（moe的输出作为ple的输入）
# short_cut：主要用于并行结构， 通过一个浅层tower来提权输入层的特征， 同 moe侧的输出进行concat， cut_tower: cut tower的结构
#is_memory: 是否支持memory网络，通过该网络拿到长期兴趣表征和短期兴趣表征，来辅助冷启动
class MsMtModel(tf.keras.Model, ABC):
    def __init__(self, model_input_config: ModelInputConfig, num_scenarios, moe_num_experts, moe_expert_layers, ordered_task_names, ple_layer_number, ple_dict, tower_dict, task_output_act,
                 tower_dependencies_dict, custom_layer_file_path, expert_use_bias=True, expert_act='relu', expert_dropout=None, expert_use_bn=False, expert_l1_reg=None, expert_l2_reg=None,
                 gate_use_bias=True, gate_l1_reg=None, gate_l2_reg=None, named_outputs=None, is_concat_gate_input=False, use_inputs_dropout=False, dropout_layer=None, is_parallel=True,
                 short_cut=False, cut_tower=None, cut_tower_act='relu', is_memory=False, controller_layers=None, controller_hidden_act='relu', controller_output_act=None, n_clusters=100, key_dims=8,
                 long_dims=8, short_dims=8, temperature=0.1, alpha=0.1, is_short=False, name='msmt'):
        super(MsMtModel, self).__init__(name=name)

        if not tower_dict or type(tower_dict) != type(dict()) or len(tower_dict) + 1 != len(ple_dict):
            raise RuntimeError(
                "'tower_dict' should not be none and should be dict type and towers size is tasks number")
        assert len(tower_dict) >= 2, "'tower_dict' should have not less than 2 tasks, got {} tasks" \
            .format(len(tower_dict))
        tns = [k for k, _ in ple_dict.items()]
        if len(ordered_task_names) + 1 != len(ple_dict) or any(
                [task_name not in tns for task_name in ordered_task_names]):
            raise RuntimeError("'ordered_task_names' should has equal task names as 'ple_dict' except 'shared_experts'")
        tns2 = [k for k, _ in tower_dict.items()]
        if any([task_name not in tns2 for task_name in ordered_task_names]):
            raise RuntimeError("'ordered_task_names' should has equal task names as 'tower_dict'")
        for task_main, task_parent in tower_dependencies_dict.items():
            if task_main not in ordered_task_names:
                raise RuntimeError(
                    "task_main {} is not in rodered_task_names from tower_dependencies_dict".format(task_main))
            if task_parent not in ordered_task_names:
                raise RuntimeError(
                    "task_parent {} is not in rodered_task_names from tower_dependencies_dict".format(task_parent))


        self.is_first = True
        self.input_layer = ModelInputLayer(model_input_config, auto_create_embedding=True) #input and embedding layer

        self.num_scenarios = num_scenarios #场景个数
        self.moe_layer = MMoELayerV2(num_scenarios, moe_num_experts, moe_expert_layers, expert_use_bias, expert_act,
                                    expert_dropout, expert_use_bn, expert_l1_reg, expert_l2_reg,
                                    gate_use_bias, gate_l1_reg, gate_l2_reg)

        self.ple_layer = PLELayer(custom_layer_file_path, ordered_task_names, ple_layer_number, ple_dict, use_inputs_dropout, dropout_layer, is_concat_gate_input)

        #tower layer
        self.towers = dict()  # 塔
        for task_name, layer_class in tower_dict.items():
            if len(layer_class) == 0:
                raise RuntimeError("layer_class is empty, layer_class is: {}".format(layer_class))
            Tower = dynamic_load_class(custom_layer_file_path, layer_class)
            self.towers[task_name] = Tower(task_name)

        # Attention层的核心使用的是LocalActivationUnit-- 这里可以通过配置
        self.attention_layer = {}  # 这里可能存在多个
        for input_config in model_input_config.all_inputs:
            name = input_config.name
            if input_config.query:
                self.attention_layer[name] = AttentionLayer(
                    LocalActivationUnit(hidden_units=[36, 1], activations=['dice'], mode='SUM',
                                        normalization=True)
                )

        # 输出层单独
        self.task_active_layers = []
        for i in range(len(ordered_task_names)):
            self.task_active_layers.append(
                tf.keras.layers.Activation(
                    create_activate_func(task_output_act[i]),
                    name=self.name + "/output_act_{}_task{}".format(task_output_act[i], i)
                ))  # 输出层单独

        #short_cut layer:
        if short_cut:
            self.cut_layer = DNNLayer(cut_tower, cut_tower_act, None,
                                                  name='cut_tower')

        #memory layer
        if is_memory:
            self.memory = MemoryLayer(controller_layers, controller_hidden_act, controller_output_act, n_clusters, key_dims, long_dims, short_dims, temperature, alpha, is_short)

        self.label_names = [i.name for i in model_input_config.all_inputs if i.is_label]
        print("label_names={}".format(self.label_names))
        if named_outputs is None:
            named_outputs = len(self.label_names) > 1
        print("named_outputs={}".format(named_outputs))

        self.model_input_config = model_input_config
        self.num_scenarios = num_scenarios
        self.moe_num_experts = moe_expert_layers
        self.moe_expert_layers = moe_expert_layers
        self.is_concat_gate_input = is_concat_gate_input
        self.custom_layer_file_path = custom_layer_file_path
        self.ordered_task_names = ordered_task_names
        self.ple_layer_number = ple_layer_number
        self.ple_dict = ple_dict
        self.tower_dict = tower_dict
        self.task_output_act = task_output_act
        self.tower_dependencies_dict = tower_dependencies_dict
        self.dropout_layer = dropout_layer
        self.use_inputs_dropout = use_inputs_dropout
        self.expert_use_bias = expert_use_bias
        self.expert_act = expert_act
        self.expert_dropout = expert_dropout
        self.expert_use_bn = expert_use_bn
        self.expert_l1_reg = expert_l1_reg
        self.expert_l2_reg = expert_l2_reg
        self.gate_use_bias = gate_use_bias
        self.gate_l1_reg = gate_l1_reg
        self.gate_l2_reg = gate_l2_reg
        self.is_parallel = is_parallel
        self.short_cut = short_cut
        self.cut_tower = cut_tower
        self.cut_tower_act = cut_tower_act

        self.is_memory = is_memory
        self.controller_layers = controller_layers
        self.controller_hidden_act = controller_hidden_act
        self.controller_output_act = controller_output_act
        self.n_clusters = n_clusters
        self.key_dims = key_dims
        self.long_dims = long_dims
        self.short_dims = short_dims
        self.temperature = temperature
        self.alpha = alpha
        self.is_short = is_short

        self.named_outputs = named_outputs

        if tf.__version__ < TF_REF_VERSION:
            self.mixed_precision = tf.keras.mixed_precision.experimental.get_layer_policy(self).loss_scale is not None
        else:
            self.mixed_precision = tf.keras.mixed_precision.global_policy().compute_dtype == tf.float16
        print("{}: mixed_precision={}".format(self.name, self.mixed_precision))


    @tf.function
    def call(self, inputs, training=None, mask=None):
        # 门控的输入
        gate_vals = []

        #正常特征
        feat_vals = []
        gate_inputs = self.input_layer(inputs, groups="gate")  # 获取指定组的特征
        for input_config in self.model_input_config.get_inputs_by_group("gate"):
            if input_config.is_gate:
                name = input_config.name
                m = gate_inputs[name]  # 特征
                val = tf.keras.layers.Flatten()(m)
                gate_vals.append(val)
        gate_concat_vals = tf.concat(gate_vals, axis=-1, name='concat_gate_features')


        query_feats = set()
        if self.is_memory:
            for input_config in self.model_input_config.get_inputs_by_group("query"):
                query_feats.add(input_config.name)
        memory_query_vals = [] #memory的 query
        isActive_vals = None #样本是否是高活用户样本 1 高活
        long_vals = []
        short_vals = []
        transformed_inputs = self.input_layer(inputs, groups='dnn') #获取指定组的特征
        for input_config in self.model_input_config.get_inputs_by_group('dnn'):
            if input_config.is_weight_col or input_config.is_label:
                continue

            name = input_config.name
            m = transformed_inputs[name]  # 特征
            if name == "isActive":
                isActive_vals = m
                continue
            if input_config.other_keys is not None:
                for key in input_config.other_keys:
                    m1 = transformed_inputs[key]  # key
                    m = tf.concat([m, m1], axis=-1)  # concat
            if input_config.query is not None:
                query_vals = []
                for query in input_config.query:
                    q = transformed_inputs[query] #query
                    query_vals.append(q)
                q = tf.concat(query_vals, axis=-1, name='concat_query')
                # 拿到attention score
                hist_len = self.input_layer.compute_mask_histlen(inputs, name,
                                                                 return_seqlist=True)
                m = self.attention_layer[name]([q, m, m, hist_len])  # [batch_size, 1, dim]
                m = tf.squeeze(m, axis=1)

            val = tf.keras.layers.Flatten()(m)

            #是否是长期兴趣
            if input_config.is_long:
                long_vals.append(val)

            if input_config.is_short:
                short_vals.append((val))

            if name in query_feats:
                memory_query_vals.append(val)

            if not val.dtype.is_integer and not val.dtype.is_floating and not val.dtype.is_bool:
                raise RuntimeError("dtype of input '{}' is {}, only float/int/bool are allowed"
                                   .format(name, val.dtype))
            elif val.dtype != self._dtype_policy.compute_dtype:
                val = tf.cast(val, self._dtype_policy.compute_dtype,
                              name=name + '_cast2' + self._dtype_policy.compute_dtype)

            feat_vals.append(val) #插入特征进入对应的侧

        #是否需要memory网络
        if self.is_memory:
            query_emb = tf.concat(memory_query_vals, axis=-1, name="concat_memory_query") #查询向量
            long_emb = tf.concat(long_vals, axis=-1, name='concat_memory_long') #查询
            long_cluster = self.memory([query_emb, long_emb, isActive_vals], training=training) #拿到长期的兴趣表征
            feat_vals.append(long_cluster)
        concat_vals = tf.concat(feat_vals, axis=-1, name='concat_features') #正常网络输入层

        moe_output = self.moe_layer([gate_concat_vals, concat_vals], training=training) #moe的输出

        ple_outputs = dict()

        # 并行结构
        if self.is_parallel:
            dnn_output = self.ple_layer(concat_vals, training=training)  # 这里只有ple的输出，不经过tower

            for task_name in [tn for tn in self.ordered_task_names]:
                ple_outputs[task_name] = tf.concat([moe_output, dnn_output[task_name]], axis=-1,
                                               name='concat_task_{}_features'.format(task_name))
        else:
            if self.short_cut:
                cut_outputs = self.cut_layer(concat_vals)  # 近路的输出
                ple_input = tf.concat([moe_output, cut_outputs], axis=-1, name='ple_input_with_cut')
                ple_outputs = self.ple_layer(ple_input, training=training)  # 这里只有ple的输出，不经过tower

            else:
                ple_outputs = self.ple_layer(moe_output, training=training)  # 这里只有ple的输出，不经过tower


        tower_result = dict()
        #是否拼接moe的输出
        for task_name in [tn for tn in self.ordered_task_names]:
            tower_result[task_name] = self.towers[task_name](ple_outputs[task_name])

        for task_main, task_parent in self.tower_dependencies_dict.items():  # task_main = task_main * task_parent，当两者都在0～1之间的时候，task_main必然小于task_parent
            tower_result[task_main] *= tower_result[task_parent]

        task_outputs = []
        out_i = 0

        for task_name, task_output in tower_result.items():  # label_name need to be equal to task name
            if self.mixed_precision and task_output.dtype != tf.float32:
                task_output = tf.cast(task_output, tf.float32, self.name + "_mp_output_{}_cast2float32".format(out_i))
            task_outputs.append(task_output)
            out_i += 1

        idx = 0
        if self.named_outputs:
            output_dict = {}
            for name, output, task_act_layer in zip(self.label_names, task_outputs, self.task_active_layers):
                if task_act_layer:
                    output_dict[name] = task_act_layer(output)
                else:
                    output_dict[name] = tf.nn.sigmoid(output)
                idx += 1
            return output_dict
        else:
            outputs = []
            for output, task_act_layer in zip(task_outputs, self.task_active_layers):
                # 如果设置output_cross为0, 则将mmoe的task_layer的输出和wide侧模型的输出进行加和, 再经过sigmoid得到最终输出
                if task_act_layer:
                    outputs.append(task_act_layer(output))
                else:
                    outputs.append(tf.nn.sigmoid(output))
                idx += 1
            return tuple(outputs)

    def get_train_model(self):

        inputs = {}
        gate_inputs_dict = {}
        dnn_inputs_dict = {}
        for i_desc in self.input_layer.get_feature_input_descs("gate"):
            gate_inputs_dict[i_desc.name] = i_desc.to_tf_input()

        for i_desc in self.input_layer.get_feature_input_descs("dnn"):
            dnn_inputs_dict[i_desc.name] = i_desc.to_tf_input()

        query_feats = set()
        if self.is_memory:
            for input_config in self.model_input_config.get_inputs_by_group("query"):
                query_feats.add(input_config.name)

        gate_vals = []

        # 正常特征
        feat_vals = []

        memory_query_vals = []  # memory的 query
        isActive_vals = None #样本是否是高活用户样本 1 高活
        long_vals = []
        short_vals = []
        transformed_inputs = self.input_layer(dnn_inputs_dict, groups="dnn")  # 获取指定组的特征

        gate_inputs = self.input_layer(gate_inputs_dict, groups="gate")  # 获取指定组的特征
        for input_config in self.model_input_config.get_inputs_by_group("gate"):
            if input_config.is_gate:
                name = input_config.name
                m = gate_inputs[name]  # 特征
                val = tf.keras.layers.Flatten()(m)
                gate_vals.append(val)
        gate_concat_vals = tf.concat(gate_vals, axis=-1, name='concat_gate_features')

        for input_config in self.model_input_config.get_inputs_by_group('dnn'):
            if input_config.is_weight_col or input_config.is_label:
                continue

            name = input_config.name
            m = transformed_inputs[name]  # 特征
            if name == "isActive":
                isActive_vals = m
                continue
            if input_config.other_keys is not None:
                for key in input_config.other_keys:
                    m1 = transformed_inputs[key]  # key
                    m = tf.concat([m, m1], axis=-1)  # concat
            if input_config.query is not None:
                query_vals = []
                for query in input_config.query:
                    q = transformed_inputs[query]  # query
                    query_vals.append(q)
                q = tf.concat(query_vals, axis=-1, name='concat_query')
                # 拿到attention score
                hist_len = self.input_layer.compute_mask_histlen(dnn_inputs_dict, name,
                                                                 return_seqlist=True)
                m = self.attention_layer[name]([q, m, m, hist_len])  # [batch_size, 1, dim]
                m = tf.squeeze(m, axis=1)

            val = tf.keras.layers.Flatten()(m)

            # 是否是长期兴趣
            if input_config.is_long:
                long_vals.append(val)

            if input_config.is_short:
                short_vals.append((val))

            if name in query_feats:
                memory_query_vals.append(val)

            if not val.dtype.is_integer and not val.dtype.is_floating and not val.dtype.is_bool:
                raise RuntimeError("dtype of input '{}' is {}, only float/int/bool are allowed"
                                   .format(name, val.dtype))
            elif val.dtype != self._dtype_policy.compute_dtype:
                val = tf.cast(val, self._dtype_policy.compute_dtype,
                              name=name + '_cast2' + self._dtype_policy.compute_dtype)

            feat_vals.append(val)  # 插入特征进入对应的侧

        # 是否需要memory网络
        if self.is_memory:
            query_emb = tf.concat(memory_query_vals, axis=-1, name="concat_memory_query")  # 查询向量
            long_emb = tf.concat(long_vals, axis=-1, name='concat_memory_long') #查询
            long_cluster = self.memory.predict_process([query_emb, long_emb, isActive_vals])  # 拿到长期的兴趣表征

            feat_vals.append(long_cluster)
        concat_vals = tf.concat(feat_vals, axis=-1, name='concat_features')  # 正常网络输入层

        moe_output = self.moe_layer([gate_concat_vals, concat_vals])  # moe的输出

        ple_outputs = dict()

        # 并行结构
        if self.is_parallel:
            dnn_output = self.ple_layer(concat_vals)  # 这里只有ple的输出，不经过tower

            for task_name in [tn for tn in self.ordered_task_names]:
                ple_outputs[task_name] = tf.concat([moe_output, dnn_output[task_name]], axis=-1,
                                                   name='concat_task_{}_features'.format(task_name))
        else:
            if self.short_cut:
                cut_outputs = self.cut_layer(concat_vals)  # 近路的输出
                ple_input = tf.concat([moe_output, cut_outputs], axis=-1, name='ple_input_with_cut')
                ple_outputs = self.ple_layer(ple_input)  # 这里只有ple的输出，不经过tower

            else:
                ple_outputs = self.ple_layer(moe_output)  # 这里只有ple的输出，不经过tower

        tower_result = dict()
        # 是否拼接moe的输出
        for task_name in [tn for tn in self.ordered_task_names]:
            tower_result[task_name] = self.towers[task_name](ple_outputs[task_name])

        for task_main, task_parent in self.tower_dependencies_dict.items():  # task_main = task_main * task_parent，当两者都在0～1之间的时候，task_main必然小于task_parent
            tower_result[task_main] *= tower_result[task_parent]

        task_outputs = []
        out_i = 0

        for task_name, task_output in tower_result.items():  # label_name need to be equal to task name
            if self.mixed_precision and task_output.dtype != tf.float32:
                task_output = tf.cast(task_output, tf.float32, self.name + "_mp_output_{}_cast2float32".format(out_i))
            task_outputs.append(task_output)
            out_i += 1

        idx = 0
        inputs.update(gate_inputs_dict)
        inputs.update(dnn_inputs_dict)
        if self.named_outputs:
            output_dict = {}
            for name, output, task_act_layer in zip(self.label_names, task_outputs, self.task_active_layers):
                if task_act_layer:
                    output_dict[name] = task_act_layer(output)
                else:
                    output_dict[name] = tf.nn.sigmoid(output)
                idx += 1
            return tf.keras.Model(inputs, outputs=output_dict, name=self.name + "-ctr_model")

        else:
            outputs = []
            for output, task_act_layer in zip(task_outputs, self.task_active_layers):
                # 如果设置output_cross为0, 则将mmoe的task_layer的输出和wide侧模型的输出进行加和, 再经过sigmoid得到最终输出
                if task_act_layer:
                    outputs.append(task_act_layer(output))
                else:
                    outputs.append(tf.nn.sigmoid(output))
                idx += 1
            return tf.keras.Model(inputs, outputs=tuple(outputs), name=self.name + "-ctr_model")

    def get_config(self):
        config = {
            'model_input_config': self.model_input_config,
            'num_scenarios': self.num_scenarios,
            'moe_num_experts': self.moe_num_experts,
            'moe_expert_layers': self.moe_expert_layers,
            'custom_layer_file_path': self.custom_layer_file_path,
            'ordered_task_names': self.ordered_task_names,
            'layer_number': self.ple_layer_number,
            'ple_dict': self.ple_dict,
            'tower_dict': self.tower_dict,
            'task_output_act': self.task_output_act,
            'tower_dependencies_dict': self.tower_dependencies_dict,
            'dropout_layer': self.dropout_layer,
            'use_inputs_dropout': self.use_inputs_dropout,
            'is_concat_gate_input': self.is_concat_gate_input,
            'expert_use_bias': self.expert_use_bias,
            'expert_act': self.expert_act,
            'expert_dropout': self.expert_dropout,
            'expert_use_bn':  self.expert_use_bn,
            'expert_l1_reg':  self.expert_l1_reg,
            'expert_l2_reg': self.expert_l2_reg,
            'gate_use_bias': self.gate_use_bias,
            'gate_l1_reg': self.gate_l1_reg,
            'gate_l2_reg': self.gate_l2_reg,
            'is_parallel': self.is_parallel,
            'namded_outputs': self.named_outputs,
            'is_memory': self.is_memory,
            'controller_layers': self.controller_layers,
            'controller_hidden_act': self.controller_hidden_act,
            'controller_output_act': self.controller_output_act,
            'n_clusters': self.n_clusters,
            'key_dims': self.key_dims,
            'long_dims': self.long_dims,
            'short_dims': self.short_dims,
            'temperature': self.temperature,
            'alpha': self.alpha,
            'is_short': self.is_short,
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
