from abc import ABC

from pkgs.tf.feature_util import *
from pkgs.tf.extend_layers import ModelInputLayer, MMoELayer, DNNLayer, LocalActivationUnit, AttentionLayer, CINLayer
from pkgs.tf.helperfuncs import TF_REF_VERSION
from pkgs.tf.helperfuncs import create_activate_func
from collections import defaultdict

class CrossNetModel(tf.keras.models.Model, ABC):
    def __init__(self, model_input_config: ModelInputConfig, source_task_structs=None, source_hidden_act=None, source_output_act=None,
                 source_use_bias=True, target_task_structs=None, target_hidden_act=None, target_output_act=None, target_use_bias=True, source_task_dropout=None,
                 target_task_dropout=None, use_mmoe=False, num_experts=None, expert_layers=None, expert_use_bias=True, expert_act='relu',
                 expert_dropout=None, gate_use_bias=True, name='crossnet'):
        super(CrossNetModel, self).__init__(name=name)

        #先校验source任务和target任务是否已经配置
        if not isinstance(source_task_structs, (list, tuple)) or not source_task_structs:
            raise RuntimeError("'source_task_structs' should be a non-empty list of list, got '{}': {}"
                               .format(type(source_task_structs), source_task_structs))

        for i, ts in enumerate(source_task_structs):
            if not ts or not all([isinstance(i, int) and i > 0 for i in ts]):
                raise RuntimeError("{}th task struct is invalid: {}".format(i, ts))

        if not isinstance(target_task_structs, (list, tuple)) or not target_task_structs:
            raise RuntimeError("'source_task_structs' should be a non-empty list of list, got '{}': {}"
                               .format(type(target_task_structs), target_task_structs))

        for i, ts in enumerate(target_task_structs):
            if not ts or not all([isinstance(i, int) and i > 0 for i in ts]):
                raise RuntimeError("{}th task struct is invalid: {}".format(i, ts))

        def __normalize_task_dnn_args(args, args_name, arg_types):
            if isinstance(args, (list, tuple)):
                # if len(args) != self.num_tasks:
                #     raise RuntimeError("#{} != #task {}".format(args_name, len(args), self.num_tasks))
                if not all([isinstance(a, arg_types) for a in args]):
                    raise RuntimeError("'{}' should be list of {}, got: {}".format(args_name, arg_types, args))
                args = [a.strip() if isinstance(a, str) else a for a in args]
                args = [None if isinstance(i, str) and (not i or i.lower() == 'none') else i for i in args]
            elif isinstance(args, arg_types) or args is None:
                if isinstance(args, str):
                    args = args.strip()
                    args = None if not args or args.lower() == 'none' else args
            else:
                raise RuntimeError("'{}' should be a {}/list of {}, got '{}': {}"
                                   .format(args_name, arg_types, arg_types, type(args), args))

            print("processed {}={}".format(args_name, args))
            return args

        source_hidden_act_chk = __normalize_task_dnn_args(source_hidden_act, 'source_hidden_act', str)
        source_output_act_chk = __normalize_task_dnn_args(source_output_act, 'source_output_act', str)
        source_use_bias_chk = __normalize_task_dnn_args(source_use_bias, 'source_use_bias', (bool, list))
        target_hidden_act_chk = __normalize_task_dnn_args(target_hidden_act, 'target_hidden_act', str)
        target_output_act_chk = __normalize_task_dnn_args(target_output_act, 'target_output_act', str)
        target_use_bias_chk = __normalize_task_dnn_args(target_use_bias, 'target_use_bias', (bool, list))
        source_task_dropout_chk = __normalize_task_dnn_args(source_task_dropout, 'source_task_dropout', (float, list))
        target_task_dropout_chk = __normalize_task_dnn_args(target_task_dropout, 'target_task_dropout', (float, list))

        self.input_layer = ModelInputLayer(model_input_config, auto_create_embedding=True)

        self.source_item_attention_layer = None

        self.target_task_layer = DNNLayer(target_task_structs, target_hidden_act_chk, target_output_act_chk, target_task_dropout, use_bias=target_use_bias_chk, name='target_task_layer')

        self.source_task_layer = DNNLayer(source_task_structs, source_hidden_act_chk, source_output_act_chk, source_task_dropout, use_bias=source_use_bias_chk, name='source_task_layer')

        self.mmoe_layer = None
        if use_mmoe:
            self.mmoe_layer = MMoELayer(2, num_experts, expert_layers, expert_use_bias, expert_act,
                                        expert_dropout, gate_use_bias=gate_use_bias, share_gates=False)


        self.label_names = [i.name for i in model_input_config.all_inputs if i.is_label]
        print("label_names={}".format(self.label_names))

        self.model_input_config = model_input_config
        self.source_task_structs = source_task_structs
        self.source_hidden_act = source_hidden_act_chk
        self.source_output_act = source_output_act_chk
        self.source_use_bias = source_use_bias_chk
        self.target_task_structs = target_task_structs
        self.target_hidden_act = target_hidden_act_chk
        self.target_output_act = target_output_act_chk
        self.target_use_bias = target_use_bias_chk
        self.source_task_dropout = source_task_dropout_chk
        self.target_task_dropout = target_task_dropout_chk
        self.use_mmoe = use_mmoe
        self.num_experts = num_experts
        self.expert_use_bias = expert_use_bias
        self.expert_act = expert_act
        self.expert_dropout = expert_dropout
        self.gate_use_bias = gate_use_bias
    @tf.function
    def call(self, inputs, training=None, mask=None):

        #共享用户侧特征
        user_feat_vals = []
        transformed_user_inputs = self.input_layer(inputs, groups='c_user')
        for val in transformed_user_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            user_feat_vals.append(val)
        user_concat_vals = tf.concat(user_feat_vals, axis=-1, name='concat_user_features')

        #target域 用户侧特征
        target_user_feat_vals = []
        transformed_target_user_inputs = self.input_layer(inputs, groups='t_user')
        for val in transformed_target_user_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            target_user_feat_vals.append(val)
        target_user_concat_vals = tf.concat(target_user_feat_vals, axis=-1, name='concat_target_user_features')


        item_feat_vals = []
        transformed_item_inputs = self.input_layer(inputs, groups='t_item')
        for val in transformed_item_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            item_feat_vals.append(val)
        item_concat_vals = tf.concat(item_feat_vals, axis=-1, name='concat_target_item_features')

        if self.use_mmoe:
            mmoe_outputs = self.mmoe_layer(user_concat_vals, training=training) #mmoe的输出
            final_target_input = tf.concat([mmoe_outputs[0], target_user_concat_vals, item_concat_vals], -1, name='concat_target_features')
        else:
            final_target_input = tf.concat([target_user_concat_vals, item_concat_vals], -1, name='concat_target_features')
        target_output = self.target_task_layer(final_target_input) #目标域的任务输出
        source_item_feat_vals = []
        transformed_source_item_inputs = self.input_layer(inputs, groups='s_item')
        for val in transformed_source_item_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            source_item_feat_vals.append(val)
        source_item_concat_vals = tf.concat(source_item_feat_vals, axis=-1, name='concat_source_item_features')

        if self.use_mmoe:
            final_source_input = tf.concat([mmoe_outputs[1], source_item_concat_vals], -1, name='concat_source_features')
        else:
            final_source_input = tf.concat([user_concat_vals, source_item_concat_vals], -1, name='concat_source_features')

        source_output = self.source_task_layer(final_source_input) # 源域的任务输出


        outputs = []
        outputs.append(target_output)
        outputs.append(source_output)
        return tuple(outputs)

    def get_target_model(self):
        inputs = {}
        groups = ['c_user', 't_user', 't_item']
        feat_vals = []
        for group in groups:
            for i_desc in self.input_layer.get_feature_input_descs(group):
                inputs[i_desc.name] = i_desc.to_tf_input()

            transformed_inputs = self.input_layer(inputs, groups=group)

        for val in transformed_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            feat_vals.append(val)
        concat_vals = tf.concat(feat_vals, axis=-1, name='concat_target_features')
        embeddings = self.target_task_layer(concat_vals)

        return tf.keras.Model(inputs, outputs=embeddings, name=self.name + "-target")

    def get_config(self):
        config = {
            'model_input_config': self.model_input_config,
            'source_task_structs': self.source_task_structs,
            'source_hidden_act': self.source_hidden_act,
            'source_output_act': self.source_output_act,
            'source_use_bias': self.source_use_bias,
            'target_task_structs': self.target_task_structs,
            'target_hidden_act': self.target_hidden_act,
            'target_output_act': self.target_output_act,
            'target_use_bias': self.target_use_bias,
            'source_task_dropout': self.source_task_dropout,
            'target_task_dropout': self.target_task_dropout,
            'use_mmoe': self.mmoe,
            'num_experts': self.num_experts,
            'expert_use_bias': self.expert_use_bias,
            'expert_act': self.expert_act,
            'expert_dropout': self.expert_dropout,
            'gate_use_bias': self.gate_use_bias,
            'name': self.name
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
