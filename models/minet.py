from abc import ABC

from pkgs.tf.feature_util import *
from pkgs.tf.extend_layers import ModelInputLayer, MMoELayer, DNNLayer, LocalActivationUnit, AttentionLayer, CINLayer, ItemAttentionLayer, InterestAttentionLayer
from pkgs.tf.helperfuncs import TF_REF_VERSION
from pkgs.tf.helperfuncs import create_activate_func
from collections import defaultdict

class MiNetModel(tf.keras.models.Model, ABC):
    def __init__(self, model_input_config: ModelInputConfig, attention_dim, width, height, attention_act='relu', attention_use_bias=True, use_transfer=False, source_task_structs=None, source_hidden_act=None,
                 source_output_act=None, source_use_bias=True, target_task_structs=None, target_hidden_act=None, target_output_act=None, target_use_bias=True, source_task_dropout=None,
                 target_task_dropout=None, use_source_interest=False, name='minet'):
        super(MiNetModel, self).__init__(name=name)

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

        attention_use_bias_chk = __normalize_task_dnn_args(attention_use_bias, 'attention_use_bias', (bool, list))
        attention_act_chk = __normalize_task_dnn_args(attention_act, 'attention_act', (str, list))
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

        #是否使用源域的序列特征
        if use_transfer:
            self.source_item_attention_layer = ItemAttentionLayer(attention_dim, width, height, attention_act=attention_act_chk, use_bias=attention_use_bias_chk, use_transfer=use_transfer)

        self.target_item_attention_layer = ItemAttentionLayer(attention_dim, width, height, attention_act=attention_act,
                                                              use_bias=attention_act_chk, use_transfer=use_transfer)


        self.interest_attention_layer = InterestAttentionLayer(attention_dim, attention_act=attention_act, use_bias=attention_use_bias_chk, use_source_interest=use_source_interest)

        self.target_task_layer = DNNLayer(target_task_structs, target_hidden_act_chk, target_output_act_chk, target_task_dropout, use_bias=target_use_bias_chk, name='target_task_layer')

        self.source_task_layer = DNNLayer(source_task_structs, source_hidden_act_chk, source_output_act_chk, source_task_dropout, use_bias=source_use_bias_chk, name='source_task_layer')

        self.label_names = [i.name for i in model_input_config.all_inputs if i.is_label]
        print("label_names={}".format(self.label_names))

        self.model_input_config = model_input_config
        self.attention_dim = attention_dim
        self.width = width
        self.height = height
        self.attention_act = attention_act_chk
        self.attention_use_bias = attention_use_bias_chk
        self.use_transfer = use_transfer
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
        self.use_source_interest = use_source_interest
    @tf.function
    def call(self, inputs, training=None, mask=None):
        user_feat_vals = []
        transformed_user_inputs = self.input_layer(inputs, groups='user')
        for val in transformed_user_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            user_feat_vals.append(val)
        user_concat_vals = tf.concat(user_feat_vals, axis=-1, name='concat_user_features')

        item_feat_vals = []
        transformed_item_inputs = self.input_layer(inputs, groups='t_item')
        for val in transformed_item_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            item_feat_vals.append(val)
        item_concat_vals = tf.concat(item_feat_vals, axis=-1, name='concat_item_features')

        target_feat_vals = []
        transformed_target_inputs = self.input_layer(inputs, groups='target')
        for val in transformed_target_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            target_feat_vals.append(val)
        target_concat_vals = tf.concat(target_feat_vals, axis=-1, name='concat_target_features')

        target_hist_len = self.input_layer.compute_mask_histlen(inputs, name='target_seq_list',
                                                                return_seqlist=True)
        target_attention_output = self.target_item_attention_layer(
            [target_concat_vals, item_concat_vals, user_concat_vals, target_hist_len])

        source_attention_output = None
        if self.use_transfer:
            source_feat_vals = []
            transformed_source_inputs = self.input_layer(inputs, groups='source')
            for val in transformed_source_inputs.values():
                val = tf.keras.layers.Flatten()(val)
                source_feat_vals.append(val)
            source_concat_vals = tf.concat(source_feat_vals, axis=-1, name='concat_source_features')

            hist_len = self.input_layer.compute_mask_histlen(inputs, name='source_seq_list',
                                                             return_seqlist=True)

            source_attention_output = self.source_item_attention_layer([source_concat_vals, item_concat_vals, user_concat_vals, hist_len])
            final_target_input = self.interest_attention_layer([item_concat_vals, user_concat_vals, source_attention_output, target_attention_output])
        else:
            final_target_input = self.interest_attention_layer(
                [item_concat_vals, user_concat_vals, target_attention_output])

        target_output = self.target_task_layer(final_target_input) #目标域的任务输出
        source_item_feat_vals = []
        transformed_source_item_inputs = self.input_layer(inputs, groups='s_item')
        for val in transformed_source_item_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            source_item_feat_vals.append(val)
        source_item_concat_vals = tf.concat(source_item_feat_vals, axis=-1, name='concat_source_item_features')
        final_source_input = tf.concat([user_concat_vals, source_item_concat_vals], -1)

        source_output = self.source_task_layer(final_source_input) # 源域的任务输出


        outputs = []
        outputs.append(target_output)
        outputs.append(source_output)
        return tuple(outputs)

    def get_config(self):
        config = {
            'model_input_config': self.model_input_config,
            'attention_dim': self.attention_dim,
            'width': self.width,
            'height': self.height,
            'attention_act': self.attention_act,
            'attention_use_bias': self.attention_use_bias,
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
            'use_source_interest': self.use_source_interest,
            'name': self.name
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
