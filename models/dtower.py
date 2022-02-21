# coding=utf-8
# @Time     : 2021/4/29 10:43
# @Auther   : lionpeng@tencent.com
from abc import ABC

from pkgs.tf.feature_util import *
from pkgs.tf.extend_layers import ModelInputLayer, DNNLayer
from pkgs.tf.extend_utils import sampling_prob_correction


class TowerLayer(DNNLayer):
    def __init__(self, layer_widthes, hidden_active_fn=None, output_active_fn=None, dropout_prob=None,
                 use_bn=False, l1_reg=None, l2_reg=None, use_bias=True, name="tower_layer", trainable=True,
                 l2_norm=False, **kwargs):
        super(TowerLayer, self).__init__(layer_widthes, hidden_active_fn=hidden_active_fn,
                                         output_active_fn=output_active_fn, dropout_prob=dropout_prob,
                                         use_bn=use_bn, l1_reg=l1_reg, l2_reg=l2_reg, use_bias=use_bias, name=name,
                                         trainable=trainable,
                                         **kwargs)
        self.l2_norm = l2_norm

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        y = super(TowerLayer, self).call(inputs, training=training, **kwargs)
        if self.l2_norm:
            y = tf.nn.l2_normalize(y, axis=-1)
        return y

    def get_config(self):
        config = super(TowerLayer, self).get_config()
        config.update({
            "l2_norm": self.l2_norm
        })
        return config


#model_type: 模型类型 pointwise, pairwise and constrative(对比学习）
class DTowerModel(tf.keras.Model, ABC):
    def __init__(self, model_input_config: ModelInputConfig, user_tower_units=[],
                 user_tower_hidden_act='relu', user_tower_output_act=None, user_tower_dropout=None,
                 user_tower_use_bn=None, user_tower_l1_reg=None, user_tower_l2_reg=None, user_tower_use_bias=False,
                 item_tower_units=[], item_tower_hidden_act='relu', item_tower_output_act=None,
                 item_tower_dropout=None, item_tower_use_bn=None, item_tower_l1_reg=None,
                 item_tower_l2_reg=None, item_tower_use_bias=False, model_type='pointwise', use_cosine=False,
                 temperature=None, correct_bias=False, use_shallow=False, shallow_tower_units=[], fusion_tower=None, fusion_tower_act='relu',
                 fusion_tower_output_act=None, name='dtower'):
        super(DTowerModel, self).__init__(name=name)

        if not isinstance(user_tower_units, (list, tuple)) or not user_tower_units or \
                not all([isinstance(i, int) and i > 0 for i in user_tower_units]):
            raise RuntimeError("'user_tower_units' should be a non-empty postive integer list/tuple, got '{}': {}"
                               .format(type(user_tower_units, user_tower_units)))

        if not isinstance(item_tower_units, (list, tuple)) or not item_tower_units or \
                not all([isinstance(i, int) and i > 0 for i in item_tower_units]):
            raise RuntimeError("'item_tower_units' should be a non-empty postive integer list/tuple, got '{}': {}"
                               .format(type(item_tower_units, item_tower_units)))

        if user_tower_units[-1] != item_tower_units[-1]:
            raise RuntimeError("output dimension size of user and item tower must be the same, got {} and {}"
                               .format(user_tower_units[-1], item_tower_units[-1]))

        self.input_layer = ModelInputLayer(model_input_config, auto_create_embedding=True,
                                           name=self.name + "/input_layer")
        self.user_tower = TowerLayer(user_tower_units, user_tower_hidden_act, user_tower_output_act,
                                     user_tower_dropout, user_tower_use_bn, user_tower_l1_reg,
                                     user_tower_l2_reg, user_tower_use_bias, name=self.name + '/user_tower_dnnlayer',
                                     l2_norm=use_cosine)
        self.item_tower = TowerLayer(item_tower_units, item_tower_hidden_act, item_tower_output_act,
                                     item_tower_dropout, item_tower_use_bn, item_tower_l1_reg,
                                     item_tower_l2_reg, item_tower_use_bias, name=self.name + "/item_tower_dnnlayer",
                                     l2_norm=use_cosine)

        if use_shallow:
            self.user_shallow_tower = TowerLayer(shallow_tower_units, None, user_tower_output_act,
                                     user_tower_dropout, user_tower_use_bn, user_tower_l1_reg,
                                     user_tower_l2_reg, user_tower_use_bias, name=self.name + '/user_shallow_tower_dnnlayer',
                                     l2_norm=use_cosine)
            self.item_shallow_tower = TowerLayer(shallow_tower_units, None, item_tower_output_act,
                                     item_tower_dropout, item_tower_use_bn, item_tower_l1_reg,
                                     item_tower_l2_reg, item_tower_use_bias, name=self.name + '/item_shallow_tower_dnnlayer',
                                     l2_norm=use_cosine)
            self.user_fusion_tower = TowerLayer(shallow_tower_units, None, fusion_tower_output_act,
                                     name=self.name + '/user_fusion_tower_dnnlayer',
                                     l2_norm=use_cosine)
            self.item_fusion_tower = TowerLayer(shallow_tower_units, None, fusion_tower_output_act,
                                                name=self.name + '/item_fusion_tower_dnnlayer',
                                                l2_norm=use_cosine)

        self.model_input_config = model_input_config
        self.user_tower_units = user_tower_units
        self.user_tower_hidden_act = user_tower_hidden_act
        self.user_tower_output_act = user_tower_output_act
        self.user_tower_dropout = user_tower_dropout
        self.user_tower_use_bn = user_tower_use_bn
        self.user_tower_l1_reg = user_tower_l1_reg
        self.user_tower_l2_reg = user_tower_l2_reg
        self.user_tower_use_bias = user_tower_use_bias
        self.item_tower_units = item_tower_units
        self.item_tower_hidden_act = item_tower_hidden_act
        self.item_tower_output_act = item_tower_output_act
        self.item_tower_dropout = item_tower_dropout
        self.item_tower_use_bn = item_tower_use_bn
        self.item_tower_l1_reg = item_tower_l1_reg
        self.item_tower_l2_reg = item_tower_l2_reg
        self.item_tower_use_bias = item_tower_use_bias
        self.model_type = model_type
        self.use_cosine = use_cosine
        self.temperature = temperature
        self.correct_bias = correct_bias
        self.use_shallow = use_shallow
        self.shallow_tower_units = shallow_tower_units
        self.shallow_tower = fusion_tower
        self.shallow_tower_act = fusion_tower_act

    @tf.function
    def call(self, inputs, training=None, mask=None):
        user_feat_vals = []
        transformed_user_inputs = self.input_layer(inputs, groups='user')
        for val in transformed_user_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            user_feat_vals.append(val)
        user_concat_vals = tf.concat(user_feat_vals, axis=-1, name='concat_user_features')

        # [batch, embedding_size]
        user_embeddings = self.user_tower(user_concat_vals, training=training)

        if self.use_shallow:
            # user_shallow_tower
            user_shallow_feat_vals = []
            transformed_user_shallow_inputs = self.input_layer(inputs, groups='u_shallow')
            for val in transformed_user_shallow_inputs.values():
                val = tf.keras.layers.Flatten()(val)
                user_shallow_feat_vals.append(val)
            user_shallow_concat_vals = tf.concat(user_shallow_feat_vals, axis=-1, name='concat_user_shallow_features')
            user_shallow_embeddings = self.user_shallow_tower(user_shallow_concat_vals, training=training)
            user_embeddings = tf.concat([user_embeddings, user_shallow_embeddings], axis=-1, name='concat_final_user_features')

        # if self.use_cosine:
        #     user_embeddings = tf.nn.l2_normalize(user_embeddings, axis=-1)

        if self.model_type == 'pairwise':
            p_item_feat_vals = []
            transformed_p_item_inputs = self.input_layer(inputs, groups='p_item')
            for val in transformed_p_item_inputs.values():
                val = tf.keras.layers.Flatten()(val)
                p_item_feat_vals.append(val)
            p_item_concat_vals = tf.concat(p_item_feat_vals, axis=-1, name='concat_p_item_features')
            # [batch, embedding_size]
            p_item_embeddings = self.item_tower(p_item_concat_vals, training=training)
            # if self.use_cosine:
            #     p_item_embeddings = tf.nn.l2_normalize(p_item_embeddings, axis=-1)

            n_item_feat_vals = []
            transformed_n_item_inputs = self.input_layer(inputs, groups='n_item')
            for val in transformed_n_item_inputs.values():
                val = tf.keras.layers.Flatten()(val)
                n_item_feat_vals.append(val)
            n_item_concat_vals = tf.concat(n_item_feat_vals, axis=-1, name='concat_n_item_features')
            # [batch, embedding_size]
            n_item_embeddings = self.item_tower(n_item_concat_vals, training=training)
            # if self.use_cosine:
            #     n_item_embeddings = tf.nn.l2_normalize(n_item_embeddings, axis=-1)

            # [batch, 1]
            u_pi = tf.reduce_sum(user_embeddings * p_item_embeddings, axis=-1, keepdims=True,
                                 name="u_pi_dotprod")
            # [batch, 1]
            u_ni = tf.reduce_sum(user_embeddings * n_item_embeddings, axis=-1, keepdims=True,
                                 name="u_ni_dotprod")
            # [batch, 1]
            diff = u_pi - u_ni
            return diff

        else:
            item_feat_vals = []
            transformed_item_inputs = self.input_layer(inputs, groups='item')
            for val in transformed_item_inputs.values():
                val = tf.keras.layers.Flatten()(val)
                item_feat_vals.append(val)
            item_concat_vals = tf.concat(item_feat_vals, axis=-1, name='concat_item_features')

            # [batch, embedding_size]
            item_embeddings = self.item_tower(item_concat_vals, training=training)

            if self.use_shallow:
                item_shallow_feat_vals = []
                transformed_item_shallow_inputs = self.input_layer(inputs, groups='i_shallow')
                for val in transformed_item_shallow_inputs.values():
                    val = tf.keras.layers.Flatten()(val)
                    item_shallow_feat_vals.append(val)
                item_shallow_concat_vals = tf.concat(item_shallow_feat_vals, axis=-1,
                                                     name='concat_user_shallow_features')
                item_shallow_embeddings = self.item_shallow_tower(item_shallow_concat_vals, training=training)
                item_embeddings = tf.concat([user_embeddings, item_shallow_embeddings], axis=-1,
                                            name='concat_final_item_features')

            #对比学习的方式
            if self.model_type == 'constrative':
                #[batch, batch]
                scores = tf.matmul(user_embeddings, item_embeddings, transpose_b=True)

                if self.temperature is not None:
                    scores = scores / self.temperature
                #是否进行消偏
                if self.correct_bias:
                    #获取先验概率
                    prob = self.input_layer(inputs, groups='prob')
                    scores = sampling_prob_correction(scores, prob)
                return scores

            else:
                # [batch, 1]
                if self.use_shallow:
                    user_embeddings = self.user_fusion_tower(user_embeddings, training=training)
                    item_embeddings = self.item_fusion_tower(item_embeddings, training=training)

                logits = tf.reduce_sum(user_embeddings * item_embeddings, axis=-1, keepdims=True, name='ui_dotpord')
                pred = tf.nn.sigmoid(logits, name="ui_sigmoid")
                return pred

    def get_call_signatures(self):
        fn_specs = {}
        user_input_specs = self.input_layer.get_tensor_specs(groups='user')
        fn_specs.update(user_input_specs)
        if self.model_type == 'pairwise':
            p_item_input_specs = self.input_layer.get_tensor_specs(groups='p_item')
            n_item_input_specs = self.input_layer.get_tensor_specs(groups='n_item')
            fn_specs.update(p_item_input_specs)
            fn_specs.update(n_item_input_specs)
        else:
            item_input_specs = self.input_layer.get_tensor_specs(groups='item')
            fn_specs.update(item_input_specs)
        sigs = {
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: self.call.get_concrete_function(fn_specs),
        }
        return sigs

    def get_user_model(self):
        inputs = {}
        groups = 'user'
        for i_desc in self.input_layer.get_feature_input_descs(groups):
            inputs[i_desc.name] = i_desc.to_tf_input()
        feat_vals = []
        transformed_inputs = self.input_layer(inputs, groups=groups)
        for val in transformed_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            feat_vals.append(val)
        concat_vals = tf.concat(feat_vals, axis=-1, name='concat_user_features')
        embeddings = self.user_tower(concat_vals)

        if self.use_shallow:
            for i_desc in self.input_layer.get_feature_input_descs('u_shallow'):
                inputs[i_desc.name] = i_desc.to_tf_input()
            shallow_feat_vals = []
            transformed_shallow_inputs = self.input_layer(inputs, groups='u_shallow')
            for val in transformed_shallow_inputs.values():
                val = tf.keras.layers.Flatten()(val)
                shallow_feat_vals.append(val)
            concat_shallow_vals = tf.concat(shallow_feat_vals, axis=-1, name='concat_shallow_features')
            shallow_embeddings = self.shallow_tower(concat_shallow_vals)
            concat_input_vals = tf.concat([embeddings, shallow_embeddings], axis=-1, name='concat_input_features')
            embeddings = self.user_fusion_tower(concat_input_vals)

        # if self.use_cosine:
        #     embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
        return tf.keras.Model(inputs, outputs=embeddings, name=self.name + "-user_tower")

    def get_item_model(self):
        inputs = {}
        if self.model_type == 'pairwise':
            groups = 'p_item'
        else:
            groups = 'item'

        for i_desc in self.input_layer.get_feature_input_descs(groups):
            inputs[i_desc.name] = i_desc.to_tf_input()

        feat_vals = []
        transformed_inputs = self.input_layer(inputs, groups=groups)
        for val in transformed_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            feat_vals.append(val)
        concat_vals = tf.concat(feat_vals, axis=-1, name='concat_user_features')
        embeddings = self.item_tower(concat_vals)
        # if self.use_cosine:
        #     embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
        return tf.keras.Model(inputs, outputs=embeddings, name=self.name + "-item_tower")

    def get_ctr_model(self):
        if self.model_type == 'pairwise':
            item_group = 'p_item'
        else:
            item_group = 'item'

        item_inputs = {}
        for i_desc in self.input_layer.get_feature_input_descs(item_group):
            item_inputs[i_desc.name] = i_desc.to_tf_input()

        user_inputs = {}
        for i_desc in self.input_layer.get_feature_input_descs('user'):
            user_inputs[i_desc.name] = i_desc.to_tf_input()

        item_feat_vals = []
        transformed_item_inputs = self.input_layer(item_inputs, groups=item_group)
        for val in transformed_item_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            item_feat_vals.append(val)
        item_concat_vals = tf.concat(item_feat_vals, axis=-1, name='concat_item_features')
        # [batch, embedding_size]
        item_embeddings = self.item_tower(item_concat_vals)

        user_feat_vals = []
        transformed_user_inputs = self.input_layer(user_inputs, groups='user')
        for val in transformed_user_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            user_feat_vals.append(val)
        user_concat_vals = tf.concat(user_feat_vals, axis=-1, name='concat_user_features')
        # [batch, embedding_size]
        user_embeddings = self.user_tower(user_concat_vals)

        # [batch, 1]
        logits = tf.reduce_sum(user_embeddings * item_embeddings, axis=-1, keepdims=True, name='ui_dotpord')
        pred_ctr = tf.nn.sigmoid(logits, name="ui_sigmoid")
        inputs = {}
        inputs.update(item_inputs)
        inputs.update(user_inputs)
        return tf.keras.Model(inputs, outputs=pred_ctr, name=self.name + "-ctr_model")

    def get_config(self):
        config = {
            "model_input_config": self.model_input_config,
            "user_tower_units": self.user_tower_units,
            "user_tower_hidden_act": self.user_tower_hidden_act,
            "user_tower_output_act": self.user_tower_output_act,
            "user_tower_dropout": self.user_tower_dropout,
            "user_tower_use_bn": self.user_tower_use_bn,
            "user_tower_l1_reg": self.user_tower_l1_reg,
            "user_tower_l2_reg": self.user_tower_l2_reg,
            "user_tower_use_bias": self.user_tower_use_bias,
            "item_tower_units": self.item_tower_units,
            "item_tower_hidden_act": self.item_tower_hidden_act,
            "item_tower_output_act": self.item_tower_output_act,
            "item_tower_dropout": self.item_tower_dropout,
            "item_tower_use_bn": self.item_tower_use_bn,
            "item_tower_l1_reg": self.item_tower_l1_reg,
            "item_tower_l2_reg": self.item_tower_l2_reg,
            "item_tower_use_bias": self.item_tower_use_bias,
            "model_type": self.model_type,
            "use_cosine": self.use_cosine,
            "temperature": self.temperature,
            'correct_bias': self.correct_bias,
            "use_shallow": self.use_shallow,
            "shallow_tower": self.shallow_tower,
            "shallow_tower_act": self.shallow_tower_act,
            "name": self.name
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
