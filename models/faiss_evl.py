# 安装包

# pip config set global.index-url https://mirrors.tencent.com/pypi/simple/
# def install(package:str):
#     """
#     pip安装包
#     """
#     import subprocess
#     import sys
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# install("faiss-gpu")
# install("tensorflow==2.3.0")
# install("tqdm")


print("import--------------")
from tensorflow.python.keras.models import Model
import tensorflow as tf
from collections import OrderedDict
import faiss
from collections import Counter
import os
import numpy as np
import subprocess
import sys

print("import-----------------2")


def get_item_embed_from_file(embed_file: str, embed_size: int) -> dict:
    embed_dict = OrderedDict()

    with open(embed_file, 'r') as r:
        for cnt, line in enumerate(r.readlines()):
            if cnt == 0:
                continue
            pid, meta, embedding = line.strip().split('|')
            pid = int(pid)
            embedding = [float(v) for v in embedding.strip().split(',')]
            if len(embedding) != embed_size:
                print('error line: {}'.format(line))
                continue
            embed_dict[pid] = embedding
    return embed_dict


# def propensity_score(true_playlist, gamma=1.5):
#     '''Unbiased offline recommender evaluation for missing-not-at random implicit feedback'''
#     pc = Counter(true_playlist).most_common()
#     print('most count playlist: {}\nless count playlist: {}\ntotal playlist: {}'.format(pc[:10], pc[-10:], len(pc)))
#     propensity_scores_dict = {pid: pow(float(count), (gamma + 1) / 2.0) for (pid, count) in pc}
#     return [(tp, propensity_scores_dict[tp]) for tp in true_playlist]


def __parse_csv(file, batch_size: int, field_delim: str, header: list, cols: list,
                column_dtype: list) -> tf.data.Dataset:
    ds = tf.data.experimental.make_csv_dataset(file, batch_size=batch_size, num_epochs=1, field_delim=field_delim,
                                               num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                               prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
                                               label_name=None,
                                               ignore_errors=True,
                                               shuffle=False,
                                               column_defaults=column_dtype,
                                               header=False,
                                               column_names=header, select_columns=cols).cache()

    def _map_func(x):
        for k, v in x.items():
            x[k] = tf.reshape(v, [-1, 1])
        return x

    ds = ds.map(_map_func)
    return ds


def awf_create_val_dataset_fn(file: str, batch_size: int, header: list, cols: list,
                              column_dtype: list) -> tf.data.Dataset:
    print(file)
    import glob
    print(glob.glob(file))
    return __parse_csv(file, batch_size, ',', header, cols, column_dtype)


def awf_load_model_fn(path: str) -> tf.keras.Model:
    return tf.saved_model.load(path)


def awf_load_col_dtype(model: tf.keras.Model) -> dict:
    d = dict()
    for i in model.input:
        d[i] = model.input[i].dtype
    print(d)
    return d


def awf_get_true_mid(use_col: list, name: str) -> list:
    import os
    mid_index = use_col.index(name)
    if mid_index < 0 or mid_index >= len(use_col):
        print("wrong itemname !!! {}".format(name))
        raise NameError(name)
    true_mid = []
    l = sorted(os.listdir(test_data_path))
    for file in l:
        # # 只取01
        # if not file.endswith("01"):
        #     continue
        with open(os.path.join(test_data_path, file), 'r') as r:
            for cnt, line in enumerate(r.readlines()):
                pid = line.strip().split(',')[mid_index]
                pid = int(pid)
                true_mid.append(pid)
    return true_mid


def calculate_hitrate(true_id, recall_id):
    from tqdm import tqdm
    hit_100 = 0
    hit_50 = 0
    hit_25 = 0
    hit_10 = 0

    for label, line in tqdm(zip(true_id, recall_id)):
        try:
            pos_dict = {ele: ele_index for ele_index, ele in enumerate(line)}
            pos = pos_dict.get(label, -1)
            if pos < 0: continue
            # pos = line.index(true_id)
            if pos < 10:
                hit_10 += 1
            if pos < 25:
                hit_25 += 1
            if pos < 50:
                hit_50 += 1
            if pos < 100:
                hit_100 += 1
        except ValueError:
            continue
    d = dict()
    d["hit_10"] = 1.0 * hit_10 / len(true_id)
    d["hit_25"] = 1.0 * hit_25 / len(true_id)
    d["hit_50"] = 1.0 * hit_50 / len(true_id)
    d["hit_100"] = 1.0 * hit_100 / len(true_id)
    print("hit_rate 100 ----{}, 50---{}, 25--- {}, 10 ----{} ".format(hit_100 / len(true_id), hit_50 / len(true_id),
                                                                      hit_25 / len(true_id), hit_10 / len(true_id)))
    return d


def check_first_dim(*args) -> bool:
    curr_size = 0
    ret = True
    for i in args:
        if curr_size == 0:
            print("size {}".format(len(i)))
            curr_size = len(i)
        else:
            if curr_size != len(i):
                ret = False
                raise ValueError("wrong size")
    return ret


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--emb_size', type=int, required=False, default=32)
    argparser.add_argument('--header', type=str, required=False,
                           default="ftime,uin, u_click_seq, u_age_level, u_sex, u_degree, u_income_level, u_city_level, u_os_type, u_log_days, u_net_day,u_expnum_30d ,u_clicknum_30d ,u_clickzhubonum_30d ,u_valid_viewcnt_30d_flow,u_valid_viewcnt_30d_unflow ,u_valid_viewzhubocnt_30d_flow ,u_valid_viewzhubocnt_30d_unflow,u_short_viewcnt_30d_flow ,u_short_viewcnt_30d_unflow ,u_short_viewzhubocnt_30d_flow,u_short_viewzhubocnt_30d_unflow  ,u_long_viewcnt_30d_flow ,u_long_viewcnt_30d_unflow,u_long_viewzhubocnt_30d_flow ,u_long_viewzhubocnt_30d_unflow ,u_followcnt_30d,u_unfollow_30d ,u_giftcnt_30d ,u_gift_totcnt_30d ,u_giftzhubocnt_30d ,u_chargecnt_30d,u_chargemoney_30d ,u_danmunum_30d ,u_danmuzhubocnt_30d,u_sharecnt_30d, anchor_id, a_age, a_sex,a_province, a_kaibo_type, watch_score, finance_score,  a_kaibo_period,a_kaibo_period_weight,imp_label, clk_label, play_time, u_scenario")
    argparser.add_argument('--use_col', type=str, required=False, default="ftime,uin, u_click_seq, u_age_level, u_sex, u_degree, u_income_level, u_city_level, u_os_type, u_log_days, u_net_day,u_expnum_30d ,u_clicknum_30d ,u_clickzhubonum_30d ,u_valid_viewcnt_30d_flow,u_valid_viewcnt_30d_unflow ,u_valid_viewzhubocnt_30d_flow ,u_valid_viewzhubocnt_30d_unflow,u_short_viewcnt_30d_flow ,u_short_viewcnt_30d_unflow ,u_short_viewzhubocnt_30d_flow,u_short_viewzhubocnt_30d_unflow  ,u_long_viewcnt_30d_flow ,u_long_viewcnt_30d_unflow,u_long_viewzhubocnt_30d_flow ,u_long_viewzhubocnt_30d_unflow ,u_followcnt_30d,u_unfollow_30d ,u_giftcnt_30d ,u_gift_totcnt_30d ,u_giftzhubocnt_30d ,u_chargecnt_30d,u_chargemoney_30d ,u_danmunum_30d ,u_danmuzhubocnt_30d,u_sharecnt_30d, anchor_id, a_age, a_sex,a_province, a_kaibo_type, watch_score, finance_score,  a_kaibo_period,a_kaibo_period_weight,imp_label, clk_label, play_time, u_scenario")
    # argparser.add_argument('--recall_num', type=str, required=false)
    argparser.add_argument('--test_data_path', type=str, required=False,
                           default="/mnt/ponzhao/dtower_test_runs/20210617/test_data/user")
    argparser.add_argument('--user_model_path', type=str, required=False,
                           default="/mnt/ponzhao/dtower_test_runs/20210617/saved_model/dtower_pair-user_tower")
    argparser.add_argument('--item_embed_path', type=str, required=False,
                           default="/mnt/ponzhao/dtower_test_runs/20210617/item_embeddings_test.txt")
    argparser.add_argument('--output_path', type=str, required=False, default="/mnt/ponzhao/dtower_test_runs/20210617")

    argparser.add_argument('--true_col', type=str, required=False, default="item_id")

    args = argparser.parse_args()
    print("emb_size", args.emb_size)
    print("header", args.header)
    print("use_col", args.use_col)
    # print("recall_num", args.recall_num)
    print("test_data_path", args.test_data_path)
    print("user_model_path", args.user_model_path)
    print("item_embed_path", args.item_embed_path)
    print("output_path", args.output_path)
    print("true_col", args.true_col)

    embedding_dim = args.emb_size
    all_cols = [i for i in args.header.split(",")]
    use_col = [i for i in args.use_col.split(",")]
    # recall_num = args.recall_num
    test_data_path = args.test_data_path
    ue_model_path = args.user_model_path
    item_embed_path = args.item_embed_path
    true_col = args.true_col

    #    item emb 写入faiss
    print("inserting")
    embed_dict = get_item_embed_from_file(item_embed_path, embedding_dim)
    embed_ids = list(embed_dict.keys())
    embed_index_to_id = {index: pid for index, pid in zip(list(range(len(embed_ids))), embed_ids)}

    item_embs = np.array(list(embed_dict.values()), dtype=np.float32)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(item_embs)
    print("inserting done")

    # 用户embedding预测
    model = awf_load_model_fn(ue_model_path)
    #     print('user embed model input:',model.input)
    #     dtype_dict = awf_load_col_dtype(model)

    column_dtype = [int()] * 5 + [float()] * 19 + [str()] * 10

    print(test_data_path)
    ds = awf_create_val_dataset_fn(os.path.join(test_data_path, "part*"), 4096, all_cols, use_col, column_dtype, )

    # for i in ds:
    #     print("input {} output {}".format(i,model.predict(i)))
    # iterator = ds.as_numpy_iterator()
    # print(dict(next(iterator)))
    #
    print("predicting")

    user_embedding = []

    for element in ds:
        result = model(element)
        user_embedding.append(result)

    user_embedding = tf.concat(user_embedding, axis=0).numpy()
    np.savetxt(os.path.join(args.output_path, "user_embedding.csv"), user_embedding, delimiter=",")
    print("predicting done")

    print("true miding")
    true_mid = awf_get_true_mid(all_cols, true_col)
    print("true miding done")
    # 用户embedding求相似item

    batch_list = []
    batch_size = 10000
    start_pos = 0
    print("searching")

    _, I = index.search(user_embedding, 100, )

    # print(len(user_embedding))
    # while start_pos+ batch_size <= len(user_embedding):
    #     _, batchI = index.search(user_embedding[start_pos:start_pos+ batch_size],100)
    #     start_pos+=batch_size
    #     batch_list.append(batchI)
    #     print("start_pos {}".format(start_pos))
    # if start_pos+batch_size > len(user_embedding):
    #     _, batchI  = index.search(user_embedding[start_pos:],100)
    #     batch_list.append(batchI)
    # I = np.concatenate(batch_list,axis = 0)

    print(len(I), len(user_embedding), len(true_mid))

    check_first_dim(user_embedding, true_mid, I)
    recall_id = [[embed_index_to_id[i] for i in line] for line in I]
    print("searching done")

    with open(os.path.join(args.output_path, "sim_id.txt"), 'w') as f:
        for item in recall_id:
            f.write("%s\n" % item)

    print("hitrating")
    hit_rate = calculate_hitrate(true_mid, recall_id)

    with open(os.path.join(args.output_path, "hit_rate.txt"), 'w') as f:
        for item in hit_rate:
            f.write("%s:%s\n" % (item, hit_rate[item]))
    print("hitrating done")

    with open(os.path.join(args.output_path, "embed_index_to_id.txt"), 'w') as f:
        for item in embed_index_to_id:
            f.write("%s:%s\n" % (item, embed_index_to_id[item]))

    # print(index.reconstruct(10)) // 根据index找embedding

    D, I = index.search(item_embs, 100) # D 距离 I 相似id

    recall_id = [(embed_index_to_id[cnt], [str(embed_index_to_id[i]) + ":" + str(j) for i, j in zip(line, D[cnt])]) for
                 cnt, line in enumerate(I)]

    print(len(I))
    with open(os.path.join(args.output_path, "i_sim_id.txt"), 'w') as f:
        for ori, item in recall_id:
            f.write("%s|%s\n" % (ori, item))

