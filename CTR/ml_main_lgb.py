# import xgboost as xgb
# import catboost
import lightgbm as lgb
import tqdm
import nsml
import os
import argparse
import numpy as np
import pandas as pd
import time
import datetime
import base64
import gzip
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from data_local_loader import get_data_loader
from evaluation import evaluation_metrics, evaluate
from nsml import DATASET_PATH
from data_loader import feed_infer
from collections import defaultdict
from sklearn.decomposition import PCA
from pathlib import Path
from collections import Counter
import pickle
import sys
np.set_printoptions(threshold=sys.maxsize)

def img_pca_merge(n_components=30):
    
    # with open(os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image_features.pkl'), 'rb') as handle:
    #     print("loaded image feature pickle")
    #     train_image = pickle.load(handle)
    print("loaded image feature pickle")
    # nsml.load(session='team_26/airush2/1005', checkpoint='train_image_features_pkl', load_fn=img_pickle_load)
    # print("dense feature extract")
    nsml.load(session='team_26/airush2/1300', checkpoint='train_image_features_pkl', load_fn=img_pickle_load)
    # nsml.load(session='team_26/airush2/1181', checkpoint='train_image_features_pkl', load_fn=img_pickle_load)
    
    num_features = n_components
    
    image_index_article = pd.DataFrame(train_image).T.index
    train_image_pca = PCA(n_components=num_features, random_state=42).fit_transform(pd.DataFrame(train_image).T)
    train_image_pca_df = pd.concat([pd.DataFrame(image_index_article, columns=['article_id']), pd.DataFrame(train_image_pca, columns=['pca'+str(i) for i in range(1, num_features+1)])], 1)
    
    return train_image_pca_df

def preprocess(input_data):
    
    data = input_data.copy()
    
    # preprocessing data
    hh_dict = defaultdict()
    gender_dict = defaultdict()
    age_dict = defaultdict()
    hh_unique = np.unique(data['hh'])
    gender_unique = np.unique(data['gender'])
    age_unique = np.unique(data['age_range'])        

    for idx, value in zip(hh_unique, range(1, len(hh_unique)+1)):
        hh_dict[idx]=value
    for idx, value in zip(gender_unique, range(1, len(gender_unique)+1)):
        gender_dict[idx]=value
    for idx, value in zip(age_unique, range(1, len(age_unique)+1)):
        age_dict[idx]=value

    # modify columns
    data['hh'] = data['hh'].apply(lambda x: hh_dict[x])
    data['gender'] = data['gender'].apply(lambda x: gender_dict[x])
    data['age_range'] = data['age_range'].apply(lambda x: age_dict[x])
    data['read_article_ids_len'] = data['read_article_ids'].apply(lambda x: x.count(',')+1 if str(x)!='nan' else 0)
    
    # drop columns
    data.drop(columns='read_article_ids', inplace=True)
    
    return data

def mean_encoding(input_data, feature, alpha):
    data = input_data[[feature, 'label']].copy()
    
    prior = data['label'].mean()
    n = data.groupby(feature).size()
    mu = data.groupby(feature)['label'].mean()
    mu_smoothed = (n * mu + alpha * prior) / (n + alpha)
    
    return mu_smoothed

def df_save(dir_name, *args, **kwargs):
    os.makedirs(dir_name, exist_ok=True)
    df_merge.to_csv(os.path.join(dir_name, 'df.tsv'), sep='\t', index=False)

def df_load(dir_name, *args, **kwargs):
    global df_merge
    df_merge = pd.read_csv(os.path.join(dir_name, 'df.tsv'), sep='\t')
    print('loaded df_merge data checkpoints...!')

def df_load_image(dir_name, *args, **kwargs):
    global image_vector
    image_vector = pd.read_csv(os.path.join(dir_name, 'df.tsv'), sep='\t')
    print('loaded image_vector checkpoints...!')

def df_load_title(dir_name, *args, **kwargs):
    global title_vector
    title_vector = pd.read_csv(os.path.join(dir_name, 'df.tsv'), sep='\t')
    print('loaded title_vecotr checkpoints...!')

def sim_load(dir_name):
    global doc_sim
    with open(os.path.join(dir_name, 'embed.pkl'), 'rb') as f:
        doc_sim = pickle.load(f)
    print('load title similarity')

def article_load(dir_name, *args, **kwargs):
    global train_data_article2
    train_data_article2 = pd.read_csv(os.path.join(dir_name, 'df.tsv'), sep='\t')
    print('loaded article_data checkpoints...!')

def img_pickle_load(dir_name):
    global train_image    
    with open(os.path.join(dir_name, 'image_features.pkl'), 'rb') as handle:
        train_image = pickle.load(handle)

def save(dir_name, *args, **kwargs):
    os.makedirs(dir_name, exist_ok=True)
    joblib.dump(model, os.path.join(dir_name, 'model.ckpt'))
    print('saved model checkpoints...!')

def load(dir_name, *args, **kwargs):
    global model
    model = joblib.load(os.path.join(dir_name, 'model.ckpt'))
    print('loaded model checkpoints...!')

def load_data(config):
    '''
    train_label(path): DATASET_PATH / train / train_label
    train_data(path): DATASET_PATH / train / train_data / train_data
    
    test_label(path): DATASET_PATH / test / test_label
    test_data(path): DATASET_PATH / test / test_data / test_data
    '''
    col_list = []
    if config.use_sex:
        col_list.append('gender')
    if config.use_exposed_time:
        col_list.append('hh')
    if config.use_age:
        col_list.append('age_range')
    if config.use_read_history:
        # col_list.append('read_article_ids')
        col_list.append('article_idx_len')
    if config.mode == 'train':
        # load data
        data_df = pd.read_csv(DATASET_PATH + '/train/train_data/train_data', sep='\t', dtype={
                                        'article_id': str,
                                        'hh': int, 'gender': str,
                                        'age_range': str,
                                        'read_article_ids': str
                                    })
        temp_df = data_df.copy()
        print('complete train data load')
        # label read
        train_label = pd.read_csv(DATASET_PATH + '/train/train_label', sep='\t', dtype={'label': int})
        # article information
        train_data_article = pd.read_csv(DATASET_PATH + '/train/train_data/train_data_article.tsv', sep='\t')
        ###########################################################################################################################################################################################################
        
        # preprocessing & category
        data_df = preprocess(data_df)
        data_df = pd.merge(data_df, train_data_article[['article_id', 'category_id']].fillna(-99), how='left', on='article_id').fillna(-99)
        
        # mean encoding
        train_df = data_df.copy()
        # ALPHA = 1
        # article_mean = mean_encoding(pd.concat([train_df, train_label], 1), feature='article_id', alpha=ALPHA)
        # hh_mean = mean_encoding(pd.concat([train_df, train_label], 1), feature='hh', alpha=ALPHA)
        # age_mean = mean_encoding(pd.concat([train_df, train_label], 1), feature='age_range', alpha=ALPHA)
        # history_mean = mean_encoding(pd.concat([train_df, train_label], 1), feature='read_article_ids_len', alpha=ALPHA)
        # category_mean = mean_encoding(pd.concat([train_df, train_label], 1), feature='category_id', alpha=ALPHA)

        # train_df = pd.merge(train_df, article_mean.reset_index(), how='left', on='article_id').rename(columns={0:'article_mean'})
        # train_df = pd.merge(train_df, hh_mean.reset_index(), how='left', on='hh').rename(columns={0:'hh_mean'})
        # train_df = pd.merge(train_df, age_mean.reset_index(), how='left', on='age_range').rename(columns={0:'age_mean'})
        # train_df = pd.merge(train_df, history_mean.reset_index(), how='left', on='read_article_ids_len').rename(columns={0:'history_mean'})
        # train_df = pd.merge(train_df, category_mean.reset_index(), how='left', on='category_id').rename(columns={0:'category_mean'}).fillna(0)

        # features combine
        combine_feature_list = ['cb'+str(i) for i in range(1, 17)]
        train_df['cb1'] = train_df['hh'].astype(str) + train_df['gender'].astype(str)
        train_df['cb2'] = train_df['hh'].astype(str) + train_df['age_range'].astype(str)
        train_df['cb3'] = train_df['hh'].astype(str) + train_df['read_article_ids_len'].astype(str)
        train_df['cb4'] = train_df['hh'].astype(str) + train_df['category_id'].astype(str)
        train_df['cb5'] = train_df['gender'].astype(str) + train_df['age_range'].astype(str)
        train_df['cb6'] = train_df['gender'].astype(str) + train_df['read_article_ids_len'].astype(str)
        train_df['cb7'] = train_df['gender'].astype(str) + train_df['category_id'].astype(str)
        train_df['cb8'] = train_df['age_range'].astype(str) + train_df['read_article_ids_len'].astype(str)
        train_df['cb9'] = train_df['age_range'].astype(str) + train_df['category_id'].astype(str)
        train_df['cb10'] = train_df['read_article_ids_len'].astype(str) + train_df['category_id'].astype(str)
        train_df['cb11'] = train_df['hh'].astype(str) + train_df['gender'].astype(str) + train_df['age_range'].astype(str)
        train_df['cb12'] = train_df['hh'].astype(str) + train_df['gender'].astype(str) + train_df['read_article_ids_len'].astype(str)
        train_df['cb13'] = train_df['hh'].astype(str) + train_df['gender'].astype(str) + train_df['category_id'].astype(str)
        train_df['cb14'] = train_df['gender'].astype(str) + train_df['age_range'].astype(str) + train_df['read_article_ids_len'].astype(str)
        train_df['cb15'] = train_df['gender'].astype(str) + train_df['age_range'].astype(str) + train_df['category_id'].astype(str)
        train_df['cb16'] = train_df['age_range'].astype(str) + train_df['read_article_ids_len'].astype(str) + train_df['category_id'].astype(str)
        for feature in combine_feature_list:
            train_df[feature] = train_df[feature].apply(lambda x: x.replace('.', '').replace('-', '')).astype(int)

        # history similarity
        category_dict = defaultdict()
        for idx, value in zip(train_data_article['article_id'], train_data_article['category_id']):
            category_dict[idx]=value
        def row_cal(row):
            try:
                row = row.split(',')
                row = [category_dict[i] for i in row]
            except:
                row = 0
            return row
        train_df['read_article_category'] = temp_df['read_article_ids'].apply(row_cal)
        category_count = []
        max_article = []
        for ca_id, his in zip(train_df['category_id'], train_df['read_article_category']):
            try:
                category_count.append(his.count(ca_id))
                max_article.append(list(Counter(his).keys())[0])
            except:
                category_count.append(0)
                max_article.append(-99)
        train_df['read_article_category'] = category_count
        train_df['maximum_aricle'] = max_article
        train_df['hishory_ratio'] = (train_df['read_article_category']/train_df['read_article_ids_len']).fillna(-99)

        article_dict = defaultdict()
        for article_index, article_id in zip(train_data_article.index, train_data_article['article_id']):
            article_dict[article_id] = article_index

        def article_transform(row):
            try:
                return article_dict[row]
            except:
                return -99
        def article_transform2(row):
            try:
                return category_dict[row]
            except:
                return -99
        train_df['article_num1'] = train_df['article_id'].apply(article_transform)
        train_df['recent_history1'] = temp_df['read_article_ids'].apply(lambda x: article_transform(x.split(',')[0]) if str(x)!='nan' else -99)
        train_df['recent_history2'] = temp_df['read_article_ids'].apply(lambda x: article_transform(x.split(',')[1]) if str(x)!='nan' and x.count(',')>1 else -99)
        # train_df['recent_history3'] = temp_df['read_article_ids'].apply(lambda x: article_transform(x.split(',')[2]) if str(x)!='nan' and x.count(',')>2 else -99)
        # train_df['recent_history4'] = temp_df['read_article_ids'].apply(lambda x: article_transform(x.split(',')[3]) if str(x)!='nan' and x.count(',')>3 else -99)
        # train_df['recent_history5'] = temp_df['read_article_ids'].apply(lambda x: article_transform(x.split(',')[4]) if str(x)!='nan' and x.count(',')>4 else -99)
        # train_df['recent_history6'] = temp_df['read_article_ids'].apply(lambda x: article_transform(x.split(',')[5]) if str(x)!='nan' and x.count(',')>5 else -99)
        # train_df['recent_history7'] = temp_df['read_article_ids'].apply(lambda x: article_transform(x.split(',')[6]) if str(x)!='nan' and x.count(',')>6 else -99)
        # train_df['recent_history8'] = temp_df['read_article_ids'].apply(lambda x: article_transform(x.split(',')[7]) if str(x)!='nan' and x.count(',')>7 else -99)
        # train_df['recent_history9'] = temp_df['read_article_ids'].apply(lambda x: article_transform(x.split(',')[8]) if str(x)!='nan' and x.count(',')>8 else -99)
        # train_df['recent_history10'] = temp_df['read_article_ids'].apply(lambda x: article_transform(x.split(',')[9]) if str(x)!='nan' and x.count(',')>9 else -99)
        
        train_df['category_recent_history1'] = temp_df['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[0]) if str(x)!='nan' else -99)
        train_df['category_recent_history2'] = temp_df['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[1]) if str(x)!='nan' and x.count(',')>1 else -99)
        # train_df['category_recent_history3'] = temp_df['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[2]) if str(x)!='nan' and x.count(',')>2 else -99)
        # train_df['category_recent_history4'] = temp_df['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[3]) if str(x)!='nan' and x.count(',')>3 else -99)
        # train_df['category_recent_history5'] = temp_df['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[4]) if str(x)!='nan' and x.count(',')>4 else -99)
        # train_df['category_recent_history6'] = temp_df['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[5]) if str(x)!='nan' and x.count(',')>5 else -99)
        # train_df['category_recent_history7'] = temp_df['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[6]) if str(x)!='nan' and x.count(',')>6 else -99)
        # train_df['category_recent_history8'] = temp_df['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[7]) if str(x)!='nan' and x.count(',')>7 else -99)
        # train_df['category_recent_history9'] = temp_df['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[8]) if str(x)!='nan' and x.count(',')>8 else -99)
        # train_df['category_recent_history10'] = temp_df['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[9]) if str(x)!='nan' and x.count(',')>9 else -99)

        # image feature concat
        nsml.load(session='team_26/airush2/1441', checkpoint='img_vector', load_fn=df_load_image)
        nsml.load(session='team_26/airush2/1358', checkpoint='title_vector', load_fn=df_load_title)
        train_df = pd.merge(train_df, image_vector, how='left', on='article_id').fillna(0)
        # train_df = pd.merge(train_df, title_vector, how='left', on='article_id').fillna(0)

        # title similariy
        nsml.load(session='team_26/airush2/1355', checkpoint='title_similarity', load_fn=sim_load)
        
        total_count = []
        total_top10 = []
        for idx in tqdm.tqdm(range(len(temp_df))):
            similar_count = 0
            similar_top10 = []
            try:
                temp_dic = doc_sim[temp_df.loc[idx, 'article_id']]
                temp_list = temp_df.loc[idx, 'read_article_ids'].split(',')
                for history_idx in range(2):
                    try:
                        similar_top10.append(temp_dic[temp_list[history_idx]])
                        similar_count += 1
                    except:
                        similar_top10.append(0)
            except:
                similar_top10 = [0,0]
            total_count.append(similar_count)
            total_top10.append(similar_top10)                
        
        train_df['total_count'] = total_count
        total_top10 = pd.DataFrame(total_top10, columns=['ccc'+str(i) for i in range(2)])
        print(total_top10.describe())
        train_df = pd.concat([train_df, total_top10], 1)

        # definition dataframe
        data_df = train_df.copy()
        data_df = pd.concat([data_df, train_label], 1)
        print(data_df.head())

        def reduce_mem_usage(df, verbose=True):
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            start_mem = df.memory_usage().sum() / 1024**2    
            for col in df.columns:
                col_type = df[col].dtypes
                if col_type in numerics:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    if str(col_type)[:3] == 'int':
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)  
                    else:
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float16)
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float64)    
            end_mem = df.memory_usage().sum() / 1024**2
            if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
            return df
        data_df = reduce_mem_usage(data_df)
        return data_df

    else:
        df = pd.read_csv(DATASET_PATH + '/test/test_data/test_data', sep='\t', dtype={
                                        'article_id': str,
                                        'hh': int, 'gender': str,
                                        'age_range': str,
                                        'read_article_ids': str
                                        })
        # load tarin data article 
        nsml.load(session='team_26/airush2/1019', checkpoint='article_info', load_fn=article_load)
        print('loaded article info ')
        print(train_data_article2.head())
        
        # train load
        # nsml.load(session='team_26/airush2/1362', checkpoint='dataframe', load_fn=df_load)
        # trn_label = df_merge['label']
        # df_merge.drop(columns='label', inplace=True)
        temp_df2 = df.copy()

        # preprocessing & category
        test_df = preprocess(df)
        test_df = pd.merge(test_df, train_data_article2[['article_id', 'category_id']].drop_duplicates().fillna(-99), how='left', on=['article_id']).fillna(-99)

        # mean encoding
        # test_df = pd.merge(test_df, df_merge[['article_id', 'article_mean']].drop_duplicates(), how='left', on=['article_id']).fillna(0)
        # test_df = pd.merge(test_df, df_merge[['hh', 'hh_mean']].drop_duplicates(), how='left', on=['hh']).fillna(0)
        # test_df = pd.merge(test_df, df_merge[['age_range', 'age_mean']].drop_duplicates(), how='left', on=['age_range']).fillna(0)
        # test_df = pd.merge(test_df, df_merge[['read_article_ids_len', 'history_mean']].drop_duplicates(), how='left', on=['read_article_ids_len']).fillna(0)
        # test_df = pd.merge(test_df, df_merge[['category_id', 'category_mean']].drop_duplicates(), how='left', on=['category_id']).fillna(0)

        # features combine
        combine_feature_list = ['cb'+str(i) for i in range(1, 17)]
        test_df['cb1'] = test_df['hh'].astype(str) + test_df['gender'].astype(str)
        test_df['cb2'] = test_df['hh'].astype(str) + test_df['age_range'].astype(str)
        test_df['cb3'] = test_df['hh'].astype(str) + test_df['read_article_ids_len'].astype(str)
        test_df['cb4'] = test_df['hh'].astype(str) + test_df['category_id'].astype(str)
        test_df['cb5'] = test_df['gender'].astype(str) + test_df['age_range'].astype(str)
        test_df['cb6'] = test_df['gender'].astype(str) + test_df['read_article_ids_len'].astype(str)
        test_df['cb7'] = test_df['gender'].astype(str) + test_df['category_id'].astype(str)
        test_df['cb8'] = test_df['age_range'].astype(str) + test_df['read_article_ids_len'].astype(str)
        test_df['cb9'] = test_df['age_range'].astype(str) + test_df['category_id'].astype(str)
        test_df['cb10'] = test_df['read_article_ids_len'].astype(str) + test_df['category_id'].astype(str)
        test_df['cb11'] = test_df['hh'].astype(str) + test_df['gender'].astype(str) + test_df['age_range'].astype(str)
        test_df['cb12'] = test_df['hh'].astype(str) + test_df['gender'].astype(str) + test_df['read_article_ids_len'].astype(str)
        test_df['cb13'] = test_df['hh'].astype(str) + test_df['gender'].astype(str) + test_df['category_id'].astype(str)
        test_df['cb14'] = test_df['gender'].astype(str) + test_df['age_range'].astype(str) + test_df['read_article_ids_len'].astype(str)
        test_df['cb15'] = test_df['gender'].astype(str) + test_df['age_range'].astype(str) + test_df['category_id'].astype(str)
        test_df['cb16'] = test_df['age_range'].astype(str) + test_df['read_article_ids_len'].astype(str) + test_df['category_id'].astype(str)
        for feature in combine_feature_list:
            test_df[feature] = test_df[feature].apply(lambda x: x.replace('.', '').replace('-', '')).astype(int)
        
        # history similarity
        category_dict = defaultdict()
        for idx, value in zip(train_data_article2['article_id'], train_data_article2['category_id']):
            category_dict[idx]=value
        def row_cal(row):
            try:
                row = row.split(',')
                row = [category_dict[i] for i in row]
            except:
                row = 0
            return row
        test_df['read_article_category'] = temp_df2['read_article_ids'].apply(row_cal)
        category_count = []
        max_article = []
        for ca_id, his in zip(test_df['category_id'], test_df['read_article_category']):
            try:
                category_count.append(his.count(ca_id))
                max_article.append(list(Counter(his).keys())[0])
            except:
                category_count.append(0)
                max_article.append(-99)
        test_df['read_article_category'] = category_count
        test_df['maximum_aricle'] = max_article
        test_df['hishory_ratio'] = (test_df['read_article_category']/test_df['read_article_ids_len']).fillna(-99)


        article_dict = defaultdict()
        for article_index, article_id in zip(train_data_article2.index, train_data_article2['article_id']):
            article_dict[article_id] = article_index

        def article_transform(row):
            try:
                return article_dict[row]
            except:
                return -99
        def article_transform2(row):
            try:
                return category_dict[row]
            except:
                return -99
        test_df['article_num1'] = test_df['article_id'].apply(article_transform)
        test_df['recent_history1'] = temp_df2['read_article_ids'].apply(lambda x: article_transform(x.split(',')[0]) if str(x)!='nan' else -99)
        test_df['recent_history2'] = temp_df2['read_article_ids'].apply(lambda x: article_transform(x.split(',')[1]) if str(x)!='nan' and x.count(',')>1 else -99)
        # test_df['recent_history3'] = temp_df2['read_article_ids'].apply(lambda x: article_transform(x.split(',')[2]) if str(x)!='nan' and x.count(',')>2 else -99)
        # test_df['recent_history4'] = temp_df2['read_article_ids'].apply(lambda x: article_transform(x.split(',')[3]) if str(x)!='nan' and x.count(',')>3 else -99)
        # test_df['recent_history5'] = temp_df2['read_article_ids'].apply(lambda x: article_transform(x.split(',')[4]) if str(x)!='nan' and x.count(',')>4 else -99)
        # test_df['recent_history6'] = temp_df2['read_article_ids'].apply(lambda x: article_transform(x.split(',')[5]) if str(x)!='nan' and x.count(',')>5 else -99)
        # test_df['recent_history7'] = temp_df2['read_article_ids'].apply(lambda x: article_transform(x.split(',')[6]) if str(x)!='nan' and x.count(',')>6 else -99)
        # test_df['recent_history8'] = temp_df2['read_article_ids'].apply(lambda x: article_transform(x.split(',')[7]) if str(x)!='nan' and x.count(',')>7 else -99)
        # test_df['recent_history9'] = temp_df2['read_article_ids'].apply(lambda x: article_transform(x.split(',')[8]) if str(x)!='nan' and x.count(',')>8 else -99)
        # test_df['recent_history10'] = temp_df2['read_article_ids'].apply(lambda x: article_transform(x.split(',')[9]) if str(x)!='nan' and x.count(',')>9 else -99)
        
        test_df['category_recent_history1'] = temp_df2['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[0]) if str(x)!='nan' else -99)
        test_df['category_recent_history2'] = temp_df2['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[1]) if str(x)!='nan' and x.count(',')>1 else -99)
        # test_df['category_recent_history3'] = temp_df2['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[1]) if str(x)!='nan' and x.count(',')>2 else -99)
        # test_df['category_recent_history4'] = temp_df2['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[1]) if str(x)!='nan' and x.count(',')>3 else -99)
        # test_df['category_recent_history5'] = temp_df2['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[1]) if str(x)!='nan' and x.count(',')>4 else -99)
        # test_df['category_recent_history6'] = temp_df2['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[1]) if str(x)!='nan' and x.count(',')>5 else -99)
        # test_df['category_recent_history7'] = temp_df2['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[1]) if str(x)!='nan' and x.count(',')>6 else -99)
        # test_df['category_recent_history8'] = temp_df2['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[1]) if str(x)!='nan' and x.count(',')>7 else -99)
        # test_df['category_recent_history9'] = temp_df2['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[1]) if str(x)!='nan' and x.count(',')>8 else -99)
        # test_df['category_recent_history10'] = temp_df2['read_article_ids'].apply(lambda x: article_transform2(x.split(',')[1]) if str(x)!='nan' and x.count(',')>9 else -99)

        # image feature concat
        nsml.load(session='team_26/airush2/1441', checkpoint='img_vector', load_fn=df_load_image)
        nsml.load(session='team_26/airush2/1358', checkpoint='title_vector', load_fn=df_load_title)
        test_df = pd.merge(test_df, image_vector, how='left', on='article_id').fillna(0)
        # test_df = pd.merge(test_df, title_vector, how='left', on='article_id').fillna(0)

        # title similariy
        nsml.load(session='team_26/airush2/1355', checkpoint='title_similarity', load_fn=sim_load)
        
        total_count = []
        total_top10 = []
        for idx in tqdm.tqdm(range(len(temp_df2))):
            similar_count = 0
            similar_top10 = []
            try:
                temp_dic = doc_sim[temp_df2.loc[idx, 'article_id']]
                temp_list = temp_df2.loc[idx, 'read_article_ids'].split(',')
                for history_idx in range(2):
                    try:
                        similar_top10.append(temp_dic[temp_list[history_idx]])
                        similar_count += 1
                    except:
                        similar_top10.append(0)
            except:
                similar_top10 = [0,0]
            total_count.append(similar_count)
            total_top10.append(similar_top10)                
        
        test_df['total_count'] = total_count
        total_top10 = pd.DataFrame(total_top10, columns=['ccc'+str(i) for i in range(2)])
        print(total_top10.head())
        test_df = pd.concat([test_df, total_top10], 1)

        # definition dataframe
        # test_df = test_df[df_merge.columns].drop(columns='article_id')
        return test_df

def infer(root, phase):
    # root : csv file path
    print('_infer root - : ', root)
    print('start infer')
    
    df = load_data(config)
    df = df.drop(columns='article_id')

    # model load
    y_pred = np.zeros(len(df))
    model_lists = ['epochX5', 'epochX6', 'epochX7', 'epochX8', 'epochX9']
    for i, checkpoint in enumerate(model_lists):
        nsml.load(checkpoint=checkpoint, session='team_26/airush2/1443')
        y_pred += model.predict(df)/len(model_lists)
        # y_pred += model.predict_proba(df)[:, 1]/len(model_lists)
    else:
        y_pred = custom_round(y_pred, np.quantile(y_pred, 0.87))
    
    # nsml.load(checkpoint='epochX', session='team_26/airush2/1362')
    # y_pred = model.predict_proba(df)[:, 1]
    # y_pred = custom_round(y_pred, np.quantile(y_pred, 0.88))
    
    print('end infer')

    print('pseudo labeling')
    try:
        print(trn_label)
        print(y_pred)
        print(pd.Sereis(y_pred).value_counts())
    except:
        pass

    return y_pred

def custom_round(predict, threshold):
    data = predict.copy()
    try:
        data.loc[data>=threshold] = 1
        data.loc[data<threshold] = 0
    except:
        data[data>=threshold] = 1
        data[data<threshold] = 0
    return data

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = custom_round(y_hat, np.quantile(y_hat, 0.87))
    return 'f1', f1_score(y_true, y_hat), True

def train_model(df : pd.DataFrame, fold):

    # model parametes
    params = {
        'objective':'binary',
        "boosting": "gbdt",
        'num_leaves': 500,
        'max_depth': 8,#16
        'learning_rate': 0.1,#0.1
        'min_data_in_leaf': 16, 
        'min_child_samples': 30,
        'min_child_weight': 0.5,
        'min_split_gain': 0.1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 2,
        "bagging_seed": 42,
        "metric": 'lgb_f1_score',
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        'verbose':-1
    }

    # split validation set
    skf = StratifiedKFold(n_splits=5, random_state=42)
    
    # def label, df
    label = df['label']
    df = df.drop(columns='label')
    split_basis = df['article_num1'].astype(str) + label.astype(str)
    df.drop(columns='article_id', inplace=True)
    print('start modeling')
    for fold_idx, (trn_idx, val_idx) in enumerate(skf.split(df, label)):
        
        if fold_idx == fold:

            # model = catboost.CatBoostClassifier(iterations=5000,
            #                                  learning_rate=0.05,
            #                                  depth=10,
            #                                  task_type="GPU",
            #                                  eval_metric='AUC',
            #                                  random_seed = 42,
            #                                  bagging_temperature = 0.2,
            #                                  od_type='Iter',
            #                                  metric_period = 50,
            #                                  od_wait=20)
            # model.fit(df.loc[trn_idx], label.loc[trn_idx],
            # eval_set=(df.loc[val_idx], label.loc[val_idx]),
            # use_best_model=True, early_stopping_rounds=300, 
            # verbose=500)

            trn_df = lgb.Dataset(df.loc[trn_idx], label=label.loc[trn_idx])
            val_df = lgb.Dataset(df.loc[val_idx], label=label.loc[val_idx])

            model = lgb.train(params, trn_df, 1000, 
                            valid_sets = [trn_df, val_df], 
                            early_stopping_rounds = 50, 
                            verbose_eval=100, 
                            feval=lgb_f1_score)
            break

    return model

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # use columns
    parser.add_argument('--use_sex', type=bool, default=True)
    parser.add_argument('--use_age', type=bool, default=True)
    parser.add_argument('--use_exposed_time', type=bool, default=True)
    parser.add_argument('--use_read_history', type=bool, default=True)

    # folds
    parser.add_argument('--folds', type=int, default=5)

    # nsml config
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)
    
    config = parser.parse_args()
    
    nsml.bind(save=save, load=load, infer=infer)

    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        df = load_data(config)
        # df_merge = df.drop_duplicates()
        # nsml.save('dataframe', save_fn=df_save)
        
        
        # model = train_model(df, config.folds)
        # nsml.save('epochX')

        
        for fold in range(config.folds):
            model = train_model(df, fold)
            nsml.save('epochX{}'.format(fold+5))
    else:
        model=np.array([0])
        nsml.save(model)