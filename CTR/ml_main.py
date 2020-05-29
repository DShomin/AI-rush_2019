import xgboost as xgb
# import lightgbsm as lgb
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
import pickle
import sys
np.set_printoptions(threshold=sys.maxsize)

def img_pca_merge(n_components=30):
    
    # with open(os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image_features.pkl'), 'rb') as handle:
    #     print("loaded image feature pickle")
    #     train_image = pickle.load(handle)
    print("loaded image feature pickle")
    nsml.load(session='team_26/airush2/1005', checkpoint='train_image_features_pkl', load_fn=img_pickle_load)
    
    num_features = n_components
    
    image_index_article = pd.DataFrame(train_image).T.index
    train_image_pca = PCA(n_components=num_features, random_state=42).fit_transform(pd.DataFrame(train_image).T)
    train_image_pca_df = pd.concat([pd.DataFrame(image_index_article, columns=['article_id']), pd.DataFrame(train_image_pca, columns=['pca'+str(i) for i in range(1, num_features+1)])], 1)
    
    return train_image_pca_df

def preprocess(input_data, hh_dict, gender_dict, age_dict):
    
    data = input_data.copy()
    
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
    print('loaded model checkpoints...!')

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

def infer(root, phase):
    # root : csv file path
    print('_infer root - : ', root)
    print('start infer')
    
    df = load_data(config)
    
    y_pred = np.zeros(len(df))
    # model load
    xgb_lists = (
        ('team_26/airush2/1152', 'epoch4'),
        ('team_26/airush2/1152', 'epoch5'),
        ('team_26/airush2/1152', 'epoch6'),
        ('team_26/airush2/1152', 'epoch7'),
        ('team_26/airush2/1152', 'epoch8'), 
    )
    for i, (session, checkpoint) in enumerate(xgb_lists):
        nsml.load(checkpoint=checkpoint, session=session)
        y_pred += model.predict(xgb.DMatrix(df))/len(xgb_lists)
    else:
        y_pred = custom_round(y_pred, np.quantile(y_pred, 0.86))
    print('end infer')
    return y_pred

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
        data_df = pd.read_csv(DATASET_PATH + '/train/train_data/train_data', sep='\t', dtype={
                                        'article_id': str,
                                        'hh': int, 'gender': str,
                                        'age_range': str,
                                        'read_article_ids': str
                                    })
        data_df = data_df.dropna().reset_index(drop=True)
        # label read
        train_label = pd.read_csv(DATASET_PATH + '/train/train_label', sep='\t', dtype={'label': int})

        # article information
        train_data_article = pd.read_csv(DATASET_PATH + '/train/train_data/train_data_article.tsv', sep='\t')

        # preprocessing data
        hh_dict = defaultdict()
        gender_dict = defaultdict()
        age_dict = defaultdict()
        hh_unique = np.unique(data_df['hh'])
        gender_unique = np.unique(data_df['gender'])
        age_unique = np.unique(data_df['age_range'])        

        for idx, value in zip(hh_unique, range(1, len(hh_unique)+1)):
            hh_dict[idx]=value
        for idx, value in zip(gender_unique, range(1, len(gender_unique)+1)):
            gender_dict[idx]=value
        for idx, value in zip(age_unique, range(1, len(age_unique)+1)):
            age_dict[idx]=value
           
        # preprocessing
        data_df = preprocess(data_df, hh_dict, gender_dict, age_dict)

        # category feature
        data_df = pd.merge(data_df, train_data_article[['article_id', 'category_id']].fillna(-99), how='left', on='article_id').fillna(-99)
        
        # image feature concat
        img_pca_merge_df = img_pca_merge(n_components=100)
        data_df = pd.merge(data_df, img_pca_merge_df, how='left', on='article_id')
        
        # mean encoding
        train_df = data_df.copy()
        ALPHA = 0.1
        article_mean = mean_encoding(pd.concat([train_df, train_label], 1), feature='article_id', alpha=ALPHA)
        hh_mean = mean_encoding(pd.concat([train_df, train_label], 1), feature='hh', alpha=ALPHA)
        age_mean = mean_encoding(pd.concat([train_df, train_label], 1), feature='age_range', alpha=ALPHA)
        history_mean = mean_encoding(pd.concat([train_df, train_label], 1), feature='read_article_ids_len', alpha=ALPHA)
        category_mean = mean_encoding(pd.concat([train_df, train_label], 1), feature='category_id', alpha=ALPHA)

        train_df = pd.merge(train_df, article_mean.reset_index(), how='left', on='article_id').rename(columns={0:'article_mean'})
        train_df = pd.merge(train_df, hh_mean.reset_index(), how='left', on='hh').rename(columns={0:'hh_mean'})
        train_df = pd.merge(train_df, age_mean.reset_index(), how='left', on='age_range').rename(columns={0:'age_mean'})
        train_df = pd.merge(train_df, history_mean.reset_index(), how='left', on='read_article_ids_len').rename(columns={0:'history_mean'})
        train_df = pd.merge(train_df, category_mean.reset_index(), how='left', on='category_id').rename(columns={0:'category_mean'}).fillna(0)

        # features combine
        train_df['hh_gender'] = train_df['hh'].astype(str) + train_df['gender'].astype(str)
        train_df['hh_age'] = train_df['hh'].astype(str) + train_df['age_range'].astype(str)
        train_df['hh_history'] = train_df['hh'].astype(str) + train_df['read_article_ids_len'].astype(str)
        train_df['hh_category'] = train_df['hh'].astype(str) + train_df['category_id'].astype(str)

        train_df['gender_age'] = train_df['gender'].astype(str) + train_df['age_range'].astype(str)
        train_df['gender_history'] = train_df['gender'].astype(str) + train_df['read_article_ids_len'].astype(str)
        train_df['gender_category'] = train_df['gender'].astype(str) + train_df['category_id'].astype(str)


        train_df['age_history'] = train_df['age_range'].astype(str) + train_df['read_article_ids_len'].astype(str)
        train_df['age_category'] = train_df['age_range'].astype(str) + train_df['category_id'].astype(str)

        train_df['history_category'] = train_df['read_article_ids_len'].astype(str) + train_df['category_id'].astype(str)


        train_df['hh_gender_age'] = train_df['hh'].astype(str) + train_df['gender'].astype(str) + train_df['age_range'].astype(str)
        train_df['hh_gender_history'] = train_df['hh'].astype(str) + train_df['gender'].astype(str) + train_df['read_article_ids_len'].astype(str)
        train_df['hh_gender_category_id'] = train_df['hh'].astype(str) + train_df['gender'].astype(str) + train_df['category_id'].astype(str)

        train_df['gender_age_history'] = train_df['gender'].astype(str) + train_df['age_range'].astype(str) + train_df['read_article_ids_len'].astype(str)
        train_df['gender_age_category'] = train_df['gender'].astype(str) + train_df['age_range'].astype(str) + train_df['category_id'].astype(str)

        train_df['age_history_category'] = train_df['age_range'].astype(str) + train_df['read_article_ids_len'].astype(str) + train_df['category_id'].astype(str)

        combine_feature_list = ['hh_gender', 'hh_age', 'hh_history', 'hh_category', 'gender_age', 'gender_history', 'gender_category', 
                        'age_history', 'age_category', 'history_category', 'hh_gender_age', 'hh_gender_history', 
                        'hh_gender_category_id', 'gender_age_history', 'gender_age_category', 'age_history_category']

        for feature in combine_feature_list:
            train_df[feature] = train_df[feature].apply(lambda x: x.replace('.', '').replace('-', '')).astype(int)

        # definition dataframe
        data_df = train_df.copy()
        data_df = pd.concat([data_df, train_label], 1)

        return data_df

    else:
        df = pd.read_csv(DATASET_PATH + '/test/test_data/test_data', sep='\t', dtype={
                                        'article_id': str,
                                        'hh': int, 'gender': str,
                                        'age_range': str,
                                        'read_article_ids': str
                                        })
        ###############################################################################################################################################################################
        nsml.load(session='team_26/airush2/1152', checkpoint='dataframe', load_fn=df_load) # train load ################################################################################
        ###############################################################################################################################################################################
        df_merge.drop(columns='label', inplace=True)

        # preprocessing data
        hh_dict = defaultdict()
        gender_dict = defaultdict()
        age_dict = defaultdict()
        hh_unique = np.unique(df['hh'])
        gender_unique = np.unique(df['gender'])
        age_unique = np.unique(df['age_range'])        

        for idx, value in zip(hh_unique, range(1, len(hh_unique)+1)):
            hh_dict[idx]=value
        for idx, value in zip(gender_unique, range(1, len(gender_unique)+1)):
            gender_dict[idx]=value
        for idx, value in zip(age_unique, range(1, len(age_unique)+1)):
            age_dict[idx]=value
           
        # preprocessing
        df = preprocess(df, hh_dict, gender_dict, age_dict)
        test_df = df.copy()

        test_df = pd.merge(test_df, df_merge[['article_id', 'category_id']].drop_duplicates(), how='left', on=['article_id']).fillna(-99)
        pca_list = ['article_id']
        pca_list2 = ['pca'+str(i) for i in range(1, 31)]
        pca_list.extend(pca_list2)
        test_df = pd.merge(test_df, df_merge[pca_list].drop_duplicates(), how='left', on=['article_id']).fillna(0)

        test_df = pd.merge(test_df, df_merge[['article_id', 'article_mean']].drop_duplicates(), how='left', on=['article_id']).fillna(0)
        test_df = pd.merge(test_df, df_merge[['hh', 'hh_mean']].drop_duplicates(), how='left', on=['hh']).fillna(0)
        test_df = pd.merge(test_df, df_merge[['age_range', 'age_mean']].drop_duplicates(), how='left', on=['age_range']).fillna(0)
        test_df = pd.merge(test_df, df_merge[['read_article_ids_len', 'history_mean']].drop_duplicates(), how='left', on=['read_article_ids_len']).fillna(0)
        test_df = pd.merge(test_df, df_merge[['category_id', 'category_mean']].drop_duplicates(), how='left', on=['category_id']).fillna(0)

        # features combine
        test_df['hh_gender'] = test_df['hh'].astype(str) + test_df['gender'].astype(str)
        test_df['hh_age'] = test_df['hh'].astype(str) + test_df['age_range'].astype(str)
        test_df['hh_history'] = test_df['hh'].astype(str) + test_df['read_article_ids_len'].astype(str)
        test_df['hh_category'] = test_df['hh'].astype(str) + test_df['category_id'].astype(str)

        test_df['gender_age'] = test_df['gender'].astype(str) + test_df['age_range'].astype(str)
        test_df['gender_history'] = test_df['gender'].astype(str) + test_df['read_article_ids_len'].astype(str)
        test_df['gender_category'] = test_df['gender'].astype(str) + test_df['category_id'].astype(str)


        test_df['age_history'] = test_df['age_range'].astype(str) + test_df['read_article_ids_len'].astype(str)
        test_df['age_category'] = test_df['age_range'].astype(str) + test_df['category_id'].astype(str)

        test_df['history_category'] = test_df['read_article_ids_len'].astype(str) + test_df['category_id'].astype(str)


        test_df['hh_gender_age'] = test_df['hh'].astype(str) + test_df['gender'].astype(str) + test_df['age_range'].astype(str)
        test_df['hh_gender_history'] = test_df['hh'].astype(str) + test_df['gender'].astype(str) + test_df['read_article_ids_len'].astype(str)
        test_df['hh_gender_category_id'] = test_df['hh'].astype(str) + test_df['gender'].astype(str) + test_df['category_id'].astype(str)

        test_df['gender_age_history'] = test_df['gender'].astype(str) + test_df['age_range'].astype(str) + test_df['read_article_ids_len'].astype(str)
        test_df['gender_age_category'] = test_df['gender'].astype(str) + test_df['age_range'].astype(str) + test_df['category_id'].astype(str)

        test_df['age_history_category'] = test_df['age_range'].astype(str) + test_df['read_article_ids_len'].astype(str) + test_df['category_id'].astype(str)

        combine_feature_list = ['hh_gender', 'hh_age', 'hh_history', 'hh_category', 'gender_age', 'gender_history', 'gender_category', 
                        'age_history', 'age_category', 'history_category', 'hh_gender_age', 'hh_gender_history', 
                        'hh_gender_category_id', 'gender_age_history', 'gender_age_category', 'age_history_category']

        for feature in combine_feature_list:
            test_df[feature] = test_df[feature].apply(lambda x: x.replace('.', '').replace('-', '')).astype(int)

        test_df = test_df[df_merge.columns].drop(columns='article_id')
        return test_df

def custom_round(predict, threshold):
    data = predict.copy()
    try:
        data.loc[data>=threshold] = 1
        data.loc[data<threshold] = 0
    except:
        data[data>=threshold] = 1
        data[data<threshold] = 0
    return data

def xgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = custom_round(y_hat, np.quantile(y_hat, 0.85))
    return 'f1', -f1_score(y_true, y_hat)

def train_model(df : pd.DataFrame, fold):

    # model parametes
    params = {
    'objective':'binary:logistic',
    'n_estimators':1000,
    'max_depth':8,
    'learning_rate':0.1,
    'subsample':0.9,
    'colsample_bytree':0.9,
    'reg_alpha':0.1,
    'tree_method':'gpu_hist',
    'seed':42
    }
    # split validation set
    skf = StratifiedKFold(n_splits=5, random_state=42)
    
    # def label, df
    label = df['label']
    df = df.drop(columns='label')
    df.drop(columns='article_id', inplace=True)

    for fold_idx, (trn_idx, val_idx) in enumerate(skf.split(df, label)):
        
        if fold_idx == fold:
            trn_df = xgb.DMatrix(df.loc[trn_idx], label=label.loc[trn_idx])
            val_df = xgb.DMatrix(df.loc[val_idx], label=label.loc[val_idx])

            model = xgb.train(params, 
                        trn_df, 
                        num_boost_round=5000,
                        evals=[(trn_df, 'train'), (val_df, 'val')], 
                        early_stopping_rounds = 100, verbose_eval=200, feval=xgb_f1_score)

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
        df_merge = df.drop_duplicates()
        nsml.save('dataframe', save_fn=df_save)
        for fold in range(config.folds):
            model = train_model(df, fold)
            nsml.save('epoch{}'.format(fold+4))
    else:
        nsml.load(session='team_26/airush2/1019', checkpoint='article_info', load_fn=df_load)
        global train_article
        train_article = df_merge
        print('loaded article info ', df_merge.info())