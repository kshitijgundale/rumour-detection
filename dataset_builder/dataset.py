import json
import os
import os.path as path
from dataset_builder.propagation_graph import PropagationGraph
from dataset_builder.utils import get_account_age, followers_count, following_count, is_user_verified,\
    num_hashtags, num_user_mentions, node_type, get_vader_score, time_diff_with_source
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch

def build_gdl_dataset(dataset, dataset_name, test_size, early_size, val_size=None, cv=None):

    data_list = []

    for label in ["real", "fake"]:
        p = path.join(f"./raw_data/{dataset}/{label}")

        for news in tqdm(os.listdir(p)):
            if path.exists(path.join(p, news, "tweets")):
                tweets = [json.load(open(path.join(p, news, "tweets", i))) for i in os.listdir(path.join(p, news, "tweets"))]
            else:
                continue

            retweets = None
            if path.exists(path.join(p, "retweets.json")):
                retweets = json.load(open(path.join(p, "retweets.json")))

            graph = PropagationGraph(
                tweets=tweets, 
                retweets=retweets, 
                quotes=None, 
                replies=None,
                features=[
                    "account_age", "is_verified", "follower_count", "following_count",
                    "num_hastags", "num_user_mentions", "node_type", "source_time_diff", "vader_score"
                ], 
                feature_extractors=[
                    get_account_age, followers_count, following_count, is_user_verified,
                    num_hashtags, num_user_mentions, node_type, get_vader_score, time_diff_with_source
                ], 
                deadline=None, 
                news_utc=None,
                news_id=news
            )

            data = graph.get_pyg_data_object()
            data.news_id = news
            data.y = 1 if label == "real" else 0
            data_list.append(data)

    train_dataset, test_dataset = train_test_split(data_list, test_size=test_size, stratify=[i.y for i in data_list])
    train_dataset, early_dataset = train_test_split(train_dataset, test_size=early_size/(1-test_size), stratify=[i.y for i in train_dataset])
    print("Length of test_dataset: " + str(len(test_dataset)))
    print("Length of early_dataset: " + str(len(early_dataset)))

    dataset = {}
    news_ids = {}

    if cv and not val_size:
        print("Length of train_dataset: " + str(len(train_dataset)))

        kfolds = []
        kf = StratifiedKFold(n_splits=cv)
        for i,j in kf.split(train_dataset, [i.y for i in train_dataset]):
            kfolds.append((i.tolist(), j.tolist()))
        dataset['kfolds'] = kfolds
        dataset['test_dataset'] = test_dataset
        dataset['train_dataset'] = train_dataset
        dataset['early_dataset'] = early_dataset

        news_ids['kfolds'] = kfolds
        news_ids['test_dataset'] = [(i.news_id, i.y) for i in test_dataset]
        news_ids['train_dataset'] = [(i.news_id, i.y) for i in train_dataset]
        news_ids['early_dataset'] = [(i.news_id, i.y) for i in early_dataset]

    else:
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=val_size/(1-(test_size+early_size)), stratify=[i.y for i in train_dataset])
        print("Length of train_dataset: " + str(len(train_dataset)))
        print("Length of val_dataset: " + str(len(val_dataset)))
        dataset['train_dataset'] = train_dataset
        dataset['val_dataset'] = val_dataset
        dataset['test_dataset'] = test_dataset
        dataset['early_dataset'] = early_dataset

        news_ids['test_dataset'] = [(i.news_id, i.y) for i in test_dataset]
        news_ids['train_dataset'] = [(i.news_id, i.y) for i in train_dataset]
        news_ids['val_dataset'] = [(i.news_id, i.y) for i in val_dataset]
        news_ids['early_dataset'] = [(i.news_id, i.y) for i in early_dataset]

    torch.save(dataset, f'{dataset_name}.pt')
    json.dump(news_ids, open(f"{dataset_name}_news_ids_dataset.json", 'w'))

build_gdl_dataset(
    dataset= "politifact",
    test_size = 0.1,
    early_size = 0.15,
    cv=10,
    dataset_name="politifact_gdl_dataset"
)

    