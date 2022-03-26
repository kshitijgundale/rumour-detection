from datetime import datetime

def twittertime_to_timestamp(twitter_time) -> float:
    '''
    Converts twitter time to POSIX timestamp
    '''
    return datetime.strptime(twitter_time, '%a %b %d %H:%M:%S +0000 %Y').timestamp()

def get_account_age(nodes_dict, cascades, node_id) -> int:
    '''
    Returns account age at the time of tweet creation
    '''
    node = nodes_dict[node_id]
    tweet_created_at = twittertime_to_timestamp(node["created_at"])
    user_created_at = twittertime_to_timestamp(node["user"]["created_at"])
    return "account_age", tweet_created_at - user_created_at

def is_user_verified(nodes_dict, cascades, node_id) -> int:
    node = nodes_dict[node_id]
    return "is_verified", 1 if node['user']['verified'] else 0

def followers_count(nodes_dict, cascades, node_id) -> int:
    node = nodes_dict[node_id]
    return "follower_count", node['user']["followers_count"]

def following_count(nodes_dict, cascades, node_id) -> int:
    node = nodes_dict[node_id]
    return "following_count", node["user"]["friends_count"]

def num_hashtags(nodes_dict, cascades, node_id) -> int:
    node = nodes_dict[node_id]
    return "num_hastags", len(node["entities"]["hashtags"])

def num_user_mentions(nodes_dict, cascades, node_id) -> int:
    node = nodes_dict[node_id]
    return "num_hastags", len(node["entities"]["user_mentions"])

def node_type(nodes_dict, cascades, node_id) -> int:
    node = nodes_dict[node_id]
    mapping = {"tweet":1, "reply":2, "quote":3, "retweet":4}
    return "node_type", mapping[node['node_type']]