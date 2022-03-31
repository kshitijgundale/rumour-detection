from dataset_builder.utils import twittertime_to_timestamp
from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd
import torch
from pyvis.network import Network
from torch_geometric.data import Data
from time import process_time

TWEET = "tweet"
RETWEET = "retweet"
QUOTE = "quote"
REPLY = "reply"

@dataclass
class Node:
    node_id: str
    node_type: str
    verified: bool
    username: str
    parent: str = None
    children: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class Cascade:
    cascade_id: str
    nodes: Dict[str, Node] = field(default_factory=dict)


class PropagationGraph():

    def __init__(
        self,
        tweets,
        retweets,
        quotes,
        replies,
        features,
        feature_extractors,
        deadline,
        news_utc,
        news_id
    ):

        self.tweets = tweets
        self.retweets = retweets
        self.quotes = quotes
        self.replies = replies
        self.features = features
        self.feature_extractors = feature_extractors
        self.deadline = deadline
        self.news_utc = news_utc
        self.news_id = news_id
        self.metatdata = {}
       
        self._build_heirarchy()
        self._build_cascades()
        self._build_node_features()
        
    def _build_heirarchy(self):
        self.tweets = sorted(self.tweets, key=lambda x: twittertime_to_timestamp(x['created_at']))
        self.nodes_dict = {}
        self.child_dict = {}
        self.parent_dict = {}
        self.cascade_roots = []

        self.metatdata['num_tweets'] = 0
        self.metatdata['num_retweets'] = 0
        self.metatdata['num_quotes'] = 0
        self.metatdata['num_replies'] = 0

        first_node_ts = twittertime_to_timestamp(self.tweets[0]['created_at'])
        if not self.news_utc:
            self.news_utc = first_node_ts
        last_node_ts = 0
        avg_retweets_per_tweet = []
        avg_replies_per_tweet = []
        avg_quotes_per_tweet = []

        for i in range(len(self.tweets)-1, -1, -1):
            tweet = self.tweets.pop()
            if self.deadline is not None and twittertime_to_timestamp(tweet['created_at']) - self.news_utc > self.deadline:
                break
            if twittertime_to_timestamp(tweet['created_at']) > last_node_ts:
                last_node_ts = twittertime_to_timestamp(tweet['created_at'])
            self.metatdata['num_tweets'] += 1

            tweet['node_type'] = TWEET
            self.nodes_dict[tweet['id_str']] = tweet
            if tweet['id_str'] not in self.child_dict:
                self.child_dict[tweet['id_str']] = []

            source = None
            for j in range(i-1, -1, -1):
                candidate_source = self.tweets[j]
                
                cond_a = candidate_source['user']['id_str'] in [x['id_str'] for x in tweet['entities']['user_mentions']]
                cond_b = tweet['user']['id_str'] in [x['id_str'] for x in candidate_source['entities']['user_mentions']]

                if cond_a or cond_b:
                    source = candidate_source
                    break
    
            if source:
                if source['id_str'] not in self.child_dict:
                    self.child_dict[source['id_str']] = []
                self.child_dict[source['id_str']].append(tweet['id_str'])
                self.parent_dict[tweet['id_str']] = source['id_str']
            else:
                self.cascade_roots.append(tweet['id_str'])
                self.parent_dict[tweet['id_str']] = None

        if self.quotes:
            for i in range(len(self.quotes)):
                quote = self.quotes.pop()
                if self.deadline is None or (self.deadline is not None and twittertime_to_timestamp(quote['created_at']) - self.news_utc < self.deadline):
                    if twittertime_to_timestamp(quote['created_at']) > last_node_ts:
                        last_node_ts = twittertime_to_timestamp(quote['created_at'])
                    self.metatdata['num_quotes'] += 1
                    parent = quote['quoted_status']['id_str']
                    quote['node_type'] = QUOTE
                    self.nodes_dict[qoute['id_str']] = quote
                    if parent in self.nodes_dict:
                        if quote['id_str'] not in self.child_dict:
                            self.child_dict[parent] = []
                        self.child_dict[parent].append(quote['id_str'])
                        self.parent_dict[quote['id_str']] = parent

        if self.replies:
            for i in range(len(self.replies)):
                reply = self.replies.pop()
                if self.deadline is None or (self.deadline is not None and twittertime_to_timestamp(reply['created_at']) - self.news_utc < self.deadline):
                    if twittertime_to_timestamp(reply['created_at']) > last_node_ts:
                        last_node_ts = twittertime_to_timestamp(reply['created_at'])
                    self.metatdata['num_replies'] += 1
                    parent = reply['in_reply_to_status_id_str']
                    reply['node_type'] = REPLY
                    self.nodes_dict[reply['id_str']] = reply
                    if parent in self.nodes_dict:
                        if reply['id_str'] not in self.child_dict:
                            self.child_dict[parent] = []
                        self.child_dict[parent].append(reply['id_str'])
                        self.parent_dict[reply['id_str']] = parent
        
        if self.retweets:
            for tweet_id in self.retweets:
                if tweet_id not in self.nodes_dict:
                    continue
                for i in range(len(self.retweets[tweet_id])):
                    retweet = self.retweets[tweet_id].pop()
                    if self.deadline is None or (self.deadline is not None and twittertime_to_timestamp(retweet['created_at']) - self.news_utc < self.deadline):
                        if twittertime_to_timestamp(retweet['created_at']) > last_node_ts:
                            last_node_ts = twittertime_to_timestamp(retweet['created_at'])
                        self.metatdata['num_retweets'] += 1
                            
                        retweet['node_type'] = RETWEET
                        self.nodes_dict[retweet['id_str']] = retweet

                        if retweet['id_str'] not in self.child_dict:
                            self.child_dict[retweet['id_str']] = []

                        self.child_dict[tweet_id].append(retweet['id_str'])
                        self.parent_dict[retweet['id_str']] = tweet_id

        self.metatdata['num_cascades'] = len(self.cascade_roots)
        self.metatdata['propagation_time'] = last_node_ts - first_node_ts

        del self.tweets
        del self.retweets
        del self.quotes
        del self.replies

    def _build_cascades(self):
        self.cascades = {}
        self.node_mapping = {}

        for root in self.cascade_roots:
            cascade = Cascade(cascade_id=root)
            visited = []
            visited.append(root)

            while len(visited) != 0:
                n = visited.pop()
                node = self.nodes_dict[n]
                cascade.nodes[n] = Node(
                    node_id=n, 
                    node_type=node['node_type'], 
                    parent=self.parent_dict[n], 
                    verified=node['user']['verified'],
                    children=self.child_dict[n],
                    username=node['user']['screen_name'],
                )
                self.node_mapping[n] = root

                visited.extend(self.child_dict[n])

            self.cascades[root] = cascade

        del self.cascade_roots
        del self.child_dict
        del self.parent_dict
    
    def _build_node_features(self):

        for cascade in self.cascades.values():
            for node in cascade.nodes.values():
                for func in self.feature_extractors:
                    feature_name, value = func(self.nodes_dict, self.cascades, node.node_id, self.news_utc)
                    node.features[feature_name] = value

        del self.nodes_dict

    def reach_by_hours(self, hours):
        
        raise NotImplementedError()

    def get_pyg_data_object(self):
        data = []
        edges = []
        self.pyg_node_mapping = {}

        for cascade in self.cascades.values():
            for node in cascade.nodes.values():
                data.append(node.features)
                self.pyg_node_mapping[node.node_id] = len(data) - 1
                edges.extend([(node.node_id, child) for child in node.children])

        df = pd.DataFrame(data=data)
        df = df[self.features]
        
        self.pyg_node_mapping['news'] = len(data)
        edges.extend([('news', cascade_id) for cascade_id in self.cascades])
        df.loc[len(df)] = [0 for i in range(len(df.columns))]

        feature_matrix = torch.tensor(df.values, dtype=torch.float)
        edges = [(
            self.pyg_node_mapping[i],
            self.pyg_node_mapping[j]
        ) for i,j in edges]
        edges = torch.tensor(edges, dtype=torch.long)

        return Data(x=feature_matrix, edge_index=edges.t().contiguous())
                
    def get_pyvis_diagram(self, path):
        edges = []
        net = Network(directed=True)
        for cascade in self.cascades.values():
            for node in cascade.nodes.values():
                edges.extend([(node.node_id, child) for child in node.children])

                #Shape
                if node.verified:
                    shape = "star"
                else:
                    shape = "circle"

                #Color
                if node.node_type == TWEET:
                    color = "darkgrey"
                elif node.node_type == RETWEET:
                    color = "green"
                elif node.node_type == QUOTE:
                    color = "blue"
                else:
                    color = "lightgreen"
                
                net.add_node(n_id=node.node_id, label=" ", shape=shape, color=color, size=15)

        #News node
        net.add_node(n_id="news", label=" ", shape="circle", color="black", size=1)
        edges.extend([("news", i) for i in self.cascades])
       
        net.add_edges(edges)

        return net.show(f"{path}/{self.news_id}.html")

    def get_visjs_data(self):
        nodes = []
        edges = []
        for cascade in self.cascades.values():
            for node in cascade.nodes.values():
                nodes.append({
                    "id": node.node_id,
                    "type": node.node_type,
                    "username": node.username,
                    "node_url": f"https://twitter.com/i/web/status/{node.node_id}"
                })
                edges.extend([{"from": node.node_id, "to": child} for child in node.children])

        nodes.append("news")
        edges.extend([{"from": "news", "to": cascade} for cascade in self.cascades])

        return nodes, edges

    def get_node_by_id(self, id):
        cascade = self.cascades[self.node_mapping[id]]
        return cascade.nodes[id]

    