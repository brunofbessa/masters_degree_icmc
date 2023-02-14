import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import logging
import pickle

from random import sample
from tqdm import trange
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.svm import SVC
from util import SimplePreprocessing
from upbg import UPBG
from nltk.corpus import reuters
from dataclasses import dataclass
from logger import get_logger

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, to_hetero
from torch_geometric.transforms import RandomLinkSplit


logger = get_logger('log')

@dataclass
class Database:
    """Class mimicking 20ng from sckikit learn."""
    
    target_names: list
    data: list
    target: np.array
    filenames: list
    DESCR: str
        
    def __init__(self, target_names: list, data: list, target: np.array, filenames: list, DESCR: str=""):
        self.target_names = target_names
        self.data = data
        self.target = target
        self.filenames = filenames
        self.DESCR = DESCR


        
def load_data(database_name, subset):
    
    try:
    
        if database_name == '20newsgroups':
            categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space', 'misc.forsale']
            database = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'), 
                                      categories=categories)
            target_names = database.target_names
            data = database.data
            target = database.target
            filenames = ''
            DESCR = database.DESCR
            
        elif database_name == 'reuters':
            
            documents = reuters.fileids()
            proportion = 1
            documents = sample(documents, int(proportion * len(documents)))
            documents_subset = [d for d in documents if d.startswith(subset)]
            
            target_names = ['acq', 'alum', 'barley', 'bop', 'carcass']
            data = []
            target = []
            filenames = []
            DESCR = '''The copyright for the text of newswire articles and Reuters annotations in the Reuters-21578 collection resides with Reuters Ltd. Reuters Ltd. and Carnegie Group, Inc. have agreed to allow the free distribution of this data *for research purposes only*. If you publish results based on this data set, please acknowledge its use, refer to the data set by the name 'Reuters-21578, Distribution 1.0', and inform your readers of the current location of the data set.'''
            for idx, element in enumerate(target_names):
                for doc_id in documents_subset:
                    data.append(reuters.raw(doc_id))
                    target.append(idx)
                    filenames.append(doc_id)
            target = np.array(target)

            database = Database(target_names, data, target, filenames, DESCR)

        elif database_name == 'bbc_news':
            dataframe = pd.read_csv('data/bbc_news.csv', sep=',')
            if subset == 'train':
                dataframe = dataframe.sample(frac=0.6, random_state=1234)
            elif subset == 'test':
                dataframe = dataframe.sample(frac=0.4, random_state=1234)            
            
            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            filenames = None
            DESCR = '''All rights, including copyright, in the content of the original articles are owned by the BBC. Consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005. Class Labels: 5 (business, entertainment, politics, sport, tech)'''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])

            database = Database(target_names, data, target, filenames, DESCR)
        
        logger.info(f'Loaded data for {database_name} {subset}. {len(target_names)} classes: {str(target_names)}. Number of documents: {len(data)}.')
        return database

    except Exception as e:
        logger.error(f"Error getting data: \n{e}")
        
              
def load_pbg_train(database_name, K=100, disable_tqdm=True):
    
    try: 
        subset='train'
        database = load_data(database_name=database_name, subset=subset)

        data_preprocessed = SimplePreprocessing().transform(database.data)
        vectorizer = TfidfVectorizer()
        data_vectorized_fit = vectorizer.fit_transform(data_preprocessed)
        
        logger.info(f'Performed IT-IDF on data {database_name} ({subset}).')

        K=K
        pbg = UPBG(K, alpha=0.005, beta=0.001, local_max_itr=50, global_max_itr=10,
               local_threshold=1e-6, global_threshold=1e-6,
               feature_names=vectorizer.get_feature_names_out(), disable_tqdm=disable_tqdm)

        pbg.fit(data_vectorized_fit, database.target)

        logger.info(f'Fitted pbg for {database_name} ({subset}) with K={K}.')
        return pbg
    
    except Exception as e:
        logger.info(f'Error fitting pbg for {database_name} ({subset}): \n {e}')

def load_pbg_test(database_name, pbg_model_trained=None, K=100, disable_tqdm=True):
    
    try:
        subset='test'
        database_train = load_data(database_name=database_name, subset='train')
        database_test = load_data(database_name=database_name, subset=subset)

        data_preprocessed_train = SimplePreprocessing().transform(database_train.data)
        data_preprocessed_test = SimplePreprocessing().transform(database_test.data)
        vectorizer = TfidfVectorizer()
        data_vectorized_train_fit = vectorizer.fit_transform(data_preprocessed_train)
        data_vectorized_test_fit = vectorizer.fit_transform(data_preprocessed_test)
        data_vectorized_test = vectorizer.transform(data_preprocessed_test)

        logger.info(f'Performed IT-IDF on data {database_name} ({subset}).')
        predicted_target = pbg_model_trained.predict(data_vectorized_test)

        K=K
        pbg = UPBG(K, alpha=0.005, beta=0.001, local_max_itr=50, global_max_itr=10,
               local_threshold=1e-6, global_threshold=1e-6,
               feature_names=vectorizer.get_feature_names_out(), disable_tqdm=disable_tqdm)
        pbg.fit(data_vectorized_test_fit, predicted_target)

        y_pred = pbg.predict(data_vectorized_test)
        micro = f1_score(y_pred, database_test.target, average='micro')

        logger.info(f'Loaded pbg for {database_name}({subset}) with K={K}. F1: {micro:.2f}.')
        return pbg
    
    except Exception as e:
        logger.info(f'Error fitting pbg for {database_name}({subset}): \n {e}')
        
        
def get_heterograph_pbg(pbg):

    from torch_geometric.data import HeteroData
    from torch_geometric.transforms import RandomLinkSplit, ToUndirected, AddSelfLoops, NormalizeFeatures

    _num_components = pbg.n_components

    _source_x = torch.from_numpy(pbg.log_A).float()
    _target_x = torch.from_numpy(pbg.log_B).float()

    _adjacency_matrix = pbg.Xc.todense()
    _num_rows, _num_columns = _adjacency_matrix.shape  
    _from_node = []
    _to_node = []
    
    for row in range(_num_rows):
        for column in range(_num_columns):
            if _adjacency_matrix[row, column] > 0:
                _from_node.append(row)
                _to_node.append(column)

    _from_node = np.array(_from_node)
    _to_node = np.array(_to_node)
    
    _edge_index = torch.concat((torch.from_numpy(_from_node).long(), 
                                torch.from_numpy(_to_node).long()))
    _edge_index = _edge_index.reshape(-1, _from_node.shape[0]).long()
    _y = torch.from_numpy(pbg.y).long()

    heterodata = HeteroData(
        {'source': {'x': _source_x, 'y': _y},
        'target': {'x': _target_x}},
        source__edge__target={'edge_index': _edge_index}
    )
    
    heterodata['source'].num_nodes = len(_source_x)
    heterodata['target'].num_nodes = len(_target_x)
    heterodata['source'].num_classes = pbg.n_class

    heterodata = ToUndirected()(heterodata)
    heterodata = AddSelfLoops()(heterodata)
    heterodata = NormalizeFeatures()(heterodata)
    
    logger.info(f'Generated bipartite graph: \n{heterodata}')
    return heterodata

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class HeteroGNN(torch.nn.Module):
       
    def __init__(self, metadata, hidden_channels, out_channels, num_layers, p_dropout):
        super().__init__()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels,
        self.num_layers = num_layers
        self.p_dropout = p_dropout

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels) for edge_type in metadata[1]
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.p_dropout) for key, x in x_dict.items()}
        return self.lin(x_dict["source"])
        
    
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

def seed_everything(seed: int):
    import random
    
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    

def train_model(model, train_dataset, lr=0.01, weight_decay=5e-4):
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()
    out = model(train_dataset.x_dict, train_dataset.edge_index_dict)
    loss = F.cross_entropy(out, train_dataset['source'].y)
    loss.backward()
    optimizer.step()
    return float(loss)

def test_model(model, val_dataset):
    model.eval()
    out = model(val_dataset.x_dict, val_dataset.edge_index_dict)
    prediction = out.argmax(1).cpu().numpy()
    true_values = val_dataset['source'].y.cpu().numpy()
    micro = f1_score(true_values, prediction, average='weighted', zero_division=0)
    accuracy = accuracy_score(true_values, prediction, normalize=True)
    precision = precision_score(true_values, prediction, average=None, zero_division=0)

    return micro, accuracy



def run_heterognn(heterodata, num_epochs=100, verbose=False):
    transform = RandomLinkSplit(num_val=0.05, 
                           num_test=0.1, 
                           neg_sampling_ratio=0.0,
                           edge_types=[('source', 'edge', 'target')], 
                           rev_edge_types=[('target', 'rev_edge', 'source')])
    train_dataset, val_dataset, test_dataset = transform(heterodata)
    
    model = HeteroGNN(heterodata.metadata(), hidden_channels=64, out_channels=heterodata['source']['num_classes'],
                  num_layers=3)
    for epoch in range(num_epochs):
        loss = train_model(model, train_dataset)
        micro, accuracy = test_model(model, val_dataset)
        if verbose:
            print(f"epoch: {epoch}, loss: {loss:.2f}, f1: {micro:.2f}, acc: {accuracy:.2f}")
        
    return loss, micro, accuracy

def run_heterognn_splitted(database_name,
                           heterodata_train,
                           heterodata_test,
                           hidden_channels,
                           num_layers,
                           p_dropout=0.2,
                           num_epochs=300, 
                           patience=10,
                           verbose=False):
    
    try:
        patience = int(max(patience, 10))
        transform = RandomLinkSplit(num_val=0.05, 
                               num_test=0.1, 
                               neg_sampling_ratio=0.0,
                               edge_types=[('source', 'edge', 'target')], 
                               rev_edge_types=[('target', 'rev_edge', 'source')])
        val_dataset, test_dataset, score_dataset = transform(heterodata_test)

        model = HeteroGNN(metadata=heterodata_train.metadata(), 
                          hidden_channels=hidden_channels, 
                          out_channels=heterodata_train['source']['num_classes'],
                          num_layers=num_layers,
                          p_dropout=p_dropout)
        min_loss = np.inf
        loss_array = []
        block_eval_loss = patience // 2
             
        for epoch in trange(num_epochs, disable=not verbose):
            loss = train_model(model, heterodata_train)
            loss_array.append(loss)
            if loss <= min_loss:
                best_model = model
                min_loss = loss

            _micro, _accuracy = test_model(model, val_dataset)
            if verbose:
                logger.info(f"epoch: {epoch}, loss{loss:.2f}, f1: {_micro:.2f}, acc: {_accuracy:.2f}")
            if epoch >= patience:
                tmp_loss_array = loss_array[-block_eval_loss+1: -1]
                if loss > np.mean(tmp_loss_array).item():
                # Convergence
                    if verbose:
                        logger.info(f"Optimal solution: epoch: {epoch}, loss{loss:.2f}, f1: {_micro:.2f}, acc: {_accuracy:.2f}")
                    break

            micro, accuracy = test_model(best_model, test_dataset)

        with open(f'./pickle_objects/model_heterognn_{database_name}.pickle', 'wb') as f:
            pickle.dump(best_model, f, pickle.HIGHEST_PROTOCOL)
        return loss, micro, accuracy, epoch+1
    
    except Exception as e:
        logger.error(f'Error training model on heterodata: \n {e}')


def split_heterodata(heterodata): 
    try:

        transform = RandomLinkSplit(num_val=0, 
                        num_test=0, 
                        disjoint_train_ratio=1,
                        edge_types=[('source', 'edge', 'target')], 
                        rev_edge_types=[('target', 'rev_edge', 'source')])

        heterodata_splitted_test, heterodata_splitted_val, heterodata_splitted_score = transform(heterodata)

        return heterodata_splitted_test, heterodata_splitted_val, heterodata_splitted_score

    except Exception as e:
        logger.error(f'Error splitting heterogeneous graph: \n {e}')

def run_heterognn_splitted_v2(database_name,
                           heterodata_train,
                           heterodata_val,
                           heterodata_test,
                           hidden_channels,
                           num_layers,
                           p_dropout=0.2,
                           num_epochs=300, 
                           patience=10,
                           verbose=False):
    
    try:
        patience = int(max(patience, 10))

        model = HeteroGNN(metadata=heterodata_train.metadata(), 
                          hidden_channels=hidden_channels, 
                          out_channels=heterodata_train['source']['num_classes'],
                          num_layers=num_layers,
                          p_dropout=p_dropout)
        min_loss = np.inf
        loss_array = []
        block_eval_loss = patience // 2
             
        for epoch in trange(num_epochs, disable=not verbose):
            loss = train_model(model, heterodata_train)
            loss_array.append(loss)
            if loss <= min_loss:
                best_model = model
                min_loss = loss

            _micro, _accuracy = test_model(model, heterodata_val)
            if verbose:
                logger.info(f"epoch: {epoch}, loss{loss:.2f}, f1: {_micro:.2f}, acc: {_accuracy:.2f}")
            if epoch >= patience:
                tmp_loss_array = loss_array[-block_eval_loss+1: -1]
                if loss > np.mean(tmp_loss_array).item():
                # Convergence
                    if verbose:
                        logger.info(f"Optimal solution: epoch: {epoch}, loss{loss:.2f}, f1: {_micro:.2f}, acc: {_accuracy:.2f}")
                    break

            micro, accuracy = test_model(best_model, heterodata_test)

        with open(f'./pickle_objects/model_heterognn_{database_name}.pickle', 'wb') as f:
            pickle.dump(best_model, f, pickle.HIGHEST_PROTOCOL)
        return loss, micro, accuracy, epoch+1
    
    except Exception as e:
        logger.error(f'Error training model on heterodata: \n {e}')


def experiment_gnn(database_name, 
                        heterodata_pbg_train, 
                        heterodata_pbg_val,
                        heterodata_pbg_test, 
                        num_epochs=300, 
                        patience=10,
                        hidden_channels_list=[50, 100],
                        num_layers_list=[3, 4],
                        p_dropout_list=[0.1, 0.2, 0.5, 0.7],
                        verbose=False):
    try:
        import warnings
        import timeit 

        warnings.simplefilter(action='ignore', category=FutureWarning)

        output_list = []
        df = pd.DataFrame(columns=['database_name', 'hidden_channels', 'num_layers', 'p_dropout', 'loss', 'f1', 'acc', 'epochs', 'elapsed_time'])

        for hidden_channels in hidden_channels_list:
            for num_layers in num_layers_list:
                for p_dropout in p_dropout_list:
                    time_start = timeit.default_timer()
                    output = run_heterognn_splitted_v2(database_name, 
                                    heterodata_train=heterodata_pbg_train, 
                                    heterodata_val=heterodata_pbg_val,
                                    heterodata_test=heterodata_pbg_test,
                                    hidden_channels=hidden_channels,
                                    num_layers=num_layers,
                                    p_dropout=p_dropout,
                                    num_epochs=num_epochs, 
                                    patience=patience,
                                    verbose=False)
                    time_end = timeit.default_timer()
                    elapsed_time = round((time_end - time_start) * 10 ** 0, 3)
                    output_list = [database_name] + [hidden_channels, num_layers, p_dropout] + list(output) + [elapsed_time]
                    row = pd.Series(output_list, index=df.columns)
                    df = df.append(row,ignore_index=True) 
                    if verbose:
                        print(f'database_name: {output_list[0]}, hidden_channels: {output_list[1]}, num_layers: {output_list[2]}, p_dropout: {output_list[3]}, loss: {output_list[4]:.2f}, f1: {output_list[5]:.2f}, acc: {output_list[6]:.2f}, epochs: {output_list[7]}, elapsed_time: {output_list[8]}')
        return df
    
    except Exception as e:
        logger.info(f'Error during experiment: \n{e}')

