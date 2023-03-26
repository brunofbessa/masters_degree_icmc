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
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from util import SimplePreprocessing
from upbg import UPBG
from gnn import *
from nltk.corpus import reuters
from dataclasses import dataclass
from logger import get_logger

import ktrain
from ktrain import text

import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit

logger = get_logger('log_functions')

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
    
        if database_name in ['20newsgroups', '20ng']:
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
        
        elif database_name in ['bbc_news', 'bbcnews']:
            dataframe = pd.read_csv('data/bbc_news.csv', sep=',')

            if subset == 'train':
                dataframe, _ = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            elif subset == 'test':
                _, dataframe = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])            
            
            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            filenames = None
            DESCR = '''All rights, including copyright, in the content of the original articles are owned by the BBC. Consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005. Class Labels: 5 (business, entertainment, politics, sport, tech).'''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])

            database = Database(target_names, data, target, filenames, DESCR)

        elif database_name in ['ag_news', 'agnews']:
            if subset == 'train':
                dataframe = pd.read_csv('data/ag_news_csv/train.csv', sep=',', header=None, names=['category', 'title', 'text'])
            elif subset == 'test':
                dataframe = pd.read_csv('data/ag_news_csv/test.csv', sep=',', header=None, names=['category', 'title', 'text'])
            
            target_names_dict = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
            dataframe['category'] = dataframe['category'].map(target_names_dict)
            
            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            filenames = None
            DESCR = '''The AG's news topic classification dataset is constructed by choosing 4 largest classes from the original corpus. Each class contains 30,000 training samples and 1,900 testing samples. The total number of training samples is 120,000 and testing 7,600.'''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])
            
            database = Database(target_names, data, target, filenames, DESCR)
        
        elif database_name == 'classic4':
            dataframe = pd.read_csv('data/classic4.csv', sep=',')

            if subset == 'train':
                dataframe, _ = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            elif subset == 'test':
                _, dataframe = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])  

            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            filenames = None
            DESCR = '''Classic4 collection [Research, 2010] are composed by 4 distinct collections: CACM (titles and abstracts from the journal Communications of the ACM), CISI (information retrieval papers), CRANFIELD (aeronautical system papers), and MEDLINE (medical journals).'''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])

            database = Database(target_names, data, target, filenames, DESCR) 
                                    
        elif database_name == 'nsf':
            
            selected_target_names = ['ecology', 'economics', 'statistics', 'politic', 'math']
            dataframe = pd.read_csv('data/nsf.csv', sep=',')

            if subset == 'train':
                dataframe, _ = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            elif subset == 'test':
                _, dataframe = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])  

            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            filenames = None
            DESCR = '''NSF (National Science Foundation) collection [Pazzani and Meyers, 2003] are com- posed by abstracts of grants awarded by the National Science Foundation8 between 1999 and August 2003.'''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])

            database = Database(target_names, data, target, filenames, DESCR)     
        
        elif database_name == 'webkb':
            dataframe = pd.read_csv('data/webkb.csv', sep=',')

            if subset == 'train':
                dataframe, _ = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            elif subset == 'test':
                _, dataframe = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])  

            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            filenames = None
            DESCR = '''WebKB colleciton is composed by web pages collected from computer science de- partments of various universities in January 1997 by the World Wide Knowledge Base15 (WebKb) project of the CMU Text Learning Group.'''
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

        predicted_target = pbg_model_trained.predict(data_vectorized_test)
        micro_train = f1_score(predicted_target, database_test.target, average='micro')
        logger.info(f'Performance on pretrained pbg: F1: {micro_train:.2f}.')

        logger.info(f'Performed IT-IDF on data {database_name} ({subset}).')


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

    
    try:
        database_train = load_data(database_name=database_name, subset='train')
        database_test = load_data(database_name=database_name, subset='test')

        data_preprocessed_train = SimplePreprocessing().transform(database_train.data)
        data_preprocessed_test = SimplePreprocessing().transform(database_test.data)
        vectorizer = TfidfVectorizer()
        data_vectorized_train_fit = vectorizer.fit_transform(data_preprocessed_train)
        data_vectorized_test = vectorizer.transform(data_preprocessed_test)
        
        pbg_train = UPBG(K, alpha=0.005, beta=0.001, local_max_itr=50, global_max_itr=10,
           local_threshold=1e-6, global_threshold=1e-6,
           feature_names=vectorizer.get_feature_names_out(), disable_tqdm=disable_tqdm)
        pbg_train.fit(data_vectorized_train_fit, database_train.target)
        
        predicted_target = pbg_train.predict(data_vectorized_test)
        micro_train = f1_score(predicted_target, database_test.target, average='micro')
        logger.info(f'Performance on pretrained pbg on {database_name} with K={K}. F1: {micro_train:.4f}.')
        
        vectorizer = TfidfVectorizer()
        data_vectorized_test_fit = vectorizer.fit_transform(data_preprocessed_test)
        data_vectorized_test = vectorizer.transform(data_preprocessed_test)
        pbg_test = UPBG(K, alpha=0.005, beta=0.001, local_max_itr=50, global_max_itr=10,
               local_threshold=1e-6, global_threshold=1e-6,
               feature_names=vectorizer.get_feature_names_out(), disable_tqdm=disable_tqdm)
        pbg_test.fit(data_vectorized_test_fit, predicted_target)
        y_pred = pbg_test.predict(data_vectorized_test)
        micro_test = f1_score(y_pred, database_test.target, average='micro')

        logger.info(f'Loaded pbg for {database_name} with K={K}. F1: {micro_test:.4f}.')
        return pbg_train, pbg_test
    
    except Exception as e:
        logger.info(f'Error fitting pbg for {database_name}: \n {e}')    

def run_pbg_deprecated(database_name, K=100, disable_tqdm=True):
    
    try:
        database_train = load_data(database_name=database_name, subset='train')
        database_test = load_data(database_name=database_name, subset='test')

        data_preprocessed_train = SimplePreprocessing().transform(database_train.data)
        data_preprocessed_test = SimplePreprocessing().transform(database_test.data)
        vectorizer = TfidfVectorizer()
        data_vectorized_train_fit = vectorizer.fit_transform(data_preprocessed_train)
        data_vectorized_test = vectorizer.transform(data_preprocessed_test)
        data_vectorized_test_fit = vectorizer.fit_transform(data_preprocessed_test)
                
        pbg_train = UPBG(K, alpha=0.005, beta=0.001, local_max_itr=50, global_max_itr=10,
           local_threshold=1e-6, global_threshold=1e-6,
           feature_names=vectorizer.get_feature_names_out(), disable_tqdm=disable_tqdm)
        pbg_train.fit(data_vectorized_train_fit, database_train.target)
        
        predicted_target = pbg_train.predict(data_vectorized_test)
        micro_train = f1_score(predicted_target, database_test.target, average='micro')
        logger.info(f'Performance on pretrained pbg on {database_name} with K={K}. F1: {micro_train:.4f}.')
        
        data_vectorized_test = vectorizer.transform(data_preprocessed_test)
        pbg_test = UPBG(K, alpha=0.005, beta=0.001, local_max_itr=50, global_max_itr=10,
               local_threshold=1e-6, global_threshold=1e-6,
               feature_names=vectorizer.get_feature_names_out(), disable_tqdm=disable_tqdm)
        pbg_test.fit(data_vectorized_test_fit, predicted_target)
        y_pred = pbg_test.predict(data_vectorized_test)
        micro_test = f1_score(y_pred, database_test.target, average='micro')

        logger.info(f'Loaded pbg for {database_name} with K={K}. F1: {micro_test:.4f}.')
        return pbg_train, pbg_test
    
    except Exception as e:
        logger.info(f'Error fitting pbg for {database_name}: \n {e}')    

def run_pbg(database_name, K=100, disable_tqdm=True):
    
    try:
        database_train = load_data(database_name=database_name, subset='train')
        database_test = load_data(database_name=database_name, subset='test')

        data_preprocessed_train = SimplePreprocessing().transform(database_train.data)
        data_preprocessed_test = SimplePreprocessing().transform(database_test.data)
        vectorizer = TfidfVectorizer()
        data_vectorized_train_fit = vectorizer.fit_transform(data_preprocessed_train)
        data_vectorized_test_fit = vectorizer.fit_transform(data_preprocessed_test)
        data_vectorized_test = vectorizer.transform(data_preprocessed_test)
                
        pbg_train = UPBG(K, alpha=0.005, beta=0.001, local_max_itr=50, global_max_itr=10,
           local_threshold=1e-6, global_threshold=1e-6,
           feature_names=vectorizer.get_feature_names_out(), disable_tqdm=disable_tqdm)
        pbg_train.fit(data_vectorized_train_fit, database_train.target)
        
        predicted_target = pbg_train.predict(data_vectorized_test)
        
        data_vectorized_test = vectorizer.transform(data_preprocessed_test)
        pbg_test = UPBG(K, alpha=0.005, beta=0.001, local_max_itr=50, global_max_itr=10,
               local_threshold=1e-6, global_threshold=1e-6,
               feature_names=vectorizer.get_feature_names_out(), disable_tqdm=disable_tqdm)
        
        # Hide true labels. Mock labels with predictions from training set
        pbg_test.fit(data_vectorized_test_fit, predicted_target)
        y_pred = pbg_test.predict(data_vectorized_test)
        micro_test = f1_score(y_pred, database_test.target, average='micro')

        logger.info(f'Loaded pbg for {database_name} with K={K}. F1: {micro_test:.4f}.')
        return pbg_train, pbg_test
    
    except Exception as e:
        logger.info(f'Error fitting pbg for {database_name}: \n {e}')    


def get_lda_train(database_name, K=100):
    try:
        subset='train'
        database = load_data(database_name=database_name, subset=subset)
        cv = CountVectorizer(max_df=0.95, min_df=2, max_features=5000, stop_words='english', ngram_range=(1, 2))
        lda = LDA(n_components=K, max_iter=30, random_state=1, n_jobs=-1)
        rfc = RFC(n_estimators=1000, n_jobs=-1)
        estimators = [("cv", cv), ("lda", lda), ("rfc", rfc)]

        pipe = Pipeline(estimators)
        pipe.fit(database.data, database.target)
        return pipe
    
    except Exception as e:
        logger.info(f'Error fitting lda for {database_name}({subset}): \n {e}')

def get_nmf_train(database_name, K=100):
    try:
        subset='train'
        K=100
        database = load_data(database_name=database_name, subset=subset)
        tfidf = TfidfVectorizer(max_df=0.95, min_df=2, max_features=5000, stop_words='english', ngram_range=(1, 2))
        nmf = NMF(n_components=K, max_iter=30, tol=1e-2, random_state=1)
        rfc = RFC(n_estimators=1000, n_jobs=-1)
        estimators = [("tfidf", tfidf), ("nmf", nmf), ("rfc", rfc)]

        pipe = Pipeline(estimators)
        pipe.fit(database.data, database.target)
        return pipe

    except Exception as e:
        logger.info(f'Error fitting nmf for {database_name}({subset}): \n {e}')

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
    
    logger.info(f'Generated bipartite graph: \nnum. source nodes: {len(_source_x)}, \nnum. target nodes: {len(_target_x)}, \nnum. classes: {pbg.n_class}')

    return heterodata

def load_bert(database_name):
    
    try: 
        database_train = load_data(database_name=database_name, subset="train")
        database_test = load_data(database_name=database_name, subset="test")
        
        (x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(
            x_train=database_train.data,
            y_train=database_train.target,
            x_test=database_test.data, 
            y_test=database_test.target,
            class_names=database_train.target_names,
            preprocess_mode='bert',
            maxlen=350, 
            max_features=35000)
        
        bert_model = text.text_classifier("bert", train_data=(x_train, y_train),
                                          preproc=preproc)
        learner = ktrain.get_learner(bert_model, train_data=(x_train, y_train), 
                                        batch_size=6)
        learner.fit_onecycle(2e-5, 4)
        
        with open(f"./pickle_objects/models/model_bert_{database_name}.pickle", "wb") as f:
            pickle.dump(learner, f, pickle.HIGHEST_PROTOCOL)
        
        logger.info(f'Fitted BERT model on dataset {database_name}.')
        conf_table = learner.validate(val_data=(x_test, y_test),
                            class_names=database_train.target_names)
        
        return conf_table
  
    except Exception as e:
        logger.info(f'Error fitting pbg for {database_name}: \n {e}')

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
    try: 
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.zero_grad()
        out = model(train_dataset.x_dict, train_dataset.edge_index_dict)
        loss = F.cross_entropy(out, train_dataset['source'].y)
        loss.backward()
        optimizer.step()
        return float(loss)
    except Exception as e:
        logger.error(f'Error training model: \n {e}')  

def test_model(model, val_dataset):
    try:
        model.eval()
        out = model(val_dataset.x_dict, val_dataset.edge_index_dict)
        prediction = out.argmax(1).cpu().numpy()
        true_values = val_dataset['source'].y.cpu().numpy()
        micro = f1_score(true_values, prediction, average='weighted', zero_division=0)
        accuracy = accuracy_score(true_values, prediction, normalize=True)
        loss = float(F.cross_entropy(out, val_dataset['source'].y))
        precision = precision_score(true_values, prediction, average=None, zero_division=0)
        return micro, accuracy, loss

    except Exception as e:
        logger.error(f'Error testing model: \n {e}')  

def test_pipe(model, database_name):
    
    try:
        database_test = load_data(database_name=database_name, subset="test")
        y_pred = model.predict(database_test.data)
        micro = f1_score(y_pred, database_test.target, average='micro')
        return micro

    except Exception as e:
        logger.error(f'Error testing model: \n {e}')  

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
            logger.info(f"epoch: {epoch}, loss: {loss:.2f}, f1: {micro:.2f}, acc: {accuracy:.2f}")
        
    return loss, micro, accuracy

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
                           patience=100,
                           verbose=False):

    try:

        import warnings
        import timeit 

        warnings.simplefilter(action='ignore', category=FutureWarning)

        time_start = timeit.default_timer()

        model = HeteroGNN(metadata=heterodata_train.metadata(), 
                          hidden_channels=hidden_channels, 
                          out_channels=heterodata_train['source']['num_classes'],
                          num_layers=num_layers,
                          p_dropout=p_dropout)
        min_loss = np.inf
        max_acc = 0
        loss_array = []
        epoch_convergence = 0
        block_eval_loss = patience // 10

        output_list = []
        df = pd.DataFrame(columns=['database_name', 'hidden_channels', 'num_layers', 'p_dropout', 'loss_train', 'loss_val', 'micro_val', 'acc_val', 'epoch', 'epoch_convergence', 'elapsed_time'])

        model_name = f"model_{database_name}_hid_{hidden_channels}_layers_{num_layers}_pdrop_{p_dropout}".replace("=", "_").replace(" ", "_")

        for epoch in trange(num_epochs, disable=not verbose):
            loss_train = train_model(model=model, train_dataset=heterodata_train, lr=0.001, weight_decay=5e-4)

            micro_train, acc_train, _ = test_model(model, heterodata_val)
            micro_val, acc_val, loss_val = test_model(model, heterodata_val)
            
            loss_array.append(loss_train)
            if loss_train <= min_loss and acc_train >= max_acc:
                min_loss = loss_train
                max_acc = acc_train
                best_model = model
                epoch_convergence += 1
                if verbose:
                    logger.info(f'Best model updated at epoch {epoch}.')
                    logger.info(f"[VAL. SET] epoch: {epoch}, loss: {loss_val:.4f}, f1: {micro_val:.4f}, acc: {acc_val:.4f}")


            time_end = timeit.default_timer()
            elapsed_time = round((time_end - time_start) * 10 ** 0, 3)
            output_list = [database_name, hidden_channels, num_layers, p_dropout, loss_train, loss_val, micro_val, acc_val, epoch, epoch_convergence, elapsed_time]
            row = pd.Series(output_list, index=df.columns)
            df = df.append(row,ignore_index=True) 
            if epoch >= patience:
                tmp_loss_array = loss_array[-block_eval_loss+1: -1]
                if loss_train > np.mean(tmp_loss_array).item():
                    break

        micro_test, acc_test, loss_test = test_model(best_model, heterodata_test)
        logger.info(f"[TEST SET] Optimal solution: epoch: {epoch}, loss: {loss_test:.4f}, f1: {micro_test:.4f}, acc: {acc_test:.4f}")
        
        df.to_csv(f'./csv_objects/training/{model_name}.csv', sep=';', decimal=',', index=False)
        
        with open(f'./pickle_objects/models/{model_name}.pickle', 'wb') as f:
            pickle.dump(best_model, f, pickle.HIGHEST_PROTOCOL)
        return loss_test, micro_test, acc_test, epoch_convergence+1
    
    except Exception as e:
        logger.error(f'Error training model on heterodata: \n {e}')


def experiment_gnn(database_name, 
                        heterodata_pbg_train, 
                        heterodata_pbg_val,
                        heterodata_pbg_test, 
                        num_epochs, 
                        patience,
                        hidden_channels_list,
                        num_layers_list,
                        p_dropout_list,
                        verbose):
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

                    logger.info(f'Experiment init: database_name: {database_name}, hidden_channels: {hidden_channels}, num_layers: {num_layers}, p_dropout: {p_dropout}.')

                    output = run_heterognn_splitted_v2(database_name, 
                                    heterodata_train=heterodata_pbg_train, 
                                    heterodata_val=heterodata_pbg_val,
                                    heterodata_test=heterodata_pbg_test,
                                    hidden_channels=hidden_channels,
                                    num_layers=num_layers,
                                    p_dropout=p_dropout,
                                    num_epochs=num_epochs, 
                                    patience=patience,
                                    verbose=verbose)
                    time_end = timeit.default_timer()
                    elapsed_time = round((time_end - time_start) * 10 ** 0, 3)
                    output_list = [database_name] + [hidden_channels, num_layers, p_dropout] + list(output) + [elapsed_time]
                    row = pd.Series(output_list, index=df.columns)
                    df = df.append(row,ignore_index=True) 
                    
                    logger.info(f'Experiment finished: database_name: {output_list[0]}, hidden_channels: {output_list[1]}, num_layers: {output_list[2]}, p_dropout: {output_list[3]}, loss: {output_list[4]:.2f}, f1: {output_list[5]:.2f}, acc: {output_list[6]:.2f}, epochs: {output_list[7]}, elapsed_time: {output_list[8]}')
        return df
    
    except Exception as e:
        logger.info(f'Error during experiment: \n{e}')

