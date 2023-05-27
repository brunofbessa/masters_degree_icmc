import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import logging
import pickle
import re

from random import sample, shuffle
from sklearn.utils import shuffle as shuffle_df
from tqdm import trange
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from util import SimplePreprocessing
from upbg import UPBG
from pbg import PBG
from tpbg import TPBG
from gnn import *
from nltk.corpus import reuters
from dataclasses import dataclass
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from chunkdot import cosine_similarity_top_k

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
    is_train: np.array
    filenames: list
    DESCR: str
        
    def __init__(self, target_names: list, data: list, target: np.array, is_train: np.array, filenames: list, DESCR: str=""):
        self.target_names = target_names
        self.data = data
        self.target = target
        self.is_train = is_train
        self.filenames = filenames
        self.DESCR = DESCR


def load_data_bkp(database_name, subset):
    
    try:
    
        if database_name in ['20newsgroups', '20ng']:
            database = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
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
        
        elif database_name in ['bbc_news', 'bbcnews', 'bbc']:
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

            target = np.array(target)
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
            
            target = np.array(target)
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

            target = np.array(target)
            database = Database(target_names, data, target, filenames, DESCR) 
                                    
        elif database_name == 'nsf':
            
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

            target = np.array(target)
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

            target = np.array(target)
            database = Database(target_names, data, target, filenames, DESCR)     
        
        else:
             logger.info(f'Database name provided ({database_name}) is not valid.')
             raise  


        logger.info(f'Loaded data for {database_name} {subset}. {len(target_names)} classes: {str(target_names)}. Number of documents: {len(data)}.')                                    
        return database
    
    except Exception as e:
        logger.error(f"Error getting data: \n{e}")
        raise
 

def load_data(database_name):
    
    try:
    
        if database_name in ['20newsgroups', '20ng']:
            categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space', 'rec.motorcycles']
            database_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
            database_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)
            database_train.is_train = np.ones(len(database_train.data))
            database_test.is_train = np.zeros(len(database_test.data))

            target_names = database_train.target_names
            data = database_train.data + database_test.data
            target = np.append(database_train.target, database_test.target)
            is_train = np.append(database_train.is_train, database_test.is_train)
            filenames = ''
            DESCR = database_train.DESCR

            ind_list = [i for i in range(len(data))]
            shuffle(ind_list)
            data = [data[index] for index in ind_list]
            target = target[ind_list]
            is_train = is_train[ind_list]

            database = Database(target_names, data, target, is_train, filenames, DESCR)
            
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
        
        elif database_name in ['bbc_news', 'bbcnews', 'bbc']:
            dataframe = pd.read_csv('data/bbc_news.csv', sep=',')
            dataframe_train, dataframe_test = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            dataframe_train['is_train'] = 1.0
            dataframe_test['is_train'] = 0.0

            dataframe = pd.concat([dataframe_train, dataframe_test])
            dataframe = shuffle_df(dataframe)
            
            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            is_train = []
            filenames = None
            DESCR = '''All rights, including copyright, in the content of the original articles are owned by the BBC. Consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005. Class Labels: 5 (business, entertainment, politics, sport, tech).'''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])
                is_train.append(row['is_train'])

            target = np.array(target)
            is_train = np.array(is_train)
            database = Database(target_names, data, target, is_train, filenames, DESCR)

        elif database_name in ['ag_news', 'agnews']:
            dataframe_train, dataframe_test = pd.read_csv('data/ag_news_csv/train.csv', sep=',', header=None, names=['category', 'title', 'text'])
            dataframe_train['is_train'] = 1.0
            dataframe_test['is_train'] = 0.0

            dataframe = pd.concat([dataframe_train, dataframe_test])
            dataframe = shuffle_df(dataframe)

            target_names_dict = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
            dataframe['category'] = dataframe['category'].map(target_names_dict)
            
            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            is_train = []
            filenames = None
            DESCR = '''The AG's news topic classification dataset is constructed by choosing 4 largest classes from the original corpus. Each class contains 30,000 training samples and 1,900 testing samples. The total number of training samples is 120,000 and testing 7,600.'''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])
                is_train.append(row['is_train'])
            
            target = np.array(target)
            is_train = np.array(is_train)
            database = Database(target_names, data, target, is_train, filenames, DESCR)
        
        elif database_name == 'classic4':
            dataframe = pd.read_csv('data/classic4.csv', sep=',')
            dataframe_train, dataframe_test = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            dataframe_train['is_train'] = 1.0
            dataframe_test['is_train'] = 0.0

            dataframe = pd.concat([dataframe_train, dataframe_test])
            dataframe = shuffle_df(dataframe)

            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            is_train = []
            filenames = None
            DESCR = '''Classic4 collection [Research, 2010] are composed by 4 distinct collections: CACM (titles and abstracts from the journal Communications of the ACM), CISI (information retrieval papers), CRANFIELD (aeronautical system papers), and MEDLINE (medical journals).'''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])
                is_train.append(row['is_train'])

            target = np.array(target)
            is_train = np.array(is_train)
            database = Database(target_names, data, target, is_train, filenames, DESCR) 
                                    
        elif database_name == 'nsf':          
            dataframe = pd.read_csv('data/nsf.csv', sep=',')
            dataframe_train, dataframe_test = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            dataframe_train['is_train'] = 1.0
            dataframe_test['is_train'] = 0.0

            dataframe = pd.concat([dataframe_train, dataframe_test])
            dataframe = shuffle_df(dataframe)

            categories = ['ecology', 'oceanography', 'politic', 'theory', 'data']
            dataframe = dataframe.loc[dataframe['category'].isin(categories)]

            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            is_train = []
            filenames = None
            DESCR = '''NSF (National Science Foundation) collection [Pazzani and Meyers, 2003] are com- posed by abstracts of grants awarded by the National Science Foundation8 between 1999 and August 2003.'''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])
                is_train.append(row['is_train'])

            target = np.array(target)
            is_train = np.array(is_train)
            database = Database(target_names, data, target, is_train, filenames, DESCR)  
        
        elif database_name == 'webkb': 
            #reduce classes ['course', 'student', 'other', 'faculty', 'project', 'staff', 'department']
            dataframe = pd.read_csv('data/webkb.csv', sep=',')
            dataframe_train, dataframe_test = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            dataframe_train['is_train'] = 1.0
            dataframe_test['is_train'] = 0.0

            dataframe = pd.concat([dataframe_train, dataframe_test])
            dataframe = shuffle_df(dataframe)

            categories = ['course', 'student', 'other', 'staff', 'department']
            dataframe = dataframe.loc[dataframe['category'].isin(categories)]

            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            is_train = []
            filenames = None
            DESCR = '''WebKB colleciton is composed by web pages collected from computer science de- partments of various universities in January 1997 by the World Wide Knowledge Base15 (WebKb) project of the CMU Text Learning Group.'''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])
                is_train.append(row['is_train'])

            target = np.array(target)
            is_train = np.array(is_train)
            database = Database(target_names, data, target, is_train, filenames, DESCR)    

        elif database_name == 'cstr':
            dataframe = pd.read_csv('data/CSTR.csv', sep=',')
            dataframe_train, dataframe_test = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            dataframe_train['is_train'] = 1.0
            dataframe_test['is_train'] = 0.0

            dataframe = pd.concat([dataframe_train, dataframe_test])
            dataframe = shuffle_df(dataframe)

            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            is_train = []
            filenames = None
            DESCR = ''' '''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])
                is_train.append(row['is_train'])

            target = np.array(target)
            is_train = np.array(is_train)
            database = Database(target_names, data, target, is_train, filenames, DESCR)    
      
        elif database_name == 'dmoz_computers': 
            #reduce classes: ['Software', 'Computer', 'Programming', 'Mobile', 'Multimedia', 'Robotics', 'Hardware', 'Graphics', 'Data', 'Artificial', 'Internet', 'Companies', 'Systems', 'CAD', 'Education', 'Security', 'Open', 'Consultants']
            dataframe = pd.read_csv('data/Dmoz-Computers.csv', sep=',')
            dataframe_train, dataframe_test = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            dataframe_train['is_train'] = 1.0
            dataframe_test['is_train'] = 0.0

            dataframe = pd.concat([dataframe_train, dataframe_test])
            dataframe = shuffle_df(dataframe)

            categories = ['Software', 'Hardware', 'Graphics', 'Education', 'Security']
            dataframe = dataframe.loc[dataframe['category'].isin(categories)]

            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            is_train = []
            filenames = None
            DESCR = ''' '''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])
                is_train.append(row['is_train'])

            target = np.array(target)
            is_train = np.array(is_train)
            database = Database(target_names, data, target, is_train, filenames, DESCR)    
      
        elif database_name == 'dmoz_health': 
            #reduce classes: ['Senior', 'Reproductive', 'Public', 'Mental', 'Professions', 'Medicine', 'Pharmacy', 'Alternative', 'Nutrition', 'Addictions', 'Animal', 'Conditions', 'Nursing']
            dataframe = pd.read_csv('data/Dmoz-Health.csv', sep=',')
            dataframe_train, dataframe_test = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            dataframe_train['is_train'] = 1.0
            dataframe_test['is_train'] = 0.0

            dataframe = pd.concat([dataframe_train, dataframe_test])
            dataframe = shuffle_df(dataframe)

            categories = ['Medicine', 'Pharmacy', 'Public', 'Mental', 'Animal']
            dataframe = dataframe.loc[dataframe['category'].isin(categories)]

            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            is_train = []
            filenames = None
            DESCR = ''' '''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])
                is_train.append(row['is_train'])

            target = np.array(target)
            is_train = np.array(is_train)
            database = Database(target_names, data, target, is_train, filenames, DESCR)  

        elif database_name == 'dmoz_science': 
            #reduce classes: ['Chemistry', 'Earth', 'Math', 'Agriculture', 'Physics', 'Social', 'Environment', 'Instruments', 'Science', 'Biology', 'Technology', 'Astronomy']
            dataframe = pd.read_csv('data/Dmoz-Science.csv', sep=',')
            dataframe_train, dataframe_test = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            dataframe_train['is_train'] = 1.0
            dataframe_test['is_train'] = 0.0

            dataframe = pd.concat([dataframe_train, dataframe_test])
            dataframe = shuffle_df(dataframe)

            categories = ['Chemistry', 'Earth', 'Environment', 'Instruments', 'Science']
            dataframe = dataframe.loc[dataframe['category'].isin(categories)]

            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            is_train = []
            filenames = None
            DESCR = ''' '''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])
                is_train.append(row['is_train'])

            target = np.array(target)
            is_train = np.array(is_train)
            database = Database(target_names, data, target, is_train, filenames, DESCR)   

        elif database_name == 'dmoz_sports': 
            #reduce classes: ['Hockey', 'Fencing', 'Basketball', 'Martial', 'Track', 'Water', 'Tennis', 'Paintball', 'Running', 'Cycling', 'Winter', 'Soccer', 'Wrestling', 'Golf', 'Football', 'Softball', 'Gymnastics', 'Baseball', 'Skating', 'Lacrosse', 'Strength', 'Motorsports', 'Bowling', 'Volleyball', 'Flying', 'Cricket', 'Equestrian']
            dataframe = pd.read_csv('data/Dmoz-Sports.csv', sep=',')
            dataframe_train, dataframe_test = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            dataframe_train['is_train'] = 1.0
            dataframe_test['is_train'] = 0.0

            dataframe = pd.concat([dataframe_train, dataframe_test])
            dataframe = shuffle_df(dataframe)

            categories = ['Wrestling', 'Golf', 'Paintball', 'Running', 'Cycling']
            dataframe = dataframe.loc[dataframe['category'].isin(categories)]

            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            is_train = []
            filenames = None
            DESCR = ''' '''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])
                is_train.append(row['is_train'])

            target = np.array(target)
            is_train = np.array(is_train)
            database = Database(target_names, data, target, is_train, filenames, DESCR)   

        elif database_name == 're8':
            dataframe = pd.read_csv('data/re8.csv', sep=',')
            dataframe_train, dataframe_test = train_test_split(dataframe, test_size=0.4, random_state=1234, stratify=dataframe[['category']])
            dataframe_train['is_train'] = 1.0
            dataframe_test['is_train'] = 0.0

            dataframe = pd.concat([dataframe_train, dataframe_test])
            dataframe = shuffle_df(dataframe)

            target_names = dataframe["category"].unique().tolist()
            mapping_categories = {}
            for idx, element in enumerate(target_names):
                mapping_categories[element] = idx
            data = []
            target = []
            is_train = []
            filenames = None
            DESCR = ''' '''
            for index, row in dataframe.iterrows():
                data.append(row['text'])
                target.append(mapping_categories[row['category']])
                is_train.append(row['is_train'])

            target = np.array(target)
            is_train = np.array(is_train)
            database = Database(target_names, data, target, is_train, filenames, DESCR)   

        else:
             logger.info(f'Database name provided ({database_name}) is not valid.')
             raise  


        logger.info(f'Loaded data for {database_name}. {len(database.target_names)} classes: {str(database.target_names)}. Number of documents: {len(database.data)}.')                                    
        return database
    
    except Exception as e:
        logger.error(f"Error getting data: \n{e}")
        raise
              
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

def run_tpbg_on_dataset(database_name):

    try:
    
        database = load_data(database_name=database_name)
        data_preprocessed = SimplePreprocessing().transform(database.data)
        vectorizer = TfidfVectorizer()
        data_vectorized_fit = vectorizer.fit_transform(data_preprocessed)
        y = database.target
        y_train_real = y.copy()
        
        y[database.is_train==0] = -1
        
        def eval(self):        
            self.create_transduction()    
            y_predicted = self.transduction_[database.is_train==0]    
            y_real = y_train_real[database.is_train==0]    
            logger.info(f'Classification_report for TPBG on {database_name}:\n {str(classification_report(y_predicted, y_real, digits=4))}')
            
        K = len(database.target_names)
        tpbg = TPBG(K, alpha=0.05, beta=0.0001, local_max_itr=30,
                         global_max_itr=5, local_threshold=1e-6, global_threshold=1e-6,
                         save_interval=-1, 
                         feature_names=vectorizer.get_feature_names_out(), 
                         target_name=database.target_names, 
                         silence=True, eval_func=eval)   
        tpbg.fit(data_vectorized_fit, y)
        eval(tpbg)
        logger.info(f'Loaded TPBG for {database_name} with K={K}.')
        
        # doc2vec embeddings
        sentences = [re.findall("[a-z\-]+", s.lower()) for s in data_preprocessed]
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
        doc2vec_model = Doc2Vec(documents, vector_size=400, window=10, min_count=1, workers=4)
        doc2vec_model.train(documents, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
        document_features = doc2vec_model.dv.vectors
        logger.info(f'Computed 400 document features with doc2vec.')

        tpbg.document_features = document_features
        tpbg.is_train = database.is_train
        tpbg.n_class = len(database.target_names)

        # Final embeddings
        tpbg_train = TPBG(K)
        tpbg_train.log_A = tpbg.log_A[database.is_train==1]
        tpbg_train.log_B = tpbg.log_B
        tpbg_train.document_features = tpbg.document_features[database.is_train==1]
        tpbg_train.Xc = tpbg.X[database.is_train==1]
        tpbg_train.y = y_train_real[database.is_train==1]
        tpbg_train.n_class = len(database.target_names)

        tpbg_test = TPBG(K)
        tpbg_test.log_A = tpbg.log_A[database.is_train==0]
        tpbg_test.log_B = tpbg.log_B
        tpbg_test.document_features = tpbg.document_features[database.is_train==0]
        tpbg_test.Xc = tpbg.X[database.is_train==0]
        tpbg_test.y = y_train_real[database.is_train==0]
        tpbg_test.n_class = len(database.target_names)
        
        return tpbg_train, tpbg_test

    except Exception as e:
        logger.info(f'Error fitting TPBG for {database_name}: \n {e}')

def run_pbg_on_dataset(database_name, K, K_cosine=0, disable_tqdm=True):

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
        micro = f1_score(predicted_target, database_test.target, average='micro')
        logger.info(f'Loaded pbg for {database_name} (train set) with K={K}. F1: {micro:.4f}.')

        data_vectorized_test_fit = vectorizer.fit_transform(data_preprocessed_test)
        data_vectorized_test = vectorizer.transform(data_preprocessed_test)

        pbg_test = UPBG(K, alpha=0.005, beta=0.001, local_max_itr=50, global_max_itr=10,
            local_threshold=1e-6, global_threshold=1e-6,
            feature_names=vectorizer.get_feature_names_out(), disable_tqdm=disable_tqdm)

        # Hide true labels. Mock labels with predictions from training set
        pbg_test.fit(data_vectorized_test_fit, predicted_target)
        y_pred = pbg_test.predict(data_vectorized_test)
        micro_test = f1_score(y_pred, database_test.target, average='micro')

        #Write true labels with trained embeddings for future GNN test of accuracy
        for index, value in enumerate(pbg_test.y):
            pbg_test.y[index] = np.array(database_test.target[index])


        sentences = [re.findall("[a-z\-]+",s.lower()) for s in data_preprocessed_train+data_preprocessed_test]

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
        doc2vec_model = Doc2Vec(documents, vector_size=K_cosine, window=10, min_count=1, workers=4)
        doc2vec_model.train(documents, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

        document_features = doc2vec_model.dv.vectors

        document_features_train = document_features[0:len(data_preprocessed_train)]
        document_features_test = document_features[len(data_preprocessed_train):]

        pbg_train.document_features = document_features_train
        pbg_test.document_features = document_features_test


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
    try:

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
    
    except Exception as e:
        logger.info(f'Error heterograph for pbg.\n{e}')

def get_heterograph_pbg_features(pbg, doc_features=None):
    try:

        from torch_geometric.data import HeteroData
        from torch_geometric.transforms import RandomLinkSplit, ToUndirected, AddSelfLoops, NormalizeFeatures

        _num_components = pbg.n_components
        
        if doc_features == 'merge':
            _source_x = torch.from_numpy(np.concatenate((pbg.log_A, pbg.document_features), axis=1)).float()
        elif doc_features == 'replace':
            _source_x = torch.from_numpy(pbg.document_features).float()
        else:
            _source_x = torch.from_numpy(pbg.log_A).float()

        _target_x = torch.from_numpy(pbg.log_B).float()
        _adjacency_matrix = pbg.Xc.todense()
        
        logger.info(f'Creating source-target edges.')

        idx = np.where(_adjacency_matrix >0)
        _from_node = np.array(idx[0])
        _to_node = np.array(idx[1])
        
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
        
        logger.info(f'Generated bipartite graph: num. source nodes: {len(_source_x)}, num. target nodes: {len(_target_x)}, num. classes: {pbg.n_class}')

        return heterodata
    
    except Exception as e:
        logger.info(f'Error heterograph for pbg.\n{e}')


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


def test_pipe(model, database_name):
    
    try:
        database_test = load_data(database_name=database_name, subset="test")
        y_pred = model.predict(database_test.data)
        micro = f1_score(y_pred, database_test.target, average='micro')
        return micro

    except Exception as e:
        logger.error(f'Error testing model: \n {e}')  

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


def run_heterognn_splitted(database_name,
                           description, 
                           heterodata_train,
                           heterodata_test,
                           hidden_channels,
                           num_layers,
                           p_dropout,
                           num_epochs, 
                           aggr,
                           version, 
                           loss_function, 
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
                          p_dropout=p_dropout,
                          aggr=aggr, 
                          version=version)
        min_loss = np.inf
        max_acc = 0
        patience = int(num_epochs/2)
        epoch_convergence = 0

        output_list = []
        df = pd.DataFrame(columns=['database_name', 'description', 'hidden_channels', 'num_layers', 'p_dropout', 'loss_function', 'version', 'loss_train', 'micro_train', 'acc_train', 'loss_test', 'micro_test', 'acc_test', 'epoch', 'epoch_convergence', 'elapsed_time'])

        model_name = f"model_{database_name}_{description}_hid_{hidden_channels}_layers_{num_layers}_pdrop_{p_dropout}".replace("=", "_").replace(" ", "_")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        for epoch in trange(num_epochs):
            loss_train, micro_train, acc_train = train_model(model=model, train_dataset=heterodata_train, optimizer=optimizer, loss_function=loss_function)
            loss_test, micro_test, acc_test = test_model(model, heterodata_test, loss_function=loss_function)

            if loss_train <= min_loss and acc_test >= max_acc:
                min_loss = loss_train
                max_acc = acc_test
                best_model = model
                best_loss_test = loss_test
                best_micro_test = micro_test
                best_acc_test = acc_test
                epoch_convergence = epoch

            time_end = timeit.default_timer()
            elapsed_time = round((time_end - time_start) * 10 ** 0, 3)
            output_list = [database_name, description, hidden_channels, num_layers, p_dropout, loss_function, version, loss_train, micro_train, acc_train, loss_test, micro_test, acc_test, epoch, epoch_convergence, elapsed_time]
            row = pd.Series(output_list, index=df.columns)
            df = df.append(row,ignore_index=True) 

            if verbose:
                logger.info(f'\nLoss (train): {loss_train:.4f}, Loss (test): {loss_test:.4f}, F1 (train): {micro_train:.4f}, F1 (test): {micro_test:.4f}')

            if (epoch >= patience or acc_train > 0.99) and (epoch - epoch_convergence) > 500:
                logger.info(f'Early stopping at epoch {epoch}.')
                break

        logger.info(f'Optimal sol. database {database_name}: \nGNN ver. {version}, lf. {loss_function} \nepoch: {epoch_convergence}/{num_epochs}, loss (test): {best_loss_test:.4f}, f1 (test): {best_micro_test:.4f}, acc (test): {best_acc_test:.4f}')
        
        df.to_csv(f'./csv_objects/training/{model_name}.csv', sep=';', decimal=',', index=False)
        
        with open(f'./pickle_objects/models/{model_name}.pickle', 'wb') as f:
            pickle.dump(best_model, f, pickle.HIGHEST_PROTOCOL)
        return best_loss_test, best_micro_test, best_acc_test, epoch_convergence
    
    except Exception as e:
        logger.error(f'Error training model on heterodata: \n {e}')
