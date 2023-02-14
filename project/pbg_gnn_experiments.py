from functions import *
import pickle

seed_everything(seed=42)
logger.info("Loading preprocessed bipartite graphs")

with open('./pickle_objects/heterodata_pbg_20ng_k10_test.pickle', 'rb') as f:
    heterodata_pbg_20ng_k10_test = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_20ng_k10_train.pickle', 'rb') as f:
    heterodata_pbg_20ng_k10_train = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_20ng_k10_val.pickle', 'rb') as f:
    heterodata_pbg_20ng_k10_val = pickle.load(f)

with open('./pickle_objects/heterodata_pbg_20ng_k50_test.pickle', 'rb') as f:
    heterodata_pbg_20ng_k50_test = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_20ng_k50_train.pickle', 'rb') as f:
    heterodata_pbg_20ng_k50_train = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_20ng_k50_val.pickle', 'rb') as f:
    heterodata_pbg_20ng_k50_val = pickle.load(f)
        
with open('./pickle_objects/heterodata_pbg_20ng_k100_test.pickle', 'rb') as f:
    heterodata_pbg_20ng_k100_test = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_20ng_k100_train.pickle', 'rb') as f:
    heterodata_pbg_20ng_k100_train = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_20ng_k100_val.pickle', 'rb') as f:
    heterodata_pbg_20ng_k100_val = pickle.load(f)
        
with open('./pickle_objects/heterodata_pbg_bbc_news_k10_test.pickle', 'rb') as f:
    heterodata_pbg_bbc_news_k10_test = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_bbc_news_k10_train.pickle', 'rb') as f:
    heterodata_pbg_bbc_news_k10_train = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_bbc_news_k10_val.pickle', 'rb') as f:
    heterodata_pbg_bbc_news_k10_val = pickle.load(f)
        
with open('./pickle_objects/heterodata_pbg_bbc_news_k50_test.pickle', 'rb') as f:
    heterodata_pbg_bbc_news_k50_test = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_bbc_news_k50_train.pickle', 'rb') as f:
    heterodata_pbg_bbc_news_k50_train = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_bbc_news_k50_val.pickle', 'rb') as f:
    heterodata_pbg_bbc_news_k50_val = pickle.load(f)
        
with open('./pickle_objects/heterodata_pbg_bbc_news_k100_test.pickle', 'rb') as f:
    heterodata_pbg_bbc_news_k100_test = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_bbc_news_k100_train.pickle', 'rb') as f:
    heterodata_pbg_bbc_news_k100_train = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_bbc_news_k100_val.pickle', 'rb') as f:
    heterodata_pbg_bbc_news_k100_val = pickle.load(f)

with open('./pickle_objects/heterodata_pbg_reuters_k10_test.pickle', 'rb') as f:
    heterodata_pbg_reuters_k10_test = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_reuters_k10_train.pickle', 'rb') as f:
    heterodata_pbg_reuters_k10_train = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_reuters_k10_val.pickle', 'rb') as f:
    heterodata_pbg_reuters_k10_val = pickle.load(f)
        
with open('./pickle_objects/heterodata_pbg_reuters_k50_test.pickle', 'rb') as f:
    heterodata_pbg_reuters_k50_test = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_reuters_k50_train.pickle', 'rb') as f:
    heterodata_pbg_reuters_k50_train = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_reuters_k50_val.pickle', 'rb') as f:
    heterodata_pbg_reuters_k50_val = pickle.load(f)

with open('./pickle_objects/heterodata_pbg_reuters_k100_test.pickle', 'rb') as f:
    heterodata_pbg_reuters_k100_test = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_reuters_k100_train.pickle', 'rb') as f:
    heterodata_pbg_reuters_k100_train = pickle.load(f)
with open('./pickle_objects/heterodata_pbg_reuters_k100_val.pickle', 'rb') as f:
    heterodata_pbg_reuters_k100_val = pickle.load(f)

hidden_channels_list=[10, 50, 100, 200]
num_layers_list=[2, 3]
p_dropout_list=[0.1, 0.15, 0.2]
patience=10

logger.info("Running experiments on 20newsgroups K=10")
df_experiment_20ng_k10 = experiment_gnn(database_name='20newsgroups',
                        heterodata_pbg_train=heterodata_pbg_20ng_k10_train,
                        heterodata_pbg_val=heterodata_pbg_20ng_k10_val,
                        heterodata_pbg_test=heterodata_pbg_20ng_k10_test,
                        hidden_channels_list=hidden_channels_list,
                        num_layers_list=num_layers_list,
                        p_dropout_list=p_dropout_list,
                        patience=patience,
                        verbose=False)
with open("./pickle_objects/df_experiment_20ng_k10.pickle", "wb") as f:
    pickle.dump(df_experiment_20ng_k10, f, pickle.HIGHEST_PROTOCOL)

logger.info("Running experiments on 20newsgroups k=50")
df_experiment_20ng_k50 = experiment_gnn(database_name='20newsgroups',
                        heterodata_pbg_train=heterodata_pbg_20ng_k50_train,
                        heterodata_pbg_val=heterodata_pbg_20ng_k50_val,
                        heterodata_pbg_test=heterodata_pbg_20ng_k50_test,
                        hidden_channels_list=hidden_channels_list,
                        num_layers_list=num_layers_list,
                        p_dropout_list=p_dropout_list,
                        patience=patience,
                        verbose=False)
with open("./pickle_objects/df_experiment_20ng_k50.pickle", "wb") as f:
    pickle.dump(df_experiment_20ng_k50, f, pickle.HIGHEST_PROTOCOL)

logger.info("Running experiments on 20newsgroups K=100")
df_experiment_20ng_k100 = experiment_gnn(database_name='20newsgroups',
                        heterodata_pbg_train=heterodata_pbg_20ng_k100_train,
                        heterodata_pbg_val=heterodata_pbg_20ng_k100_val,
                        heterodata_pbg_test=heterodata_pbg_20ng_k100_test,
                        hidden_channels_list=hidden_channels_list,
                        num_layers_list=num_layers_list,
                        p_dropout_list=p_dropout_list,
                        patience=patience,
                        verbose=False)
with open("./pickle_objects/df_experiment_20ng_k100.pickle", "wb") as f:
    pickle.dump(df_experiment_20ng_k100, f, pickle.HIGHEST_PROTOCOL)

logger.info("Running experiments on reuters K=10")
df_experiment_reuters_k10 = experiment_gnn(database_name='reuters',
                        heterodata_pbg_train=heterodata_pbg_reuters_k10_train,
                        heterodata_pbg_val=heterodata_pbg_reuters_k10_val,
                        heterodata_pbg_test=heterodata_pbg_reuters_k10_test,
                        hidden_channels_list=hidden_channels_list,
                        num_layers_list=num_layers_list,
                        p_dropout_list=p_dropout_list,
                        patience=patience,
                        verbose=False)
with open("./pickle_objects/df_experiment_reuters_k10.pickle", "wb") as f:
    pickle.dump(df_experiment_reuters_k10, f, pickle.HIGHEST_PROTOCOL)

logger.info("Running experiments on reuters k=50")
df_experiment_reuters_k50 = experiment_gnn(database_name='reuters',
                        heterodata_pbg_train=heterodata_pbg_reuters_k50_train,
                        heterodata_pbg_val=heterodata_pbg_reuters_k50_val,
                        heterodata_pbg_test=heterodata_pbg_reuters_k50_test,
                        hidden_channels_list=hidden_channels_list,
                        num_layers_list=num_layers_list,
                        p_dropout_list=p_dropout_list,
                        patience=patience,
                        verbose=False)
with open("./pickle_objects/df_experiment_reuters_k50.pickle", "wb") as f:
    pickle.dump(df_experiment_reuters_k50, f, pickle.HIGHEST_PROTOCOL)

logger.info("Running experiments on reuters K=100")
df_experiment_reuters_k100 = experiment_gnn(database_name='reuters',
                        heterodata_pbg_train=heterodata_pbg_reuters_k100_train,
                        heterodata_pbg_val=heterodata_pbg_reuters_k100_val,
                        heterodata_pbg_test=heterodata_pbg_reuters_k100_test,
                        hidden_channels_list=hidden_channels_list,
                        num_layers_list=num_layers_list,
                        p_dropout_list=p_dropout_list,
                        patience=patience,
                        verbose=False)
with open("./pickle_objects/df_experiment_reuters_k100.pickle", "wb") as f:
    pickle.dump(df_experiment_reuters_k100, f, pickle.HIGHEST_PROTOCOL)

logger.info("Running experiments on bbc_news K=10")
df_experiment_bbc_news_k10 = experiment_gnn(database_name='bbc_news',
                        heterodata_pbg_train=heterodata_pbg_bbc_news_k10_train,
                        heterodata_pbg_val=heterodata_pbg_bbc_news_k10_val,
                        heterodata_pbg_test=heterodata_pbg_bbc_news_k10_test,
                        hidden_channels_list=hidden_channels_list,
                        num_layers_list=num_layers_list,
                        p_dropout_list=p_dropout_list,
                        patience=patience,
                        verbose=False)
with open("./pickle_objects/df_experiment_bbc_news_k10.pickle", "wb") as f:
    pickle.dump(df_experiment_bbc_news_k10, f, pickle.HIGHEST_PROTOCOL)

logger.info("Running experiments on reuters k=50")
df_experiment_bbc_news_k50 = experiment_gnn(database_name='bbc_news',
                        heterodata_pbg_train=heterodata_pbg_bbc_news_k50_train,
                        heterodata_pbg_val=heterodata_pbg_bbc_news_k50_val,
                        heterodata_pbg_test=heterodata_pbg_bbc_news_k50_test,
                        hidden_channels_list=hidden_channels_list,
                        num_layers_list=num_layers_list,
                        p_dropout_list=p_dropout_list,
                        patience=patience,
                        verbose=False)
with open("./pickle_objects/df_experiment_bbc_news_k50.pickle", "wb") as f:
    pickle.dump(df_experiment_bbc_news_k50, f, pickle.HIGHEST_PROTOCOL)

logger.info("Running experiments on bbc_news K=100")
df_experiment_bbc_news_k100 = experiment_gnn(database_name='bbc_news',
                        heterodata_pbg_train=heterodata_pbg_bbc_news_k100_train,
                        heterodata_pbg_val=heterodata_pbg_bbc_news_k100_val,
                        heterodata_pbg_test=heterodata_pbg_bbc_news_k100_test,
                        hidden_channels_list=hidden_channels_list,
                        num_layers_list=num_layers_list,
                        p_dropout_list=p_dropout_list,
                        patience=patience,
                        verbose=False)
with open("./pickle_objects/df_experiment_bbc_news_k100.pickle", "wb") as f:
    pickle.dump(df_experiment_bbc_news_k100, f, pickle.HIGHEST_PROTOCOL)