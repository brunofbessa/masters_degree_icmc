from functions import *
import pickle

seed_everything(seed=42)

logger.info('Generating heterographs for benchmark datasets')

logger.info('Test pbg on 20ng, K=10')
pbg_20ng_k10_train = load_pbg_train(database_name='20newsgroups', K=10, disable_tqdm=False)
heterodata_pbg_20ng_k10_train = get_heterograph_pbg(pbg_20ng_k10_train)
with open("./pickle_objects/pbg_20ng_k10_train.pickle", "wb") as f:
    pickle.dump(pbg_20ng_k10_train, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k10_train.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k10_train, f, pickle.HIGHEST_PROTOCOL)
logger.info('Test pbg on 20ng , K=10')
pbg_20ng_k10_test = load_pbg_test(database_name='20newsgroups', pbg_model_trained=pbg_20ng_k10_train, K=10, disable_tqdm=False)
heterodata_pbg_20ng_k10_test_full = get_heterograph_pbg(pbg_20ng_k10_test)
heterodata_pbg_20ng_k10_val, heterodata_pbg_20ng_k10_test, heterodata_pbg_20ng_k10_score = split_heterodata(heterodata_pbg_20ng_k10_test_full)
with open("./pickle_objects/pbg_20ng_k10_test.pickle", "wb") as f:
    pickle.dump(pbg_20ng_k10_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k10_test_full.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k10_test_full, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k10_val.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k10_val, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k10_test.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k10_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k10_score.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k10_score, f, pickle.HIGHEST_PROTOCOL)
############    
    
logger.info('Test pbg on 20ng, K=50')
pbg_20ng_k50_train = load_pbg_train(database_name='20newsgroups', K=50, disable_tqdm=False)
heterodata_pbg_20ng_k50_train = get_heterograph_pbg(pbg_20ng_k50_train)
with open("./pickle_objects/pbg_20ng_k50_train.pickle", "wb") as f:
    pickle.dump(pbg_20ng_k50_train, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k50_train.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k50_train, f, pickle.HIGHEST_PROTOCOL)
logger.info('Test pbg on 20ng , K=10')
pbg_20ng_k50_test = load_pbg_test(database_name='20newsgroups', pbg_model_trained=pbg_20ng_k50_train, K=50, disable_tqdm=False)
heterodata_pbg_20ng_k50_test_full = get_heterograph_pbg(pbg_20ng_k50_test)
heterodata_pbg_20ng_k50_val, heterodata_pbg_20ng_k50_test, heterodata_pbg_20ng_k50_score = split_heterodata(heterodata_pbg_20ng_k50_test_full)
with open("./pickle_objects/pbg_20ng_k50_test.pickle", "wb") as f:
    pickle.dump(pbg_20ng_k50_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k50_test_full.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k50_test_full, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k50_val.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k50_val, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k50_test.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k50_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k50_score.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k50_score, f, pickle.HIGHEST_PROTOCOL)
############    

logger.info('Test pbg on 20ng, K=100')
pbg_20ng_k100_train = load_pbg_train(database_name='20newsgroups', K=100, disable_tqdm=False)
heterodata_pbg_20ng_k100_train = get_heterograph_pbg(pbg_20ng_k100_train)
with open("./pickle_objects/pbg_20ng_k100_train.pickle", "wb") as f:
    pickle.dump(pbg_20ng_k100_train, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k100_train.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k100_train, f, pickle.HIGHEST_PROTOCOL)
logger.info('Test pbg on 20ng , K=10')
pbg_20ng_k100_test = load_pbg_test(database_name='20newsgroups', pbg_model_trained=pbg_20ng_k100_train, K=100, disable_tqdm=False)
heterodata_pbg_20ng_k100_test_full = get_heterograph_pbg(pbg_20ng_k100_test)
heterodata_pbg_20ng_k100_val, heterodata_pbg_20ng_k100_test, heterodata_pbg_20ng_k100_score = split_heterodata(heterodata_pbg_20ng_k100_test_full)
with open("./pickle_objects/pbg_20ng_k100_test.pickle", "wb") as f:
    pickle.dump(pbg_20ng_k100_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k100_test_full.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k100_test_full, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k100_val.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k100_val, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k100_test.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k100_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_20ng_k100_score.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_20ng_k100_score, f, pickle.HIGHEST_PROTOCOL)
############   


logger.info('Test pbg on reuters, K=10')
pbg_reuters_k10_train = load_pbg_train(database_name='reuters', K=10, disable_tqdm=False)
heterodata_pbg_reuters_k10_train = get_heterograph_pbg(pbg_reuters_k10_train)
with open("./pickle_objects/pbg_reuters_k10_train.pickle", "wb") as f:
    pickle.dump(pbg_reuters_k10_train, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k10_train.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k10_train, f, pickle.HIGHEST_PROTOCOL)
logger.info('Test pbg on reuters , K=10')
pbg_reuters_k10_test = load_pbg_test(database_name='reuters', pbg_model_trained=pbg_reuters_k10_train, K=10, disable_tqdm=False)
heterodata_pbg_reuters_k10_test_full = get_heterograph_pbg(pbg_reuters_k10_test)
heterodata_pbg_reuters_k10_val, heterodata_pbg_reuters_k10_test, heterodata_pbg_reuters_k10_score = split_heterodata(heterodata_pbg_reuters_k10_test_full)
with open("./pickle_objects/pbg_reuters_k10_test.pickle", "wb") as f:
    pickle.dump(pbg_reuters_k10_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k10_test_full.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k10_test_full, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k10_val.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k10_val, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k10_test.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k10_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k10_score.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k10_score, f, pickle.HIGHEST_PROTOCOL)
############    
    
logger.info('Test pbg on reuters, K=50')
pbg_reuters_k50_train = load_pbg_train(database_name='reuters', K=50, disable_tqdm=False)
heterodata_pbg_reuters_k50_train = get_heterograph_pbg(pbg_reuters_k50_train)
with open("./pickle_objects/pbg_reuters_k50_train.pickle", "wb") as f:
    pickle.dump(pbg_reuters_k50_train, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k50_train.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k50_train, f, pickle.HIGHEST_PROTOCOL)
logger.info('Test pbg on reuters , K=10')
pbg_reuters_k50_test = load_pbg_test(database_name='reuters', pbg_model_trained=pbg_reuters_k50_train, K=50, disable_tqdm=False)
heterodata_pbg_reuters_k50_test_full = get_heterograph_pbg(pbg_reuters_k50_test)
heterodata_pbg_reuters_k50_val, heterodata_pbg_reuters_k50_test, heterodata_pbg_reuters_k50_score = split_heterodata(heterodata_pbg_reuters_k50_test_full)
with open("./pickle_objects/pbg_reuters_k50_test.pickle", "wb") as f:
    pickle.dump(pbg_reuters_k50_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k50_test_full.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k50_test_full, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k50_val.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k50_val, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k50_test.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k50_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k50_score.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k50_score, f, pickle.HIGHEST_PROTOCOL)
############    

logger.info('Test pbg on reuters, K=100')
pbg_reuters_k100_train = load_pbg_train(database_name='reuters', K=100, disable_tqdm=False)
heterodata_pbg_reuters_k100_train = get_heterograph_pbg(pbg_reuters_k100_train)
with open("./pickle_objects/pbg_reuters_k100_train.pickle", "wb") as f:
    pickle.dump(pbg_reuters_k100_train, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k100_train.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k100_train, f, pickle.HIGHEST_PROTOCOL)
logger.info('Test pbg on reuters , K=10')
pbg_reuters_k100_test = load_pbg_test(database_name='reuters', pbg_model_trained=pbg_reuters_k100_train, K=100, disable_tqdm=False)
heterodata_pbg_reuters_k100_test_full = get_heterograph_pbg(pbg_reuters_k100_test)
heterodata_pbg_reuters_k100_val, heterodata_pbg_reuters_k100_test, heterodata_pbg_reuters_k100_score = split_heterodata(heterodata_pbg_reuters_k100_test_full)
with open("./pickle_objects/pbg_reuters_k100_test.pickle", "wb") as f:
    pickle.dump(pbg_reuters_k100_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k100_test_full.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k100_test_full, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k100_val.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k100_val, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k100_test.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k100_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_reuters_k100_score.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_reuters_k100_score, f, pickle.HIGHEST_PROTOCOL)
############   

logger.info('Test pbg on bbc_news, K=10')
pbg_bbc_news_k10_train = load_pbg_train(database_name='bbc_news', K=10, disable_tqdm=False)
heterodata_pbg_bbc_news_k10_train = get_heterograph_pbg(pbg_bbc_news_k10_train)
with open("./pickle_objects/pbg_bbc_news_k10_train.pickle", "wb") as f:
    pickle.dump(pbg_bbc_news_k10_train, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k10_train.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k10_train, f, pickle.HIGHEST_PROTOCOL)
logger.info('Test pbg on bbc_news , K=10')
pbg_bbc_news_k10_test = load_pbg_test(database_name='bbc_news', pbg_model_trained=pbg_bbc_news_k10_train, K=10, disable_tqdm=False)
heterodata_pbg_bbc_news_k10_test_full = get_heterograph_pbg(pbg_bbc_news_k10_test)
heterodata_pbg_bbc_news_k10_val, heterodata_pbg_bbc_news_k10_test, heterodata_pbg_bbc_news_k10_score = split_heterodata(heterodata_pbg_bbc_news_k10_test_full)
with open("./pickle_objects/pbg_bbc_news_k10_test.pickle", "wb") as f:
    pickle.dump(pbg_bbc_news_k10_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k10_test_full.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k10_test_full, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k10_val.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k10_val, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k10_test.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k10_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k10_score.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k10_score, f, pickle.HIGHEST_PROTOCOL)
############    
    
logger.info('Test pbg on bbc_news, K=50')
pbg_bbc_news_k50_train = load_pbg_train(database_name='bbc_news', K=50, disable_tqdm=False)
heterodata_pbg_bbc_news_k50_train = get_heterograph_pbg(pbg_bbc_news_k50_train)
with open("./pickle_objects/pbg_bbc_news_k50_train.pickle", "wb") as f:
    pickle.dump(pbg_bbc_news_k50_train, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k50_train.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k50_train, f, pickle.HIGHEST_PROTOCOL)
logger.info('Test pbg on bbc_news , K=10')
pbg_bbc_news_k50_test = load_pbg_test(database_name='bbc_news', pbg_model_trained=pbg_bbc_news_k50_train, K=50, disable_tqdm=False)
heterodata_pbg_bbc_news_k50_test_full = get_heterograph_pbg(pbg_bbc_news_k50_test)
heterodata_pbg_bbc_news_k50_val, heterodata_pbg_bbc_news_k50_test, heterodata_pbg_bbc_news_k50_score = split_heterodata(heterodata_pbg_bbc_news_k50_test_full)
with open("./pickle_objects/pbg_bbc_news_k50_test.pickle", "wb") as f:
    pickle.dump(pbg_bbc_news_k50_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k50_test_full.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k50_test_full, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k50_val.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k50_val, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k50_test.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k50_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k50_score.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k50_score, f, pickle.HIGHEST_PROTOCOL)
############    

logger.info('Test pbg on bbc_news, K=100')
pbg_bbc_news_k100_train = load_pbg_train(database_name='bbc_news', K=100, disable_tqdm=False)
heterodata_pbg_bbc_news_k100_train = get_heterograph_pbg(pbg_bbc_news_k100_train)
with open("./pickle_objects/pbg_bbc_news_k100_train.pickle", "wb") as f:
    pickle.dump(pbg_bbc_news_k100_train, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k100_train.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k100_train, f, pickle.HIGHEST_PROTOCOL)
logger.info('Test pbg on bbc_news , K=10')
pbg_bbc_news_k100_test = load_pbg_test(database_name='bbc_news', pbg_model_trained=pbg_bbc_news_k100_train, K=100, disable_tqdm=False)
heterodata_pbg_bbc_news_k100_test_full = get_heterograph_pbg(pbg_bbc_news_k100_test)
heterodata_pbg_bbc_news_k100_val, heterodata_pbg_bbc_news_k100_test, heterodata_pbg_bbc_news_k100_score = split_heterodata(heterodata_pbg_bbc_news_k100_test_full)
with open("./pickle_objects/pbg_bbc_news_k100_test.pickle", "wb") as f:
    pickle.dump(pbg_bbc_news_k100_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k100_test_full.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k100_test_full, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k100_val.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k100_val, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k100_test.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k100_test, f, pickle.HIGHEST_PROTOCOL)
with open("./pickle_objects/heterodata_pbg_bbc_news_k100_score.pickle", "wb") as f:
    pickle.dump(heterodata_pbg_bbc_news_k100_score, f, pickle.HIGHEST_PROTOCOL)
############   