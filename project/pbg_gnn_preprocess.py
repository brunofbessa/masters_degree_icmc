from functions import *
import pickle

logger = get_logger('log_preprocess')

seed_everything(seed=42)

def preprocess_datasets_pbg(databse_name, K):
    try:

        logger.info(f'Running PBG on {databse_name} with K={K}')
        pbg_train, pbg_test = run_pbg(database_name=databse_name, K=K, disable_tqdm=True)
        heterodata_pbg_train = get_heterograph_pbg(pbg_train)
        heterodata_pbg_test_full = get_heterograph_pbg(pbg_test)
        heterodata_pbg_val, heterodata_pbg_test, heterodata_pbg_score = split_heterodata(heterodata_pbg_test_full)

        with open(f'./pickle_objects/preprocess/pbg_{database_name}_k{K}_train.pickle', 'wb') as f:
            pickle.dump(heterodata_pbg_train, f, pickle.HIGHEST_PROTOCOL)
        with open(f'./pickle_objects/preprocess/heterodata_pbg_{database_name}_k{K}_train.pickle', 'wb') as f:
            pickle.dump(heterodata_pbg_train, f, pickle.HIGHEST_PROTOCOL)
        with open(f'./pickle_objects/preprocess/pbg_{database_name}_k{K}_test.pickle', 'wb') as f:
            pickle.dump(pbg_test, f, pickle.HIGHEST_PROTOCOL)
        with open(f'./pickle_objects/preprocess/heterodata_pbg_{database_name}_k{K}_test_full.pickle', 'wb') as f:
            pickle.dump(heterodata_pbg_test_full, f, pickle.HIGHEST_PROTOCOL)
        with open(f'./pickle_objects/preprocess/heterodata_pbg_{database_name}_k{K}_val.pickle', 'wb') as f:
            pickle.dump(heterodata_pbg_val, f, pickle.HIGHEST_PROTOCOL)
        with open(f'./pickle_objects/preprocess/heterodata_pbg_{database_name}_k{K}_test.pickle', 'wb') as f:
            pickle.dump(heterodata_pbg_test, f, pickle.HIGHEST_PROTOCOL)
        with open(f'./pickle_objects/preprocess/heterodata_pbg_{database_name}_k{K}_score.pickle', 'wb') as f:
            pickle.dump(heterodata_pbg_score, f, pickle.HIGHEST_PROTOCOL)

        logger.info(f'Executed PBG on {databse_name} with K={K}. Results saved as pickle objects.')

    except Exception as e:
        logger.info(f'Error occurred: \n{e}')
        raise

if __name__ == '__main__':

    databse_list = ['20ng', 'agnews', 'reuters', 'bbcnews', 'classic4', 'nsf', 'webkb']
    K_values = [10, 50, 100]
    logger.info('Generating heterographs for benchmark datasets.')

    for database_name in databse_list:
        for K in K_values:
            preprocess_datasets_pbg(database_name, K)

