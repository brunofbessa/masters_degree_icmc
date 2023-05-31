from functions import *
import os
import pickle

logger = get_logger('log_preprocess')

seed_everything(seed=42)

def preprocess_datasets_pbg(databse_name, docf_values, is_preproc=False):
    try:
        if is_preproc == False:
            logger.info(f'Running TPBG on {databse_name}.')
            pbg_train, pbg_test = run_tpbg_on_dataset(database_name=databse_name)

            base_path = f'./pickle_objects/preprocess/tpbg/{database_name}/'
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            
            with open(f'{base_path}/heterodata_pbg_{database_name}_train.pickle', 'wb') as f:
                pickle.dump(pbg_train, f, pickle.HIGHEST_PROTOCOL)
            with open(f'{base_path}/pbg_{database_name}_test.pickle', 'wb') as f:
                pickle.dump(pbg_test, f, pickle.HIGHEST_PROTOCOL)

            logger.info(f'Executed TPBG on {databse_name}. Results saved as pickle objects.')

        else:
            logger.info(f'Already preprocessed pbg objects for {databse_name} with K={K}, Kc={Kc}.')
            with open(f'{base_path}/pbg_{database_name}_train.pickle', 'rb') as f:
                pbg_train = pickle.load(f)
            with open(f'{base_path}/pbg_{database_name}_test.pickle', 'rb') as f:
                pbg_test = pickle.load(f)

        for docf in docf_values:

            description = f'tpbg_docf_{str(docf)}'
    
            heterodata_pbg_train = get_heterograph_pbg_features(pbg_train, doc_features=docf)
            heterodata_pbg_test = get_heterograph_pbg_features(pbg_test, doc_features=docf)
    
            with open(f'{base_path}/heterodata_pbg_{database_name}_{description}_train.pickle', 'wb') as f:
                pickle.dump(heterodata_pbg_train, f, pickle.HIGHEST_PROTOCOL)  
            with open(f'{base_path}/heterodata_pbg_{database_name}_{description}_test.pickle', 'wb') as f:
                pickle.dump(heterodata_pbg_test, f, pickle.HIGHEST_PROTOCOL)

            logger.info(f'Created heterograph for TPBG on {databse_name} ({description}). Results saved as pickle objects.')

    except Exception as e:
        logger.info(f'Error occurred: \n{e}')
        raise

if __name__ == '__main__':

    databse_list = ['webkb']
    docf_values = [None, 'merge', 'replace']

    logger.info('Generating heterographs for benchmark datasets.')

    for database_name in databse_list:
        preprocess_datasets_pbg(database_name, docf_values, is_preproc=False)



