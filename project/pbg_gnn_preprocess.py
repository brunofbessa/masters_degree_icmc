from functions import *
import pickle

logger = get_logger('log_preprocess')

seed_everything(seed=42)

def preprocess_datasets_pbg(databse_name, K, Kc, docf_values, docst_values, preproc_pbg=False):
    try:
        if preproc_pbg == False:
            logger.info(f'Running PBG on {databse_name} with K={K}, Kc={Kc}')
            pbg_train, pbg_test = run_pbg_on_dataset(database_name=databse_name, K=K, K_cosine=Kc, disable_tqdm=True)
            
            with open(f'./pickle_objects/preprocess/{database_name}/pbg_{database_name}_K_{K}_Kc_{Kc}_train.pickle', 'wb') as f:
                pickle.dump(pbg_train, f, pickle.HIGHEST_PROTOCOL)
            with open(f'./pickle_objects/preprocess/{database_name}/pbg_{database_name}_K_{K}_Kc_{Kc}_test.pickle', 'wb') as f:
                pickle.dump(pbg_test, f, pickle.HIGHEST_PROTOCOL)

        else:
            logger.info(f'Already preprocessed pbg objects for {databse_name} with K={K}, Kc={Kc}.')
            with open(f'./pickle_objects/preprocess/{database_name}/pbg_{database_name}_K_{K}_Kc_{Kc}_train.pickle', 'rb') as f:
                pbg_train = pickle.load(f)
            with open(f'./pickle_objects/preprocess/{database_name}/pbg_{database_name}_K_{K}_Kc_{Kc}_test.pickle', 'rb') as f:
                pbg_test = pickle.load(f)

        for docf in docf_values:
            for docst in docst_values:

              description = f'K_{K}_Kc_{Kc}_docf_{str(docf)}_docst_{str(docst)}'
        
              heterodata_pbg_train = get_heterograph_pbg_features(pbg_train, doc_features=docf, doc_similarity_thres=docst)
              heterodata_pbg_test = get_heterograph_pbg_features(pbg_test, doc_features=docf, doc_similarity_thres=docst)
        
              with open(f'./pickle_objects/preprocess/{database_name}/heterodata_pbg_{database_name}_{description}_train.pickle', 'wb') as f:
                  pickle.dump(heterodata_pbg_train, f, pickle.HIGHEST_PROTOCOL)  
              with open(f'./pickle_objects/preprocess/{database_name}/heterodata_pbg_{database_name}_{description}_test.pickle', 'wb') as f:
                  pickle.dump(heterodata_pbg_test, f, pickle.HIGHEST_PROTOCOL)

              logger.info(f'Executed PBG on {databse_name} ({description}). Results saved as pickle objects.')

    except Exception as e:
        logger.info(f'Error occurred: \n{e}')
        raise

if __name__ == '__main__':

    databse_list = ['20ng', '20ng', 'bbcnews', 'reuters', 'classic4', 'nsf', 'webkb', 'agnews']
    K = 50
    Kc = 400
    docf_values = [None]
    docst_values = [0.5]

    
    logger.info('Generating heterographs for benchmark datasets.')

    for database_name in databse_list:
        preprocess_datasets_pbg(database_name, K, Kc, docf_values, docst_values, preproc_pbg=True)



