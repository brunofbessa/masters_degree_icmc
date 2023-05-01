from functions import *
logger = get_logger('log_experiments')
seed_everything(seed=42)

def run_experiment_gnn(database_name, 
                       K, 
                       hidden_channels, 
                       num_layers, 
                       p_dropout, 
                       Kc,
                       docf, 
                       docst,
                       loss_function, 
                       version,
                       num_epochs):
    
    try:
        
        heterodata_description = f'{database_name}_K_{K}_Kc_{Kc}_docf_{str(docf)}_docst_{str(docst)}'
        with open(f'./pickle_objects/preprocess/{database_name}/heterodata_pbg_{heterodata_description}_train.pickle', 'rb') as f:
            heterodata_train = pickle.load(f)
        with open(f'./pickle_objects/preprocess/{database_name}/heterodata_pbg_{heterodata_description}_test.pickle', 'rb') as f:
            heterodata_test = pickle.load(f)
                        
        logger.info(f'Loaded preprocessed heterographs {heterodata_description}.')

        training_description = f'{heterodata_description}_act_{loss_function}_ver_{version}'

        df_experiment = run_heterognn_splitted(database_name=database_name,
                           description=training_description, 
                           heterodata_train=heterodata_train,
                           heterodata_test=heterodata_test,
                           hidden_channels=hidden_channels,
                           num_layers=num_layers,
                           p_dropout=p_dropout,
                           num_epochs=num_epochs, 
                           aggr='sum',
                           version=version, 
                           loss_function=loss_function, 
                           verbose=True)
        
        loss_test, micro_test, acc_test, epoch_convergence = df_experiment
        df = pd.DataFrame(columns=['database_name', 'K', 'Kc', 'docf', 'docst', 'hidden_channels', 'num_layers', 'p_dropout', 'activation', 'version', 'loss_test', 'micro_test', 'acc_test', 'epoch_convergence'])
        output_list = [database_name, K, Kc, docf, docst, hidden_channels, num_layers, p_dropout, activation, version, loss_test, micro_test, acc_test, epoch_convergence]
        row = pd.Series(output_list, index=df.columns)
        df = df.append(row,ignore_index=True) 
        
#         with open('./csv_objects/summary/experiments_summary.csv', 'a') as f:
#             df.to_csv(f, mode='a', sep=';', decimal=',', index=False, header=f.tell()==0)
        
        logger.info(f'Executed experiments on {database_name} {training_description}. Results saved as pickle objects.')

    except Exception as e:
        logger.info(f'Error occurred: \n{e}')
        pass

if __name__ == '__main__':

    databse_list = ['classic4']#['20ng', '20ng', 'bbcnews', 'reuters', 'classic4', 'nsf', 'webkb', 'agnews']
    K = 50
    hidden_channels = 400
    num_layers = 3
    p_dropout = 0.2
    Kc = 400
    docf = 'replace'
    docst = 0.5
    num_epochs = 1500
    loss_function_list = ['fl']#['ce', 'fl']
    gnn_version_list = [1, 2, 3, 4]
    

    logger.info('Running experiments on datasets with heterographs and GNNs.')

    for database_name in databse_list:
        for loss_function in loss_function_list:
            for version in gnn_version_list:
                run_experiment_gnn(database_name, 
                                  K, 
                                  hidden_channels, 
                                  num_layers, 
                                  p_dropout, 
                                  Kc,
                                  docf, 
                                  docst,
                                  loss_function, 
                                  version,
                                  num_epochs)

