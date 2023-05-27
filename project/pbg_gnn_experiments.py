from functions import *
import itertools
import os
logger = get_logger('log_experiments')
seed_everything(seed=42)

def run_experiment_gnn(database_name, 
                       hidden_channels, 
                       num_layers, 
                       p_dropout,
                       docf,
                       loss_function, 
                       version,
                       num_epochs):
    
    try:
        
        heterodata_description = f'heterodata_pbg_{database_name}_tpbg_docf_{str(docf)}'
        base_path = f'./pickle_objects/preprocess/tpbg/{database_name}'

        with open(f'{base_path}/{heterodata_description}_train.pickle', 'rb') as f:
            heterodata_train = pickle.load(f)
        with open(f'{base_path}/{heterodata_description}_test.pickle', 'rb') as f:
            heterodata_test = pickle.load(f)
                        
        logger.info(f'Loaded preprocessed heterographs {heterodata_description}.')

        training_description = f'{database_name}_tpbg_docf_{str(docf)}_nl_{num_layers}_pd_{p_dropout}_act_{loss_function}_ver_{version}'

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
                           verbose=False)
        
        loss_test, micro_test, acc_test, epoch_convergence = df_experiment
        df = pd.DataFrame(columns=['database_name', 'docf', 'hidden_channels', 'num_layers', 'p_dropout', 'loss_function', 'version', 'loss_test', 'micro_test', 'acc_test', 'epoch_convergence'])
        output_list = [database_name, docf, hidden_channels, num_layers, p_dropout, loss_function, version, loss_test, micro_test, acc_test, epoch_convergence]
        row = pd.Series(output_list, index=df.columns)
        df = df.append(row,ignore_index=True) 
        
        with open('./csv_objects/summary/experiments_summary.csv', 'a') as f:
            df.to_csv(f, mode='a', sep=';', decimal=',', index=False, header=f.tell()==0)
        
        logger.info(f'Executed experiments on {database_name} {training_description}. Results saved as pickle objects.')

    except Exception as e:
        logger.info(f'Error occurred: \n{e}')
        pass

if __name__ == '__main__':

    databse_list = ['20ng', 'bbc', 'classic4', 'nsf', 'cstr', 'dmoz_computers', 'dmoz_health', 'dmoz_science', 'dmoz_sports', 're8']
    K_list = [None]
    docf_list = [None, 'merge']
    hidden_channels_list = [20, 40, 100]
    num_layers_list = [2, 3, 4]
    p_dropout_list = [0.0, 0.2]
    loss_function_list = ['ce', 'fl']
    gnn_version_list = [1]
    num_epochs = [1500]

    iter_params = itertools.product(databse_list, 
                               K_list, 
                               docf_list, 
                               hidden_channels_list,
                               num_layers_list, 
                               p_dropout_list,
                               loss_function_list, 
                               gnn_version_list,
                               num_epochs)

    logger.info('Running experiments on datasets with heterographs and GNNs.')

    for params in iter_params:
        database_name = params[0]
        K = params[1]
        docf = str(params[2])
        hidden_channels = params[3]
        num_layers = params[4]
        p_dropout = params[5]
        loss_function = params[6]
        version = params[7]
        num_epochs = params[8]
                    
        logger.info(f'''Experiment setup:
            database_name={database_name}, 
            hidden_channels={hidden_channels}, 
            num_layers={num_layers}, 
            p_dropout={p_dropout}, 
            docf={docf}, 
            loss_function={loss_function}, 
            version={version},
            num_epochs={num_epochs}
            ''')

        run_experiment_gnn(database_name, 
                            hidden_channels, 
                            num_layers, 
                            p_dropout, 
                            docf, 
                            loss_function, 
                            version,
                            num_epochs)

