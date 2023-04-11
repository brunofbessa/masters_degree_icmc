from functions import *
import pickle
import glob

logger = get_logger('log_experiments')

seed_everything(seed=42)

def run_experiment_gnn(database_name, K, hidden_channels_list, num_layers_list, p_dropout_list, patience):
    try:

        logger.info(f'Running experiments on {database_name} K={K}')
        with open(f'./pickle_objects/preprocess/heterodata_pbg_{database_name}_k{K}_test.pickle', 'rb') as f:
            heterodata_pbg_test = pickle.load(f)
        with open(f'./pickle_objects/preprocess/heterodata_pbg_{database_name}_k{K}_train.pickle', 'rb') as f:
            heterodata_pbg_train = pickle.load(f)
        with open(f'./pickle_objects/preprocess/heterodata_pbg_{database_name}_k{K}_val.pickle', 'rb') as f:
            heterodata_pbg_val = pickle.load(f)

        df_experiment = experiment_gnn(database_name=f'{database_name} K={K}',
                                heterodata_pbg_train=heterodata_pbg_train,
                                heterodata_pbg_val=heterodata_pbg_val,
                                heterodata_pbg_test=heterodata_pbg_test,
                                hidden_channels_list=hidden_channels_list,
                                num_layers_list=num_layers_list,
                                p_dropout_list=p_dropout_list,
                                num_epochs=2*patience, 
                                patience=patience,
                                verbose=False)
        
        with open(f'./pickle_objects/experiments/df_experiment_{database_name}_k{K}.pickle', 'wb') as f:
            pickle.dump(df_experiment, f, pickle.HIGHEST_PROTOCOL)

        logger.info(f'Executed experiments on {database_name} K={K}. Results saved as pickle objects.')

    except Exception as e:
        logger.info(f'Error occurred: \n{e}')
        raise

if __name__ == '__main__':

    databse_list = ['reuters', 'bbcnews', 'classic4']
    K_values = [100 , 50, 10]
    hidden_channels_list = [100, 50, 10]
    num_layers_list = [2, 3, 4]
    p_dropout_list = [0.0]
    patience = 200

    logger.info('Running experiments on datasets with heterographs and GNNs.')

    for database_name in databse_list:
        for K in K_values:
            run_experiment_gnn(database_name, K, hidden_channels_list, num_layers_list, p_dropout_list, patience)


dataframes_path_list = []
for file in glob.glob('pickle_objects/experiments/*.pickle'):
    dataframes_path_list.append(file)

df_all_experiments = pd.DataFrame()

for df_path in dataframes_path_list:
    
    with open(df_path, 'rb') as f:
        df = pickle.load(f)
            
    if df_path == dataframes_path_list[0]:
        df_all_experiments = df
    else:
        df_all_experiments= df_all_experiments.append(df,ignore_index=True)
    
df_all_experiments['database_name_aux'] = df_all_experiments['database_name']
df_all_experiments['K'] = df_all_experiments['database_name_aux'].str.replace('.*K=', '').str.replace(' ', '')
df_all_experiments["database_name"] = df_all_experiments["database_name_aux"].str.replace("K=.*", "").str.replace(' ', '')

df_all_experiments = df_all_experiments.rename(columns={'database_name': 'database', 'K': 'K_z', 'hidden_channels': 'num. hidden channels'})

df_all_experiments = df_all_experiments.drop(["database_name_aux"], axis=1)

with open("./pickle_objects/experiments/df_all_experiments.pickle", "wb") as f:
    pickle.dump(df_all_experiments, f, pickle.HIGHEST_PROTOCOL)
