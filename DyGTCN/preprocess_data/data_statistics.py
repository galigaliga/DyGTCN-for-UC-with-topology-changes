import numpy as np
import pandas as pd
from tabulate import tabulate


def pprint_df(df, tablefmt='psql'):
    print(tabulate(df, headers='keys', tablefmt=tablefmt))


if __name__ == "__main__":
    dataset_name = 'ieee118'
    records = []

    edge_raw_features = np.load('../processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_features_df = pd.read_csv(f'./DG_data/{dataset_name}/all_busloads.csv')
    node_raw_features = node_features_df.values.astype(np.float32)
    node_raw_features = node_raw_features.reshape(118, -1)  # 重塑为(118, 24*365=8760)
    info = {'dataset_name': dataset_name,
            'num_nodes': node_raw_features.shape[0] - 1,
            'node_feat_dim': node_raw_features.shape[-1],
            'num_edges': edge_raw_features.shape[0] - 1,
            'edge_feat_dim': edge_raw_features.shape[-1]}
    records.append(info)

    info_df = pd.DataFrame.from_records(records)
    pprint_df(info_df)
