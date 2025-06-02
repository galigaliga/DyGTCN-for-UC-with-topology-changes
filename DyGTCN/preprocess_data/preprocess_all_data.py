import os

dataset_name = 'ieee118'
os.system(f'python preprocess_data.py --dataset_name {dataset_name}')
