import argparse
import sys
import torch


def get_DyGTCN_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for training and evaluation with DyGTCN')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='ieee118',
                        choices=['ieee118'])
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGTCN', help='name of the model, note that EdgeBank is only applicable for evaluation',
                        choices=['DyGTCN'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=5e-5, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=64, help='dimension of the time embedding')
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=48, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=30, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    if args.load_best_configs:
        load_DyGTCN_best_configs(args=args)

    return args


def load_DyGTCN_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    if args.model_name == 'DyGTCN':
        args.num_layers = 3
        args.max_input_sequence_length = 24
        args.patch_size = 1
        args.dropout = 0.3
        assert args.max_input_sequence_length % args.patch_size == 0
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")


def get_other_args(is_evaluation: bool = False):
    """
    Get the args for unit status prediction task using various models (GCN, TCN, LSTM).

    :param is_evaluation: boolean, whether in evaluation mode
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser('Interface for training and evaluation with different models (GCN, TCN, LSTM)')

    # Dataset and model settings
    parser.add_argument('--dataset_name', type=str, default='ieee118',
                        help='Name of the dataset to use', choices=['ieee118'])
    parser.add_argument('--model_name', type=str, default='STGCN',
                        help='Name of the model to train or evaluate',
                        choices=['GCN', 'TCN', 'LSTM', 'STGCN'])
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use')

    # Model architecture hyperparameters
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers in the model (used by GCN and LSTM)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of hidden layer features (used by TCN and LSTM)')
    parser.add_argument('--gcn_hidden_dim', type=int, default=64,
                        help='Hidden dimension for GCN layers')
    parser.add_argument('--channel_embedding_dim', type=int, default=50,
                        help='Dimension of channel embeddings (used by TCN)')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='Number of attention heads (if applicable)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate applied in model layers')

    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.0015,
                        help='Learning rate for optimizer')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer name', choices=['SGD', 'Adam', 'RMSprop'])
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Number of epochs to train')

    # Data splitting settings
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of test set')

    # Multiple runs and reproducibility
    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of independent runs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Evaluation and testing frequency
    parser.add_argument('--test_interval_epochs', type=int, default=5,
                        help='How often to run testing during training (in epochs)')

    # Best configuration loading
    parser.add_argument('--load_best_configs', action='store_true', default=False,
                        help='Whether to load best configurations for the selected model')

    try:
        args = parser.parse_args()
        args.device = 'cpu'
        #args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    if args.load_best_configs:
        load_model_best_configs(args=args)

    return args


def load_model_best_configs(args: argparse.Namespace):
    """
    Load the best hyperparameter configurations for different models.

    :param args: argparse.Namespace
    :return:
    """
    if args.model_name == 'GCN':
        # GCN best configuration
        args.gcn_hidden_dim = 64
        args.num_layers = 4
        args.dropout = 0.3
        args.learning_rate = 0.001
        args.weight_decay = 1e-5

    elif args.model_name == 'TCN':
        # TCN best configuration
        args.hidden_dim = 128
        args.channel_embedding_dim = 64
        args.num_layers = 3
        args.dropout = 0.3
        args.learning_rate = 0.001
        args.weight_decay = 1e-4

    elif args.model_name == 'LSTM':
        # LSTM best configuration
        args.hidden_dim = 128
        args.num_layers = 2
        args.dropout = 0.3
        args.learning_rate = 0.0003
        args.weight_decay = 1e-4

    elif args.model_name == 'STGCN':
        # STGCN best configuration
        args.gcn_hidden_dim = 64
        args.num_layers = 3
        args.dropout = 0.3
        args.learning_rate = 0.0015
        args.weight_decay = 1e-4

    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")