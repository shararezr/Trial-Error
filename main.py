

import argparse
import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
import pickle
import matplotlib.pyplot as plt
from tools import Data_Train, Data_Val, Data_Test, Data_CHLS
from model import create_model_diffu, Att_Diffuse_model
from training import model_train, LSHT_inference
from collections import Counter
plt.style.use("seaborn-dark-palette")


# Define default parameters in a dictionary
default_params = {
    'dataset': 'ml-1m',
    'log_file': 'log/',
    'random_seed': 1997,
    'max_len': 50,
    'device': 'cuda',
    'num_gpu': 4,
    'batch_size': 512,
    'hidden_size': 128,
    'dropout': 0.1,
    'emb_dropout': 0.3,
    'hidden_act': 'gelu',
    'num_blocks': 4,
    'epochs': 24,
    'decay_step': 100,
    'gamma': 0.1,
    'metric_ks': [5, 10, 20],
    'optimizer': 'Adam',
    'lr': 0.001,
    'loss_lambda': 0.001,
    'weight_decay': 0,
    'momentum': None,
    'schedule_sampler_name': 'lossaware',
    'diffusion_steps': 32,
    'lambda_uncertainty': 0.001,
    'noise_schedule': 'trunc_lin',
    'rescale_timesteps': True,
    'eval_interval': 4,
    'patience': 5,
    'description': 'Diffu_norm_score',
    'long_head': False,
    'diversity_measure': False,
    'epoch_time_avg': False
}

# Initialize the argument parser
parser = argparse.ArgumentParser()

# Add arguments from the default parameters
for arg, default_value in default_params.items():
    parser.add_argument(f'--{arg}', default=default_value)

# Parse the arguments
args = parser.parse_args([])


if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)
if not os.path.exists(args.log_file + args.dataset):
    os.makedirs(args.log_file + args.dataset )


logging.basicConfig(level=logging.INFO, filename=args.log_file + args.dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)
logger.info(args)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def item_num_create(args, item_num):
    args.item_num = item_num
    return args


def cold_hot_long_short(data_raw, dataset_name):
    item_list = []
    len_list = []
    target_item = []

    for id_temp in data_raw['train']:
        temp_list = data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp]
        len_list.append(len(temp_list))
        target_item.append(data_raw['test'][id_temp][0])
        item_list += temp_list
    item_num_count = Counter(item_list)
    split_num = np.percentile(list(item_num_count.values()), 80)
    cold_item, hot_item = [], []
    for item_num_temp in item_num_count.items():
        if item_num_temp[1] < split_num:
            cold_item.append(item_num_temp[0])
        else:
            hot_item.append(item_num_temp[0])
    cold_ids, hot_ids = [], []
    cold_list, hot_list = [], []
    for id_temp, item_temp in enumerate(data_raw['test'].values()):
        if item_temp[0] in hot_item:
            hot_ids.append(id_temp)
            if dataset_name == 'ml-1m':
                hot_list.append(data_raw['train'][id_temp+1] + data_raw['val'][id_temp+1] + data_raw['test'][id_temp+1])
            else:
                hot_list.append(data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp])
        else:
            cold_ids.append(id_temp)
            if dataset_name == 'ml-1m':
                cold_list.append(data_raw['train'][id_temp+1] + data_raw['val'][id_temp+1] + data_raw['test'][id_temp+1])
            else:
                cold_list.append(data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp])
    cold_hot_dict = {'hot': hot_list, 'cold': cold_list}

    len_short = np.percentile(len_list, 20)
    len_midshort = np.percentile(len_list, 40)
    len_midlong = np.percentile(len_list, 60)
    len_long = np.percentile(len_list, 80)

    len_seq_dict = {'short': [], 'mid_short': [], 'mid': [], 'mid_long': [], 'long': []}
    for id_temp, len_temp in enumerate(len_list):
        if dataset_name == 'ml-1m':
            temp_seq = data_raw['train'][id_temp+1] + data_raw['val'][id_temp+1] + data_raw['test'][id_temp+1]
        else:
            temp_seq = data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp]
        if len_temp <= len_short:
            len_seq_dict['short'].append(temp_seq)
        elif len_short < len_temp <= len_midshort:
            len_seq_dict['mid_short'].append(temp_seq)
        elif len_midshort < len_temp <= len_midlong:
            len_seq_dict['mid'].append(temp_seq)
        elif len_midlong < len_temp <= len_long:
            len_seq_dict['mid_long'].append(temp_seq)
        else:
            len_seq_dict['long'].append(temp_seq)
    return cold_hot_dict, len_seq_dict, split_num, [len_short, len_midshort, len_midlong, len_long], len_list, list(item_num_count.values())




def plot_training_progress(train_losses, val_metrics):
    plt.figure(figsize=(12, 6))
    # Define marker styles for each metric
    marker_styles = ['o', 's', '^', 'D', 'x', 'v']

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot validation metrics
    plt.subplot(1, 2, 2)
    metrics = list(val_metrics.keys())
    scores = list(val_metrics.values())
    for i in range(0,len(metrics)):
        plt.plot(range(0, len(scores[i])),scores[i], marker=marker_styles[i], label=metrics[i])
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics')
    plt.savefig('plot.png')  # Save the plot as an image file
    plt.close()  # Close the plot to release memory

    # Adjust layout and display plot
    plt.tight_layout()
    plt.savefig('plot.png')  # Save the plot as an image file
    plt.close()  # Close the plot to release memory


def plot_learning_rate(lr_scheduler):
    # Plotting the learning rate schedule
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(lr_scheduler) + 1), lr_scheduler, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig('plot.png')  # Save the plot as an image file
    plt.close()  # Close the plot to release memory

from sklearn.manifold import TSNE
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def plot_density_pred(all_predictions, target_y, n_clusters):
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(all_predictions)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_predictions)
    
    # Plot the result
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='tab10', s=1)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.title('t-SNE of Recommended Items with Clustering')
    plt.savefig('plot.png')  # Save the plot as an image file
    plt.close()  # Close the plot to release memory


def test_result(test_results):
    # Extracting the keys (metrics) and values (scores) from the dictionary
    metrics = list(test_results.keys())
    scores = list(test_results.values())

    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, scores, color='skyblue')

    # Adding title and labels
    plt.title('Test Results')
    plt.xlabel('Metrics')
    plt.ylabel('Scores')

    # Rotating x-axis labels for better readability
    plt.xticks(rotation=45)

    # Displaying the plot
    plt.tight_layout()
    plt.savefig('plot.png')  # Save the plot as an image file
    plt.close()  # Close the plot to release memory


def main(args):
    fix_random_seed_as(args.random_seed)
    path_data = 'dataset.pkl'
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)

    # cold_hot_long_short(data_raw, args.dataset)

    args = item_num_create(args, len(data_raw['smap']))
    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()

    item_list = []
    len_list = []
    target_item = []


    for id_temp in data_raw['train']:
      #temp_list = data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp]
      #len_list.append(len(temp_list))
      target_item.append(data_raw['test'][id_temp][0])
      #item_list += temp_list
    num_unique_items = len(set(target_item))



    diffu_rec = create_model_diffu(args)
    rec_diffu_joint_model = Att_Diffuse_model(diffu_rec, args)


    best_model, test_results, val_metrics_dict_mean, train_losses, target_pre, label_pre, learning_rates = model_train(tra_data_loader, val_data_loader, test_data_loader, rec_diffu_joint_model, args, logger)

    plot_training_progress(train_losses, val_metrics_dict_mean)
    plot_learning_rate(learning_rates)
    test_result(test_results)
    plot_density_pred(scores_rec_diffu)
    num_cluster = 6
    plot_density_pred(target_pre, label_pre,num_cluster)

    # Save the best model
    torch.save(best_model.state_dict(), 'best_model.pth')
    
    # Save the test results, validation metrics, and training losses
    with open('test_results.pkl', 'wb') as f:
        pickle.dump(test_results, f)
    
    with open('val_metrics.pkl', 'wb') as f:
        pickle.dump(val_metrics_dict_mean, f)
    
    with open('train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    
    # Save the target predictions and labels
    np.save('target_predictions.npy', target_pre)
    np.save('label_predictions.npy', label_pre)
    
    # Save the learning rates
    np.save('learning_rates.npy', learning_rates)


    if args.long_head:
        cold_hot_dict, len_seq_dict, split_hotcold, split_length, list_len, list_num = cold_hot_long_short(data_raw, args.dataset)
        cold_data = Data_CHLS(cold_hot_dict['cold'], args)
        cold_data_loader = cold_data.get_pytorch_dataloaders()
        print('--------------Cold item-----------------------')
        LSHT_inference(best_model, args, cold_data_loader)

        hot_data = Data_CHLS(cold_hot_dict['hot'], args)
        hot_data_loader = hot_data.get_pytorch_dataloaders()
        print('--------------hot item-----------------------')
        LSHT_inference(best_model, args, hot_data_loader)

        short_data = Data_CHLS(len_seq_dict['short'], args)
        short_data_loader = short_data.get_pytorch_dataloaders()
        print('--------------Short-----------------------')
        LSHT_inference(best_model, args, short_data_loader)

        mid_short_data = Data_CHLS(len_seq_dict['mid_short'], args)
        mid_short_data_loader = mid_short_data.get_pytorch_dataloaders()
        print('--------------Mid_short-----------------------')
        LSHT_inference(best_model, args, mid_short_data_loader)

        mid_data = Data_CHLS(len_seq_dict['mid'], args)
        mid_data_loader = mid_data.get_pytorch_dataloaders()
        print('--------------Mid-----------------------')
        LSHT_inference(best_model, args, mid_data_loader)

        mid_long_data = Data_CHLS(len_seq_dict['mid_long'], args)
        mid_long_data_loader = mid_long_data.get_pytorch_dataloaders()
        print('--------------Mid_long-----------------------')
        LSHT_inference(best_model, args, mid_long_data_loader)

        long_data = Data_CHLS(len_seq_dict['long'], args)
        long_data_loader = long_data.get_pytorch_dataloaders()
        print('--------------Long-----------------------')
        LSHT_inference(best_model, args, long_data_loader)

    return best_model, test_results, val_metrics_dict_mean, train_losses, target_pre, label_pre, learning_rates

if __name__ == '__main__':
    main(args)

