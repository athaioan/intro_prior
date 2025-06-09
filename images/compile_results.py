import os
import numpy as np
import pandas as pd
import scipy.stats
from PIL import Image
import ast
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_ablation_log(ablation_log, dataset, prior, num_components=100, intro_prior=True, learnable_component=True, 
                      metric='assignment_neg_entropy', entropy_reg=[0.0,1.0,10.0,100.0], ax=None, xi=None, plot_dir='./figures'):


    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
        with_ax = True
    else:
        with_ax = False

    for r_entropy in entropy_reg:
        experiment_name = f"{prior}({num_components})-{r_entropy}_LC:{learnable_component}_IP:{intro_prior}"
        
        history_metric = []

        for log_path in ablation_log[experiment_name]:

            with open(log_path) as f:
                lines = f.readlines()

            for line in lines:
                if line.split(":")[0] == metric:
                    if metric == 'assignment_neg_entropy':
                        fig_title = 'Responsibilities Normalized Entropy'
                        log_history = -np.float64(ast.literal_eval(line.split(":")[1][1:-1]))
                        log_history = np.convolve(log_history, np.ones(5_000)/5_000, mode='valid')
                    elif metric == 'fid_inception_train':
                        ## removing fid computed before starting training
                        log_history = np.float64(ast.literal_eval(line.split(":")[1][1:-1]))[1:]
                        fig_title = 'FID(GEN)'

                    else:
                        log_history = np.float64(ast.literal_eval(line.split(":")[1][1:-1]))
                        fig_title = metric
                    history_metric.append(log_history)

        mean, std = np.mean(history_metric, axis=0), np.std(history_metric, axis=0)

        if xi is None:
            xi = range(len(mean))
                           
        if with_ax:
            ax.plot(xi, mean, linestyle='dotted')
        else:
            ax.plot(xi, mean, label=r"$r_{\mathregular{entropy}}=" + f"{r_entropy}$", linestyle='solid')

        # ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)


    # Get the existing handles and labels from the current legend
    handles, labels = ax.get_legend_handles_labels()

    # Create custom legend handles
    custom_lines = [
        Line2D([0], [0], color='black', lw=2, linestyle='-'),   # solid line
        Line2D([0], [0], color='black', lw=2, linestyle='--')   # dotted line
    ]
    custom_labels = ['w/ LC', 'w/o LC']

    # Combine existing and custom handles and labels
    combined_handles = handles + custom_lines
    combined_labels = labels + custom_labels

    # Set the combined legend
    ax.legend(combined_handles, combined_labels)

    ax.set_xlabel('Iterations')
    ax.set_ylabel(fig_title)
    ax.set_aspect('auto')
    plt.rcParams.update({'figure.autolayout': True})
    plt.savefig(os.path.join(plot_dir,f"{dataset}_{num_components}_{intro_prior}_{metric}.png"), dpi=100) 

    return ax

def merge_summary_images(summary_img_paths, output_path, margin_offset=5, orientation='horizontal'):

    # taken from https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python 
    images = [Image.open(fig_name) for fig_name in summary_img_paths]
    widths, heights = zip(*(i.size for i in images))

    if orientation == 'vertical':
        max_width = max(widths)
        total_height = sum(heights) + margin_offset * (len(images) - 1)

        new_im = Image.new('RGB', (max_width, total_height), (255, 255, 255))

        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1] + margin_offset

    elif orientation == 'horizontal':
        total_width = sum(widths) + margin_offset * (len(images) - 1)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))

        y_offset = 0
        for im in images:
            new_im.paste(im, (y_offset, 0))
            y_offset += im.size[0] + margin_offset

    new_im.save(output_path)
   

def bold_best_config(result_dict, metrics_descending, metrics_ascending):
    """ Highlighting with bold the
        best performing value. """

    for current_metric in metrics_descending:
        best_value = np.inf

        for config in result_dict.keys():
           
           current_value = float(result_dict[config][current_metric].split(" ")[0])

           if current_value <= best_value:
               best_value = current_value


        ## highlight best value
        for config in result_dict.keys():
            current_value = float(result_dict[config][current_metric].split(" ")[0])           
            if current_value == best_value:
                result_dict[config][current_metric] = f'\textbf{{{result_dict[config][current_metric]}}}'
        

    for current_metric in metrics_ascending:
        best_value = - np.inf

        for config in result_dict.keys():
           
           current_value = float(result_dict[config][current_metric].split(" ")[0])

           if current_value >= best_value:
               best_value = current_value

        ## highlight best value
        for config in result_dict.keys():
            current_value = float(result_dict[config][current_metric].split(" ")[0])           
            if current_value == best_value:
                result_dict[config][current_metric] = f'\textbf{{{result_dict[config][current_metric]}}}'       
              

    return result_dict

def dict_mean(dict_list, confidence=0.95, interval_se=False, se=False):
    """ Computing the mean and confidence
        of multiple runs. """

    threshold = 1e-3  # Define the threshold for small values


    mean_dict = {}
    std_dict = {}
    se_dict = {}

    result_dict ={}
    for key in dict_list[0].keys():
        values = [d[key] for d in dict_list]
        n = len(values)

        mean_dict[key], std_dict[key], se = np.mean(values), np.std(values), scipy.stats.sem(values)
        se_dict[key] = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        
        if interval_se:
            ## standard error interval
            # if abs(mean_dict[key]) < threshold or abs(se_dict[key]) < threshold:
            #     result_dict[key] = '{:.3e} \u00B1 {:.3e}'.format(mean_dict[key], se_dict[key])
            # else:
                result_dict[key] = '{:.3f} \u00B1 {:.3f}'.format(mean_dict[key], se_dict[key])

        elif se:
            ## standard deviation interval
            if abs(mean_dict[key]) < threshold and abs(mean_dict[key])>0:
                result_dict[key] = '{:.3e} \u00B1 {:.3e}'.format(mean_dict[key], std_dict[key]/np.sqrt(n))
            else:
                result_dict[key] = '{:.3f} \u00B1 {:.3f}'.format(mean_dict[key], std_dict[key]/np.sqrt(n))
        else:
            ## standard deviation interval
            if abs(mean_dict[key]) < threshold and abs(mean_dict[key])>0:
                result_dict[key] = '{:.3e} \u00B1 {:.3e}'.format(mean_dict[key], std_dict[key])
            else:
                result_dict[key] = '{:.3f} \u00B1 {:.3f}'.format(mean_dict[key], std_dict[key])

    return result_dict

def get_stats(exp_dir, curr_id, enc_entropy_reg, best_model_mode='train', summary_seed=0):
    

    result_dicts = []
    experiments =  [os.path.join(exp_dir, experiment) for experiment in os.listdir(exp_dir) if curr_id in experiment and float(experiment.split("assingmentEncEntropyReg:")[1].split("_")[0]) == enc_entropy_reg ]
    
    summary_img_path = [os.path.join(exp_dir, experiment, 'summary_final.png')  for experiment in experiments if int(experiment.split("seed:")[-1].split("_")[0]) == summary_seed][0]
    # summary_img_path = [os.path.join(exp_dir, experiment, 'final_latent.png')  for experiment in experiments if int(experiment.split("seed:")[-1].split("_")[0]) == summary_seed][0]
    # summary_img_path = [os.path.join(exp_dir, experiment, 'final_manifold_real.png')  for experiment in experiments if int(experiment.split("seed:")[-1].split("_")[0]) == summary_seed][0]

    log_path = [os.path.join(exp_dir, experiment, 'history.txt')  for experiment in experiments]
   
    print('Experiment:', os.path.join(curr_id), ' - Number of runs', len(experiments), '\n')

    for exp in experiments:

        result_txt = os.path.join(exp, 'results_fid_{}.txt'.format(best_model_mode))

        with open(result_txt) as f:
            result = [line.rstrip('\n') for line in f]

        result_dict = {}
        for line in result:   
            key = line.split(":")[0]
            if key !='pretrained_pth':

                if key.split("_")[0] == 'fid' or key.split("_")[0] == 'precision' or key.split("_")[0] == 'recall':
                    if key.split("_")[-1] != best_model_mode:
                        continue

                value = np.float64(line.split(":")[1])
                result_dict[key] = value

        result_dicts.append(result_dict)
   
    result_dict = dict_mean(result_dicts, se=True)

    keys_to_include = ['entropy_soft_assignment', # 'drift_measure', 'ELBO_test', 'RE_mse_test', 
                       'fid_inception_train', 'fid_inception_rec_train', 'recall_train', 'precision_train',
                        'test_acc_svm_few', 'test_acc_svm_many', 'test_acc_knn_few', 'test_acc_knn_many']  # Replace with the actual keys you want to include
    result_dict = {key: result_dict[key] for key in keys_to_include if key in result_dict}

    # Define the mapping of old keys to new keys
    key_mapping = {
        'entropy_soft_assignment': 'Assignment Entropy',
        'drift_measure': 'Drift',
        'ELBO_test': 'ELBO',
        'RE_mse_test': 'RE MSE',
        'fid_inception_train': 'FID (gen)',
        'fid_inception_rec_train': 'FID (rec)',
        'precision_train': 'Precision',
        'recall_train': 'Recall',
        'test_acc_svm_few': 'SVM (few)',
        'test_acc_svm_many': 'SVM (many)',
        'test_acc_knn_few': 'KNN (few)',
        'test_acc_knn_many': 'KNN (many)',
    }

    # Rename the keys in the result_dict based on the mapping
    result_dict = {key_mapping[key]: value for key, value in result_dict.items()}


    return result_dict, summary_img_path, log_path

RESULT_DIR = 'path/to/results/results_intro_prior_images_low'
EXPERIMENT_DIR = 'path/to/experiment/dir'

EXPERIMENT_DIR = os.path.join(RESULT_DIR, EXPERIMENT_DIR)

## get datasets
DATASETS = [folder for folder in os.listdir(EXPERIMENT_DIR) if folder != 'wandb']
DATASETS = ['mnist', 'fmnist', 'cifar10']
ARCH = 'sIntroVAE'


for dataset in DATASETS:
    result_dict = {}

    RESULT_DIR = os.path.join(EXPERIMENT_DIR, dataset, ARCH)

    ablation_summary = []
    ablation_log = {}

    curr_result_dir = np.sort(np.asarray(os.listdir(RESULT_DIR))).tolist()
    id = np.asarray([np.int(i.split('vamp_numComponents:')[-1].split("_")[0]) for i in curr_result_dir[1:]]).argsort()

    curr_result_dir = [curr_result_dir[0]] + np.array(curr_result_dir[1:])[id].tolist()

   
    
    for index, folder in enumerate(curr_result_dir):
        curr_dir = os.path.join(RESULT_DIR, folder)
        
        prior = folder.split("prior:")[1].split('_')[0]
        num_component = folder.split("numComponents:")[1].split('_')[0]
        intro_prior = folder.split("introPrior:")[1].split('_')[0]
        learnable_component = folder.split("learnableComponent:")[1].split('_')[0]
        sampling_with_grad = folder.split("sampleGrad:")[1].split('_')[0]

        unique_ids = np.unique(np.array([exp_dir.split("_seed")[0] for exp_dir in os.listdir(curr_dir)])).tolist()


        best_fid = np.inf        

        for curr_id in unique_ids:

            enc_entropy_reg = float(curr_id.split("assingmentEncEntropyReg:")[1].split('_')[0])

            stats_dict, summary_img_path, log_path = get_stats(curr_dir, curr_id, enc_entropy_reg, summary_seed=1)

            fid = float(stats_dict['FID (gen)'].split(" ")[0])

            if num_component != 'None':
                experiment_name = f"{index}.{prior}({num_component})-{enc_entropy_reg}"
            else:
                experiment_name = f"{index}.{prior}-{enc_entropy_reg}"
            
            config_dict = {'LC': '\checkmark' if learnable_component == 'True' else ' ', 'IP': '\checkmark' if intro_prior == 'True' else ' '}
            config_dict.update(stats_dict)

            
            # result_dict[experiment_name] = config_dict
            # ablation_summary.append(summary_img_path)   

            # ablation_log[experiment_name[2:]+"_LC:{}_IP:{}".format(learnable_component, intro_prior)]=log_path   

            if fid < best_fid:
                best_fid = fid
                best_experiment = experiment_name
                best_config = config_dict
                best_summary_img_path = summary_img_path

        ablation_summary.append(best_summary_img_path)   
        result_dict[best_experiment] = best_config

    merge_summary_images(ablation_summary, '{}_ablation_summary.png'.format(dataset), margin_offset=5)

    result_dict = bold_best_config(result_dict, 
                                   metrics_descending = [#'Drift', 'ELBO', 'RE MSE', 
                                                         'FID (gen)', 'FID (rec)'],
                                   metrics_ascending = ['Recall', 'Precision', 
                                                        'SVM (few)', 'SVM (many)',
                                                        'KNN (few)', 'KNN (many)'])
        

    df = pd.DataFrame.from_dict(result_dict)
    df = df.transpose()

    print(df.to_latex(bold_rows=True, escape=False, column_format='c'*(len(config_dict)+1)))
    print("\n")


