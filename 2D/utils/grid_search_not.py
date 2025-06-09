import os
import numpy as np
import scipy.stats
import json 

def extract_interval(data, confidence=0.95):

    n = len(data)
    
    average, se = np.mean(data), scipy.stats.sem(data)
    interval = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    
    return average, interval

def extract_stats(results_folder):


    for result_dir in results_folder:

        session_dirs = os.listdir(result_dir)
        session_dirs = [os.path.join(result_dir, session_dir) for session_dir in session_dirs]

        gnelbo, kl, jsb = [], [], []
        for session_dir in session_dirs:


                if os.path.exists(os.path.join(session_dir,'results_log_soft_intro_vae.txt')):


                        with open(os.path.join(session_dir,'results_log_soft_intro_vae.txt')) as f:
                                results = f.readlines()

                        gnelbo.append(np.float64(results[-1].split("_")[11]))
                        kl.append(np.float64(results[-1].split("_")[13]))
                        jsb.append(np.float64(results[-1].split("_")[15]))

        if len(gnelbo):
            average_gneblo, interval_gnelbo = extract_interval(gnelbo)
            average_kl, interval_kl = extract_interval(kl)
            average_jsb, interval_jsb = extract_interval(jsb)

            with open(os.path.join(result_dir, 'stats.txt'), 'a') as fp:
                line = f'ELBO: mean={average_gneblo} || interval={interval_gnelbo}\n'  
                fp.write(line)
    
                line = f'KL: mean={average_kl} || interval={interval_kl}\n'    
                fp.write(line)

                line = f'JSD: mean={average_jsb} || interval={interval_jsb}\n'    
                fp.write(line)

def extract_mog_graph(results_folder, dataset):

    prior =  'MoG'
    current_path = os.path.join(results_folder,prior)
    
    KL_means, KL_intervals = [], []
    JSD_means, JSD_intervals = [], []

    for intro_prior in ['intro_prior_False', 'intro_prior_True']:
        KL_mean, KL_interval = [], []
        JSD_mean, JSD_interval = [], []   
        for num_c in ['num_c_1','num_c_8','num_c_16','num_c_32','num_c_64','num_c_128','num_c_256']:
            current_path = os.path.join(results_folder,prior,intro_prior,num_c,'sIntroVAE','stats.txt')
            
            # Open the file in read mode
            with open(current_path, "r") as file:
                # Read the contents of the file
                contents = file.read()

            split_string = contents.split("\n")[1].split('||')
            mean = float(split_string[0].split('=')[1])
            interval = float(split_string[1].split('=')[1])
            KL_mean.append(mean) 
            KL_interval.append(interval) 

            split_string = contents.split("\n")[2].split('||')
            mean = float(split_string[0].split('=')[1])
            interval = float(split_string[1].split('=')[1])
            JSD_mean.append(mean) 
            JSD_interval.append(interval) 
    
        KL_means.append(KL_mean) 
        KL_intervals.append(KL_interval) 

        JSD_means.append(JSD_mean) 
        JSD_intervals.append(JSD_interval) 
    

    prior =  'imposed'
    current_path = os.path.join(results_folder,prior,'intro_prior_False','num_c_1','sIntroVAE','stats.txt')

    # Open the file in read mode
    with open(current_path, "r") as file:
        # Read the contents of the file
        contents = file.read()

    # Split the string by '||' to separate the mean and interval values
    split_string = contents.split("\n")[1].split('||')
    # Extract the mean and interval values from the split string
    mean = float(split_string[0].split('=')[1])
    interval = float(split_string[1].split('=')[1])
    KL_imposed_mean = 7*[mean] 
    KL_imposed_interval = 7*[interval] 

    split_string = contents.split("\n")[2].split('||')
    mean = float(split_string[0].split('=')[1])
    interval = float(split_string[1].split('=')[1])
    JSD_imposed_mean = 7*[mean] 
    JSD_imposed_interval = 7*[interval] 

    KL_means.append(KL_imposed_mean) 
    KL_intervals.append(KL_imposed_interval) 

    JSD_means.append(JSD_imposed_mean) 
    JSD_intervals.append(JSD_imposed_interval) 



    import matplotlib.pyplot as plt
    import numpy as np

    plt.close('all')

    # Generate x-axis values
    x = [1,8,16,32,64,128,256]

    labels = ['Fixed-Prior', 'Free-Prior', 'Imposed-Prior']

    # Plotting KLs
    for i, mean_list in enumerate(KL_means):
        plt.plot(x, mean_list, label=f'{labels[i]} (KL)')

    # Plot interval confidences with legends
    for i, interval_list in enumerate(KL_intervals):
        plt.fill_between(x, np.array(KL_means[i])-np.array(interval_list), np.array(KL_means[i])+np.array(interval_list),
                        alpha=0.1)


    # Plotting JSD
    for i, mean_list in enumerate(JSD_means):
        plt.plot(x, mean_list, '--', label=f'{labels[i]} (JSD)')

    # Plot interval confidences with legends
    for i, interval_list in enumerate(JSD_intervals):
        plt.fill_between(x, np.array(JSD_means[i])-np.array(interval_list), np.array(JSD_means[i])+np.array(interval_list),
                        alpha=0.1)


    # Add labels and legends
    plt.xlabel('num of mixture components')
    plt.ylabel('Divergence')
    plt.title(dataset)
    plt.legend()

    # Show the plot
    plt.savefig(dataset + '_mog_ablation.png')



    return

def get_optimal_config(dataset, model, alpha, beta_rec, beta_kl, beta_neg, prior_mode, learnable_prob, intro_prior, num_C, optimal_configurations={}):

    
    prior_args = {'mode': prior_mode, 'num_C': num_C, 'learnable_prob': learnable_prob, 'intro_prior': intro_prior, 'init_mode': 'from_data'}


    optimal_configurations[len(optimal_configurations)]={
            'dataset': dataset,
            'num_vae_iter': 30_000 if model == 'VAE' else 2_000,
            'beta_rec': beta_rec,
            'beta_kl': beta_kl,
            'beta_neg': beta_neg,
            'alpha': alpha,
            'prior': prior_args,
            }

    return optimal_configurations


def extract_results_grid_search(result_path, json_path, num_C):
    """ Summarizing results of grid search and storing the optimal hyperparameters """

    optimal_configurations = {}

    for dataset in [ '8Gaussian', 'checkerboard', 'spiral', 'rings']:

        print(10*"#####")

        for model in ['VAE','sIntroVAE']:

            dataset_results = os.path.join(result_path, dataset, model)

            for prior in ['imposed', 'MoG', 'vamp']:

                if prior == 'imposed':

                    gnelbo = []
                    kl = []
                    jsb = []
                    succesful_configs = []

                    data_folds = os.path.join(dataset_results, prior)
                    
                    configurations = os.listdir(data_folds)

                    ## get_best_config
                    for current_configuration_ in configurations:
                        current_configuration = os.path.join(data_folds, current_configuration_)
                        current_seed = os.listdir(current_configuration)[0]
                        current_seed = os.path.join(current_configuration, current_seed)

                        if os.path.exists(os.path.join(current_seed,'results_log_soft_intro_vae.txt')):


                            with open(os.path.join(current_seed,'results_log_soft_intro_vae.txt')) as f:
                                    results = f.readlines()

                            gnelbo.append(np.float64(results[-1].split("_")[11]))
                            kl.append(np.float64(results[-1].split("_")[13]))
                            jsb.append(np.float64(results[-1].split("_")[15]))
                            succesful_configs.append(current_configuration_)

                    # Convert the lists to NumPy arrays for easier manipulation
                    gnelbo_arr = np.array(gnelbo)
                    kl_arr = np.array(kl)
                    jsb_arr = np.array(jsb)

                    # Find the index with the minimum value for jsb
                    min_idx = np.argmin(kl_arr)

                    # Extract the corresponding values for gnelbo and jsb
                    min_gnelbo = gnelbo_arr[min_idx]
                    min_kl = kl_arr[min_idx]
                    min_jsb = jsb_arr[min_idx]
                    
                    # extract hyperparams
                    alpha, beta_rec, beta_kl, beta_neg = [float(i) for i in  succesful_configs[min_idx].split(" ")[1::2]] 
                    optimal_configurations = get_optimal_config(dataset, model, alpha, beta_rec, beta_kl, beta_neg, 
                                                            prior_mode=prior, learnable_prob=False, intro_prior=False, 
                                                            num_C=num_C, optimal_configurations=optimal_configurations)

                    print("\n model: {} Dataset: {} prior: {}".format(model, dataset, prior))
                    print("\n Corresponding gnelbo value:", min_gnelbo)
                    print("\n Minimum kl value:", min_kl)
                    print("\n Corresponding jsd value:", min_jsb)
                    print("\n optimal config:", succesful_configs[min_idx])
                    print("\n len config:", len(configurations))

                    print(10*"#####")
                
                else:
                    
                    if model == 'VAE':

                        for prob in ['learnable_prob_False', 'learnable_prob_True']:
                            
                            gnelbo = []
                            kl = []
                            jsb = []
                            succesful_configs = []


                            data_folds = os.path.join(dataset_results, prior, "num_C_{}".format(num_C), prob)
                        
                            configurations = os.listdir(data_folds)

                            for current_configuration_ in configurations:
                                current_configuration = os.path.join(data_folds, current_configuration_)
                                current_seed = os.listdir(current_configuration)[0]
                                current_seed = os.path.join(current_configuration, current_seed)

                                if os.path.exists(os.path.join(current_seed,'results_log_soft_intro_vae.txt')):


                                    with open(os.path.join(current_seed,'results_log_soft_intro_vae.txt')) as f:
                                            results = f.readlines()

                                    gnelbo.append(np.float64(results[-1].split("_")[11]))
                                    kl.append(np.float64(results[-1].split("_")[13]))
                                    jsb.append(np.float64(results[-1].split("_")[15]))
                                    succesful_configs.append(current_configuration_)


                            # Convert the lists to NumPy arrays for easier manipulation
                            gnelbo_arr = np.array(gnelbo)
                            kl_arr = np.array(kl)
                            jsb_arr = np.array(jsb)

                            # Find the index with the minimum value for jsb
                            min_idx = np.argmin(kl_arr)

                            # Extract the corresponding values for gnelbo and jsb
                            min_gnelbo = gnelbo_arr[min_idx]
                            min_kl = kl_arr[min_idx]
                            min_jsb = jsb_arr[min_idx]

                            # extract hyperparams
                            alpha, beta_rec, beta_kl, beta_neg = [float(i) for i in  succesful_configs[min_idx].split(" ")[1::2]] 
                            optimal_configurations = get_optimal_config(dataset, model, alpha, beta_rec, beta_kl, beta_neg, 
                                                                    prior_mode=prior, learnable_prob=prob.split("_")[-1] == "True", intro_prior=False, num_C=num_C, optimal_configurations=optimal_configurations)

                            # Print the results                    
                            print("\n model: {} Dataset: {} prior: {} learnable_prob: {}".format(model, dataset, prior, prob))
                            print("\n Corresponding gnelbo value:", min_gnelbo)
                            print("\n Minimum kl value:", min_kl)
                            print("\n Corresponding jsd value:", min_jsb)
                            print("\n optimal config:", succesful_configs[min_idx])
                            print("\n len config:", len(configurations))
                        
                            print(10*"#####")

                    else:
                        for prob in ['learnable_prob_False', 'learnable_prob_True']:
                            for intro_prior in ['intro_prior_False', 'intro_prior_True']:
                                
                                gnelbo = []
                                kl = []
                                jsb = []
                                succesful_configs = []
                                

                                data_folds = os.path.join(dataset_results, prior, "num_C_{}".format(num_C), prob, intro_prior)

                                configurations = os.listdir(data_folds)

                                for current_configuration_ in configurations:
                                    current_configuration = os.path.join(data_folds,current_configuration_)
                                    current_seed = os.listdir(current_configuration)[0]
                                    current_seed = os.path.join(current_configuration, current_seed)

                                    if os.path.exists(os.path.join(current_seed,'results_log_soft_intro_vae.txt')):


                                        with open(os.path.join(current_seed,'results_log_soft_intro_vae.txt')) as f:
                                                results = f.readlines()

                                        gnelbo.append(np.float64(results[-1].split("_")[11]))
                                        kl.append(np.float64(results[-1].split("_")[13]))
                                        jsb.append(np.float64(results[-1].split("_")[15]))
                                        succesful_configs.append(current_configuration_)


                                # Convert the lists to NumPy arrays for easier manipulation
                                gnelbo_arr = np.array(gnelbo)
                                kl_arr = np.array(kl)
                                jsb_arr = np.array(jsb)

                                # Find the index with the minimum value for jsb
                                min_idx = np.argmin(kl_arr)

                                # Extract the corresponding values for gnelbo and jsb
                                min_gnelbo = gnelbo_arr[min_idx]
                                min_kl = kl_arr[min_idx]
                                min_jsb = jsb_arr[min_idx]

                                # extract hyperparams
                                alpha, beta_rec, beta_kl, beta_neg = [float(i) for i in  succesful_configs[min_idx].split(" ")[1::2]] 
                                optimal_configurations = get_optimal_config(dataset, model, alpha, beta_rec, beta_kl, beta_neg, 
                                                                        prior_mode=prior, learnable_prob=prob.split("_")[-1] == "True", intro_prior=intro_prior.split("_")[-1] == "True", num_C=num_C, optimal_configurations=optimal_configurations)
                                # Print the results
                                print("\n model: {} Dataset: {} prior: {} learnable_prob: {} intro_prior: {}".format(model, dataset, prior, prob, intro_prior))
                                print("\n Corresponding gnelbo value:", min_gnelbo)
                                print("\n Minimum kl value:", min_kl)
                                print("\n Corresponding jsd value:", min_jsb)
                                print("\n optimal config:", succesful_configs[min_idx])
                                print("\n len config:", len(configurations))

                                print(10*"#####")


    # Save the dictionary as JSON
    with open(json_path, "w") as json_file:
        json.dump(optimal_configurations, json_file, indent=7)

