import psutil
import math
import pandas as pd
import platform

try:
    import resource
except ImportError:
    resource = None


DATA_PATH = "data/data.csv"

def save (options, tree, saved_state, p):
    behaviours = options["behaviors"]

    enviroment = sorted(behaviours.keys())[0] #key data

    enviroment_parameters = behaviours[enviroment]

    algorithm = enviroment_parameters["trainer_type"] #key data

    hyperparameters = enviroment_parameters["hyperparameters"]

    learning_rate = hyperparameters["learning_rate"] #key data

    batch_size = hyperparameters["batch_size"] #key data

    buffer_size = hyperparameters["buffer_size"] #key data

    beta = hyperparameters.get("beta",0.0) #key data

    epsilon = hyperparameters.get("epsilon",0.0) #key data

    lambd = hyperparameters.get("lambd",0.0) #key data

    num_epoch = hyperparameters.get("num_epoch",0) #key data

    tau = hyperparameters.get("tau",0.0) #key data

    init_entcoef = hyperparameters.get("init_entcoef",0.0) #key data

    reward_signal_steps_per_update = hyperparameters.get("init_entcoef",0.0) #key data

    gamma = enviroment_parameters["reward_signals"]["extrinsic"]["gamma"] #key data

    strength = enviroment_parameters["reward_signals"]["extrinsic"]["strength"] #key data

    summary_freq = enviroment_parameters["summary_freq"] #key data

    network_settings = enviroment_parameters["reward_signals"]["extrinsic"]["network_settings"]

    hidden_units = network_settings["hidden_units"] #key data

    num_layers = network_settings["num_layers"] #key data

    time_horizon = enviroment_parameters["time_horizon"] #key data

    steps = saved_state[enviroment]["final_checkpoint"]["steps"] #key data

    cpu_cores = psutil.cpu_count() #key data

    ram_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024) #key data

    op_s = platform.system()

    peak_ram_mb = 0 #key data
    if resource and op_s != 'Windows':

        peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        if op_s == 'Darwin':
            peak_ram_mb /= 1024

    else:
        peak_ram_mb = p.memory_info().peak_wset / (1024 * 1024)
    

    training_duration_seconds = tree["total"] #key data

    gauges = tree.get("gauges",None)

    final_mean_entropy = get_timers_data(gauges,enviroment+".Policy.Entropy.mean") 

    final_sum_entropy = get_timers_data(gauges,enviroment+".Policy.Entropy.sum")

    final_mean_reward = get_timers_data(gauges,enviroment+".Policy.ExtrinsicReward.mean")

    final_sum_reward = get_timers_data(gauges,enviroment+".Policy.ExtrinsicReward.sum")

    final_cumulative_mean_reward = get_timers_data(gauges,enviroment+".Environment.CumulativeReward.mean")

    final_cumulative_sum_reward = get_timers_data(gauges,enviroment+".Environment.CumulativeReward.sum")

    final_mean_extrinsic_value_estimate = get_timers_data(gauges,enviroment+".Policy.ExtrinsicValueEstimate.mean")

    final_sum_extrinsic_value_estimate = get_timers_data(gauges,enviroment+".Policy.ExtrinsicValueEstimate.sum")

    final_mean_policy_loss = get_timers_data(gauges,enviroment+".Losses.PolicyLoss.mean")

    final_sum_policy_loss = get_timers_data(gauges,enviroment+".Losses.PolicyLoss.sum")

    final_mean_value_loss = get_timers_data(gauges,enviroment+".Losses.ValueLoss.mean")

    final_sum_value_loss = get_timers_data(gauges,enviroment+".Losses.ValueLoss.sum")

    final_mean_q1_loss = get_timers_data(gauges,enviroment+".Losses.Q1Loss.mean")

    final_sum_q1_loss = get_timers_data(gauges,enviroment+".Losses.Q1Loss.sum")

    final_mean_q2_loss = get_timers_data(gauges,enviroment+".Losses.Q2Loss.mean")

    final_sum_q2_loss = get_timers_data(gauges,enviroment+".Losses.Q2Loss.sum")
    
    final_mean_cont_entropy_coeff = get_timers_data(gauges,enviroment+".Policy.ContinuousEntropyCoeff.mean")

    final_sum_cont_entropy_coeff = get_timers_data(gauges,enviroment+".Policy.ContinuousEntropyCoeff.sum")

    final_learning_rate_mean = get_timers_data(gauges,enviroment+".Policy.LearningRate.mean")

    final_learning_rate_sum = get_timers_data(gauges,enviroment+".Policy.LearningRate.sum")

    num_parallel_agents = tree["children"]["TrainerController.start_learning"]["children"]["trainer_threads"]["total"]
    num_parallel_agents = math.ceil(num_parallel_agents)

    '''
    print(enviroment)
    print(algorithm)
    print(learning_rate)
    print(batch_size)
    print(buffer_size)
    print(hidden_units)
    print(num_layers)
    print(time_horizon)
    print(steps)
    print(cpu_cores)
    print(ram_gb)
    print(peak_ram_mb)
    print(training_duration_seconds)
    print(final_mean_reward)
    print(num_parallel_agents)
    print(os.getcwd())
    '''
    
    data = {'enviroment':[enviroment], 'algorithm':[algorithm], 'learning_rate':[learning_rate], 'batch_size':[batch_size], 'hidden_units':[hidden_units], 'num_layers':[num_layers], 'steps':[steps], 'ram_gb':[ram_gb], 'cpu_cores':[cpu_cores], 'time_horizon':[time_horizon], 'buffer_size':[buffer_size], 'beta': [beta], 'epsilon':[epsilon], 'lambd':[lambd], 'num_epoch':num_epoch, 'tau':[tau], 'reward_signal_steps_per_update':[reward_signal_steps_per_update], 'init_entcoef':[init_entcoef], 'num_parallel_agents':[num_parallel_agents], 'training_duration_seconds':[training_duration_seconds], 'peak_ram_mb':[peak_ram_mb], 'gamma':[gamma], 'strength':[strength], 'summary_freq':[summary_freq],'final_mean_entropy':[final_mean_entropy], 'final_sum_entropy':[final_sum_entropy], 'final_mean_reward':[final_mean_reward], 'final_sum_reward':[final_sum_reward], 'final_cumulative_mean_reward':[final_cumulative_mean_reward], 'final_cumulative_sum_reward':[final_cumulative_sum_reward], 'final_mean_extrinsic_value_estimate':[final_mean_extrinsic_value_estimate],'final_sum_extrinsic_value_estimate':[final_sum_extrinsic_value_estimate], 'final_mean_policy_loss':[final_mean_policy_loss],'final_sum_policy_loss':[final_sum_policy_loss],'final_mean_value_loss':[final_mean_value_loss],'final_sum_value_loss':[final_sum_value_loss], 'final_mean_q1_loss':[final_mean_q1_loss],'final_sum_q1_loss':[final_sum_q1_loss],'final_mean_q2_loss':[final_mean_q2_loss],'final_sum_q2_loss':[final_sum_q2_loss],'final_mean_cont_entropy_coeff':[final_mean_cont_entropy_coeff],'final_sum_cont_entropy_coeff':[final_sum_cont_entropy_coeff],'final_learning_rate_mean':[final_learning_rate_mean],'final_learning_rate_sum':[final_learning_rate_sum]}

    new_data = pd.DataFrame.from_dict(data)

    try:
        df = pd.read_csv(DATA_PATH)

        df = pd.concat([df,new_data])

    except FileNotFoundError:
        df = new_data
    except pd.errors.EmptyDataError:
        df = new_data

    df.to_csv(DATA_PATH, index = False)


def get_timers_data(gauges,key):
    if gauges == None:
        return 0
    
    timer_dict = gauges.get(key,dict())
    return timer_dict.get("value",0)