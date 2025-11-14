import yaml
import random

config_path = "data/config.yaml"


def generate_config():
    behaviours = ["3DBall","Basic","Crawler","FoodCollector","GridWorld","Hallway","Pyramids","Walker","WallJump","Worm","PushBlock","DungeonEscape","SoccerTwos","StrikerVsGoalie"]
    behaviour = random.choice(behaviours)

    trainer_types = [generate_poca,generate_ppo,generate_sac]
    trainer_type = random.choice(trainer_types) ##can be modified for weighted chance

    main_dict = {}

    if behaviour == "StrikerVsGoalie":
        trainer_type("Striker",main_dict=main_dict)
        trainer_type("Striker",main_dict=main_dict)
        return
    elif behaviour == "WallJump":
        trainer_type("BigWallJump",main_dict=main_dict)
        trainer_type("SmallWallJump",main_dict=main_dict)
    else:
        trainer_type(behaviour,main_dict=main_dict)

    print(f"Config saved at {config_path} for {behaviour}")

    with open(config_path, 'w+') as ff:
        yaml.dump({"behaviors":main_dict}, ff,sort_keys=False)

def generate_poca(behaviour, main_dict = {}):
    behaviour_dict = {}

    hyperparameters_dict = {}

    min_batch_size = 16
    max_batch_size = 4096

    batch_size = random.randint(min_batch_size,max_batch_size)

    min_max_steps = 100000
    max_max_steps = 20000000
    
    max_steps = random.randint(min_max_steps,max_max_steps)

    min_buffer_size = batch_size
    max_buffer_size = max_steps

    buffer_size = random.randint(min_buffer_size,max_buffer_size)

    min_learning_rate = 0.0001
    max_learning_rate = 0.01

    learning_rate = random.uniform(min_learning_rate,max_learning_rate)

    min_beta = 0.0001
    max_beta = 0.1

    beta = random.uniform(min_beta,max_beta)

    min_epsilon = 0.01
    max_epsilon = 0.5

    epsilon = random.uniform(min_epsilon,max_epsilon)

    min_lambd = 0.8
    max_lambd = 1.0

    lambd = random.uniform(min_lambd,max_lambd)

    min_num_epoch = 1
    max_num_epoch = 5

    num_epoch = random.randint(min_num_epoch,max_num_epoch)

    learning_rate_schedule = "linear"

    hyperparameters_dict["batch_size"] = batch_size
    hyperparameters_dict["buffer_size"] = buffer_size
    hyperparameters_dict["learning_rate"] = learning_rate
    hyperparameters_dict["beta"] = beta
    hyperparameters_dict["epsilon"] = epsilon
    hyperparameters_dict["lambd"] = lambd
    hyperparameters_dict["num_epoch"] = num_epoch
    hyperparameters_dict["learning_rate_schedule"] = learning_rate_schedule

    network_settings_dict = {}
    
    normalize = random.choice([True,False])

    min_hidden_units = 64
    max_hidden_units = 1024

    hidden_units = random.randint(min_hidden_units,max_hidden_units)

    min_layers = 1
    max_layers = 5

    layers = random.randint(min_layers,max_layers)

    vis_encode_type = "simple"

    network_settings_dict["normalize"] = normalize
    network_settings_dict["hidden_units"] = hidden_units
    network_settings_dict["layers"] = layers
    network_settings_dict["vis_encode_type"] = vis_encode_type

    if behaviour == "Hallway": network_settings_dict["memory"] = {"sequence_length":64, "memory_size":128} #static for purpose of consistency of this parameters solely for hallway

    reward_signals_dict = {}

    min_gamma = 0.8
    max_gamma = 1.0

    gamma = random.uniform(min_gamma,max_gamma)

    strength = 1.0

    reward_signals_dict["extrinsic"] = {"gamma":gamma,"strength":strength}

    if behaviour == "Pyramids": reward_signals_dict["curiosity"] = {"gamma":0.99, "memory_size":0.2, "network_settings":{"hidden_units":256},"learning_rate":0.0003}
    
    min_time_horizon = 1
    max_time_horizon = 2048

    keep_checkpoints = 5

    time_horizon = random.randint(min_time_horizon,max_time_horizon)

    min_summary_freq = 5000
    max_summary_freq = 100000

    summary_freq = random.randint(min_summary_freq,max_summary_freq)

    behaviour_dict["trainer_type"] = "ppo"
    behaviour_dict["hyperparameters"] = hyperparameters_dict
    behaviour_dict["network_settings"] = network_settings_dict
    behaviour_dict["reward_signals"] = reward_signals_dict
    behaviour_dict["keep_checkpoints"] = keep_checkpoints
    behaviour_dict["max_steps"] = max_steps
    behaviour_dict["time_horizon"] = time_horizon
    behaviour_dict["summary_freq"] = summary_freq

    if (behaviour == "StrikersVsGoalies" or behaviour == "SoccerTwos"): 
        behaviour_dict["self_play"] = {"save_steps": 50000,"team_change": 200000,"swap_steps": 4000,"window": 10,"play_against_latest_model_ratio": 0.5,"initial_elo": 1200.0} 
        #static for purpose of consistency of this parameters solely for football games

    main_dict[behaviour] = behaviour_dict



def generate_ppo(behaviour, main_dict = {}):
    behaviour_dict = {}

    hyperparameters_dict = {}

    min_batch_size = 16
    max_batch_size = 4096

    batch_size = random.randint(min_batch_size,max_batch_size)

    min_max_steps = 100000
    max_max_steps = 20000000
    
    max_steps = random.randint(min_max_steps,max_max_steps)

    min_buffer_size = batch_size
    max_buffer_size = max_steps

    buffer_size = random.randint(min_buffer_size,max_buffer_size)

    min_learning_rate = 0.0001
    max_learning_rate = 0.01

    learning_rate = random.uniform(min_learning_rate,max_learning_rate)

    min_beta = 0.0001
    max_beta = 0.1

    beta = random.uniform(min_beta,max_beta)

    min_epsilon = 0.01
    max_epsilon = 0.5

    epsilon = random.uniform(min_epsilon,max_epsilon)

    min_lambd = 0.8
    max_lambd = 1.0

    lambd = random.uniform(min_lambd,max_lambd)

    min_num_epoch = 1
    max_num_epoch = 5

    num_epoch = random.randint(min_num_epoch,max_num_epoch)

    learning_rate_schedule = "constant"

    hyperparameters_dict["batch_size"] = batch_size
    hyperparameters_dict["buffer_size"] = buffer_size
    hyperparameters_dict["learning_rate"] = learning_rate
    hyperparameters_dict["beta"] = beta
    hyperparameters_dict["epsilon"] = epsilon
    hyperparameters_dict["lambd"] = lambd
    hyperparameters_dict["num_epoch"] = num_epoch
    hyperparameters_dict["learning_rate_schedule"] = learning_rate_schedule

    network_settings_dict = {}
    
    normalize = random.choice([True,False])

    min_hidden_units = 64
    max_hidden_units = 1024

    hidden_units = random.randint(min_hidden_units,max_hidden_units)

    min_layers = 1
    max_layers = 5

    layers = random.randint(min_layers,max_layers)

    vis_encode_type = "simple"

    network_settings_dict["normalize"] = normalize
    network_settings_dict["hidden_units"] = hidden_units
    network_settings_dict["layers"] = layers
    network_settings_dict["vis_encode_type"] = vis_encode_type

    if behaviour == "Hallway": network_settings_dict["memory"] = {"sequence_length":64, "memory_size":128} #static for purpose of consistency of this parameters solely for hallway

    reward_signals_dict = {}

    min_gamma = 0.8
    max_gamma = 1.0

    gamma = random.uniform(min_gamma,max_gamma)

    strength = 1.0

    reward_signals_dict["extrinsic"] = {"gamma":gamma,"strength":strength}

    if behaviour == "Pyramids": reward_signals_dict["curiosity"] = {"gamma":0.99, "memory_size":0.2, "network_settings":{"hidden_units":256},"learning_rate":0.0003}
    
    min_time_horizon = 1
    max_time_horizon = 2048

    keep_checkpoints = 5

    time_horizon = random.randint(min_time_horizon,max_time_horizon)

    min_summary_freq = 5000
    max_summary_freq = 100000

    summary_freq = random.randint(min_summary_freq,max_summary_freq)

    behaviour_dict["trainer_type"] = "poca"
    behaviour_dict["hyperparameters"] = hyperparameters_dict
    behaviour_dict["network_settings"] = network_settings_dict
    behaviour_dict["reward_signals"] = reward_signals_dict
    behaviour_dict["keep_checkpoints"] = keep_checkpoints
    behaviour_dict["max_steps"] = max_steps
    behaviour_dict["time_horizon"] = time_horizon
    behaviour_dict["summary_freq"] = summary_freq

    if (behaviour == "StrikersVsGoalies" or behaviour == "SoccerTwos"): 
        behaviour_dict["self_play"] = {"save_steps": 50000,"team_change": 200000,"swap_steps": 4000,"window": 10,"play_against_latest_model_ratio": 0.5,"initial_elo": 1200.0} 
        #static for purpose of consistency of this parameters solely for football games

    main_dict[behaviour] = behaviour_dict



def generate_sac(behaviour, main_dict = {}):
    behaviour_dict = {}

    hyperparameters_dict = {}

    min_batch_size = 16
    max_batch_size = 4096

    batch_size = random.randint(min_batch_size,max_batch_size)

    min_max_steps = 100000
    max_max_steps = 20000000
    
    max_steps = random.randint(min_max_steps,max_max_steps)

    min_buffer_size = batch_size
    max_buffer_size = max_steps

    buffer_size = random.randint(min_buffer_size,max_buffer_size)

    min_learning_rate = 0.0001
    max_learning_rate = 0.01

    learning_rate = random.uniform(min_learning_rate,max_learning_rate)

    buffer_init_steps = random.choice([0,500,100])
    
    min_tau = 0.002
    max_tau = 0.02

    tau = random.uniform(min_tau,max_tau)

    steps_per_update = random.choice([5,10,15,20,25,30])

    save_replay_buffer = False

    min_init_entcoef = 0.0005
    max_init_entcoef = 1.0

    init_entcoef = random.uniform(min_init_entcoef,max_init_entcoef)

    reward_signal_steps_per_update = random.choice([5,10,15,20,25,30])

    learning_rate_schedule = "linear"

    hyperparameters_dict["batch_size"] = batch_size
    hyperparameters_dict["buffer_size"] = buffer_size
    hyperparameters_dict["learning_rate"] = learning_rate
    hyperparameters_dict["buffer_init_steps"] = buffer_init_steps
    hyperparameters_dict["tau"] = tau
    hyperparameters_dict["steps_per_update"] = steps_per_update
    hyperparameters_dict["save_replay_buffer"] = save_replay_buffer
    hyperparameters_dict["init_entcoef"] = init_entcoef
    hyperparameters_dict["reward_signal_steps_per_update"] = reward_signal_steps_per_update
    hyperparameters_dict["learning_rate_schedule"] = learning_rate_schedule

    network_settings_dict = {}
    
    normalize = random.choice([True,False])

    min_hidden_units = 64
    max_hidden_units = 1024

    hidden_units = random.randint(min_hidden_units,max_hidden_units)

    min_layers = 1
    max_layers = 5

    num_layers = random.randint(min_layers,max_layers)

    vis_encode_type = "simple"

    network_settings_dict["normalize"] = normalize
    network_settings_dict["hidden_units"] = hidden_units
    network_settings_dict["num_layers"] = num_layers
    network_settings_dict["vis_encode_type"] = vis_encode_type

    if behaviour == "Hallway": network_settings_dict["memory"] = {"sequence_length":64, "memory_size":128} #static for purpose of consistency of this parameters solely for hallway

    reward_signals_dict = {}

    min_gamma = 0.8
    max_gamma = 1.0

    gamma = random.uniform(min_gamma,max_gamma)

    strength = 1.0

    reward_signals_dict["extrinsic"] = {"gamma":gamma,"strength":strength}

    if behaviour == "Pyramids": 
        reward_signals_dict["curiosity"] = {"gamma": 0.99,"strength": 0.01,"learning_rate": 0.0003,"use_actions": True,"use_vail": False,"demo_path": "Project/Assets/ML-Agents/Examples/Pyramids/Demos/ExpertPyramid.demo"}
    
    min_time_horizon = 1
    max_time_horizon = 2048

    keep_checkpoints = 5

    time_horizon = random.randint(min_time_horizon,max_time_horizon)

    min_summary_freq = 5000
    max_summary_freq = 100000

    summary_freq = random.randint(min_summary_freq,max_summary_freq)

    behaviour_dict["trainer_type"] = "sac"
    behaviour_dict["hyperparameters"] = hyperparameters_dict
    behaviour_dict["network_settings"] = network_settings_dict
    behaviour_dict["reward_signals"] = reward_signals_dict
    behaviour_dict["keep_checkpoints"] = keep_checkpoints
    behaviour_dict["max_steps"] = max_steps
    behaviour_dict["time_horizon"] = time_horizon
    behaviour_dict["summary_freq"] = summary_freq

    if (behaviour == "StrikersVsGoalies" or behaviour == "SoccerTwos"): 
        behaviour_dict["self_play"] = {"save_steps": 50000,"team_change": 200000,"swap_steps": 4000,"window": 10,"play_against_latest_model_ratio": 0.5,"initial_elo": 1200.0} 
        #static for purpose of consistency of this parameters solely for football games

    main_dict[behaviour] = behaviour_dict

generate_config()
