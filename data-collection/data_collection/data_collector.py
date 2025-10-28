import psutil
import math
import pandas as pd
import os

DATA_PATH = "data/data.csv"

def save (options, tree, p):
    behaviours = options["behaviors"]

    enviroment = sorted(behaviours.keys())[0] #key data

    enviroment_parameters = behaviours[enviroment]

    algorithm = enviroment_parameters["trainer_type"] #key data

    hyperparameters = enviroment_parameters["hyperparameters"]

    learning_rate = hyperparameters["learning_rate"] #key data

    batch_size = hyperparameters["batch_size"] #key data

    buffer_size = hyperparameters["buffer_size"] #key data

    network_settings = enviroment_parameters["reward_signals"]["extrinsic"]["network_settings"]

    hidden_units = network_settings["hidden_units"] #key data

    num_layers = network_settings["num_layers"] #key data

    time_horizon = enviroment_parameters["time_horizon"] #key data

    max_steps = enviroment_parameters["max_steps"] #key data

    cpu_cores = psutil.cpu_count()

    ram_gb = psutil.virtual_memory().available

    # peak_ram_mb = p.memory_info().peak_wset
    #TODO Max fix and make it multiplatform

    training_duration_seconds = tree["total"]

    final_mean_reward = tree["gauges"][enviroment+".Policy.ExtrinsicReward.mean"]["value"]

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
    print(max_steps)
    print(cpu_cores)
    print(ram_gb)
    print(peak_ram_mb)
    print(training_duration_seconds)
    print(final_mean_reward)
    print(num_parallel_agents)
    print(os.getcwd())
    '''
    
    # data = {'enviroment':[enviroment], 'algorithm':[algorithm], 'learning_rate':[learning_rate], 'batch_size':[batch_size], 'hidden_units':[hidden_units], 'num_layers':[num_layers], 'max_steps':[max_steps], 'ram_gb':[ram_gb], 'cpu_cores':[cpu_cores], 'time_horizon':[time_horizon], 'buffer_size':[buffer_size], 'num_parallel_agents':[num_parallel_agents], 'training_duration_seconds':[training_duration_seconds], 'final_mean_reward':[final_mean_reward], 'peak_ram_mb':[peak_ram_mb]}

    #temp solution to peak ram not being multiplatform
    data = {'enviroment':[enviroment], 'algorithm':[algorithm], 'learning_rate':[learning_rate], 'batch_size':[batch_size], 'hidden_units':[hidden_units], 'num_layers':[num_layers], 'max_steps':[max_steps], 'ram_gb':[ram_gb], 'cpu_cores':[cpu_cores], 'time_horizon':[time_horizon], 'buffer_size':[buffer_size], 'num_parallel_agents':[num_parallel_agents], 'training_duration_seconds':[training_duration_seconds], 'final_mean_reward':[final_mean_reward]}

    new_data = pd.DataFrame.from_dict(data)

    try:
        df = pd.read_csv(DATA_PATH)

        df = pd.concat([df,new_data])

    except FileNotFoundError:
        df = new_data
    except pd.errors.EmptyDataError:
        df = new_data

    df.to_csv(DATA_PATH, index = False)





# def main():
#     learn.run_cli(learn.parse_command_line(),DataSaver())

# # For python debugger to directly run this script
# if __name__ == "__main__":
#     main()