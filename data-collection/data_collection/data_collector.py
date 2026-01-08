import psutil
import math
import pandas as pd
import platform
import filelock
import time
import json
from datetime import datetime
from pathlib import Path

try: 
    import resource
except ImportError:
    resource = None


DATA_PATH = "data/data.csv"


def save(options, tree, saved_state, p):
    """
    Save training metrics to CSV with file locking for parallel writes
    """
    behaviours = options["behaviors"]

    enviroment = sorted(behaviours.keys())[0]  # key data

    enviroment_parameters = behaviours[enviroment]

    algorithm = enviroment_parameters["trainer_type"]  # key data

    hyperparameters = enviroment_parameters["hyperparameters"]

    learning_rate = hyperparameters["learning_rate"]  # key data

    batch_size = hyperparameters["batch_size"]  # key data

    buffer_size = hyperparameters["buffer_size"]  # key data

    beta = hyperparameters.get("beta", 0.0)  # key data

    epsilon = hyperparameters.get("epsilon", 0.0)  # key data

    lambd = hyperparameters.get("lambd", 0.0)  # key data

    num_epoch = hyperparameters.get("num_epoch", 0)  # key data

    tau = hyperparameters.get("tau", 0.0)  # key data

    init_entcoef = hyperparameters.get("init_entcoef", 0.0)  # key data

    reward_signal_steps_per_update = hyperparameters.get("reward_signal_steps_per_update", 0.0)  # key data

    gamma = enviroment_parameters["reward_signals"]["extrinsic"]["gamma"]  # key data

    strength = enviroment_parameters["reward_signals"]["extrinsic"]["strength"]  # key data

    summary_freq = enviroment_parameters["summary_freq"]  # key data

    network_settings = enviroment_parameters.get("network_settings", {})

    hidden_units = network_settings.get("hidden_units", 0)  # key data

    num_layers = network_settings.get("num_layers", 0)  # key data

    time_horizon = enviroment_parameters["time_horizon"]  # key data

    max_steps = enviroment_parameters.get("max_steps", 0)  # key data

    steps = saved_state.get(enviroment, {}).get("final_checkpoint", {}).get("steps", 0)
    if steps == 0:
        # If training completed to max_steps, use that
        steps = max_steps

    cpu_cores = psutil.cpu_count()  # key data

    ram_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # key data

    op_s = platform.system()

    timestamp = datetime.now().isoformat()

    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    # GPU information
    gpu_available = torch.cuda.is_available() if torch_available else False  # key data
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"  # key data
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_available else 0.0  # key data

    peak_ram_mb = 0  # key data
    if resource and op_s != 'Windows':
        peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if op_s == 'Darwin':
            peak_ram_mb /= 1024
    else:
        try:
            peak_ram_mb = p.memory_info().peak_wset / (1024 * 1024)
        except:
            peak_ram_mb = p.memory_info().rss / (1024 * 1024)

    training_duration_seconds = tree.get("total", 0)  # key data

    gauges = tree.get("gauges", None)

    final_mean_entropy = get_timers_data(gauges, enviroment + ".Policy.Entropy.mean")

    final_sum_entropy = get_timers_data(gauges, enviroment + ".Policy.Entropy.sum")

    final_mean_reward = get_timers_data(gauges, enviroment + ".Policy.ExtrinsicReward.mean")

    final_sum_reward = get_timers_data(gauges, enviroment + ".Policy.ExtrinsicReward.sum")

    final_cumulative_mean_reward = get_timers_data(gauges, enviroment + ".Environment.CumulativeReward.mean")

    final_cumulative_sum_reward = get_timers_data(gauges, enviroment + ".Environment.CumulativeReward.sum")

    final_mean_extrinsic_value_estimate = get_timers_data(gauges, enviroment + ".Policy.ExtrinsicValueEstimate.mean")

    final_sum_extrinsic_value_estimate = get_timers_data(gauges, enviroment + ".Policy.ExtrinsicValueEstimate.sum")

    final_mean_policy_loss = get_timers_data(gauges, enviroment + ".Losses.PolicyLoss.mean")

    final_sum_policy_loss = get_timers_data(gauges, enviroment + ".Losses.PolicyLoss.sum")

    final_mean_value_loss = get_timers_data(gauges, enviroment + ".Losses.ValueLoss.mean")

    final_sum_value_loss = get_timers_data(gauges, enviroment + ".Losses.ValueLoss.sum")

    final_mean_q1_loss = get_timers_data(gauges, enviroment + ".Losses.Q1Loss.mean")

    final_sum_q1_loss = get_timers_data(gauges, enviroment + ".Losses.Q1Loss.sum")

    final_mean_q2_loss = get_timers_data(gauges, enviroment + ".Losses.Q2Loss.mean")

    final_sum_q2_loss = get_timers_data(gauges, enviroment + ".Losses.Q2Loss.sum")

    final_mean_cont_entropy_coeff = get_timers_data(gauges, enviroment + ".Policy.ContinuousEntropyCoeff.mean")

    final_sum_cont_entropy_coeff = get_timers_data(gauges, enviroment + ".Policy.ContinuousEntropyCoeff.sum")

    final_learning_rate_mean = get_timers_data(gauges, enviroment + ".Policy.LearningRate.mean")

    final_learning_rate_sum = get_timers_data(gauges, enviroment + ".Policy.LearningRate.sum")

    num_parallel_agents = tree.get("children", {}).get("TrainerController.start_learning", {}).get("children", {}).get("trainer_threads", {}).get("total", 0)
    num_parallel_agents = math.ceil(num_parallel_agents) if num_parallel_agents else 0

    data = {
        'enviroment': [enviroment],
        'algorithm': [algorithm],
        'learning_rate': [learning_rate],
        'batch_size': [batch_size],
        'hidden_units': [hidden_units],
        'num_layers': [num_layers],
        'steps': [steps],
        'max_steps': [max_steps],
        'ram_gb': [ram_gb],
        'cpu_cores': [cpu_cores],
        'time_horizon': [time_horizon],
        'buffer_size': [buffer_size],
        'beta': [beta],
        'epsilon': [epsilon],
        'lambd': [lambd],
        'num_epoch': [num_epoch],
        'tau': [tau],
        'reward_signal_steps_per_update': [reward_signal_steps_per_update],
        'init_entcoef': [init_entcoef],
        'num_parallel_agents': [num_parallel_agents],
        'training_duration_seconds': [training_duration_seconds],
        'peak_ram_mb': [peak_ram_mb],
        'gamma': [gamma],
        'strength': [strength],
        'summary_freq': [summary_freq],
        'os': [op_s],
        'timestamp': [timestamp],
        'gpu_available': [gpu_available],
        'gpu_name': [gpu_name],
        'gpu_memory_gb': [gpu_memory_gb],
        'final_mean_entropy': [final_mean_entropy],
        'final_sum_entropy': [final_sum_entropy],
        'final_mean_reward': [final_mean_reward],
        'final_sum_reward': [final_sum_reward],
        'final_cumulative_mean_reward': [final_cumulative_mean_reward],
        'final_cumulative_sum_reward': [final_cumulative_sum_reward],
        'final_mean_extrinsic_value_estimate': [final_mean_extrinsic_value_estimate],
        'final_sum_extrinsic_value_estimate': [final_sum_extrinsic_value_estimate],
        'final_mean_policy_loss': [final_mean_policy_loss],
        'final_sum_policy_loss': [final_sum_policy_loss],
        'final_mean_value_loss': [final_mean_value_loss],
        'final_sum_value_loss': [final_sum_value_loss],
        'final_mean_q1_loss': [final_mean_q1_loss],
        'final_sum_q1_loss': [final_sum_q1_loss],
        'final_mean_q2_loss': [final_mean_q2_loss],
        'final_sum_q2_loss': [final_sum_q2_loss],
        'final_mean_cont_entropy_coeff': [final_mean_cont_entropy_coeff],
        'final_sum_cont_entropy_coeff': [final_sum_cont_entropy_coeff],
        'final_learning_rate_mean': [final_learning_rate_mean],
        'final_learning_rate_sum': [final_learning_rate_sum]
    }

    new_data = pd.DataFrame.from_dict(data)

    # Use file locking for safe parallel writes
    lock_path = DATA_PATH + ".lock"
    lock = filelock.FileLock(lock_path, timeout=30)

    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            with lock:
                # Read existing data
                try:
                    df = pd.read_csv(DATA_PATH)
                    df = pd.concat([df, new_data], ignore_index=True)
                except FileNotFoundError:
                    df = new_data
                except pd.errors.EmptyDataError:
                    df = new_data

                # Write back to file
                df.to_csv(DATA_PATH, index=False)

            print(f"[OK] Data saved to {DATA_PATH}")
            print(f"[OK] Environment: {enviroment}, Algorithm: {algorithm}")
            print(f"[OK] Steps: {steps:,}, Duration: {training_duration_seconds:.1f}s, Final Reward: {final_cumulative_mean_reward:.4f}")
            break

        except filelock.Timeout:
            retry_count += 1
            print(f"[WARN] File locked, retrying ({retry_count}/{max_retries})...")
            time.sleep(2)

        except Exception as e:
            print(f"[ERROR] Error saving data: {e}")
            break

    if retry_count >= max_retries:
        print(f"[ERROR] Failed to save data after {max_retries} retries")
        # Save to a temporary file as backup
        backup_path = DATA_PATH.replace(".csv", f"_backup_{timestamp.replace(':', '-')}.csv")
        new_data.to_csv(backup_path, index=False)
        print(f"[OK] Data saved to backup file: {backup_path}")


def get_timers_data(gauges, key):
    if gauges == None:
        return 0

    timer_dict = gauges.get(key, dict())
    return timer_dict.get("value", 0)
