"""
ML-Agents Config Generator
Generates multiple YAML config files with random hyperparameters
Organized by: data-collection/configs/{environment}/{algorithm}/config_{id}.yaml
"""

import yaml
import random
import os
from pathlib import Path


class ConfigGenerator:
    """
    Generates ML-Agents training configurations with randomized hyperparameters.
    Creates organized folder structure: configs/{env_name}/{algorithm}/config_{id}.yaml
    """

    def __init__(self, output_dir="data-collection/configs"):
        """
        Initialize the config generator.
        
        Args:
            output_dir: Root directory for generated configs (default: data-collection/configs)
        """
        self.output_dir = output_dir
        self.environments = [
            "3DBall", "Basic", "Crawler", "FoodCollector", 
            "GridWorld", "Hallway", "PushBlock", "Pyramids", 
            "Walker", "WallJump", "Worm"
        ]
        self.algorithms = ["ppo", "sac"]

    def generate_all_configs(self, configs_per_env_algo=10):
        """
        Generate config files for all environments and algorithms.
        
        Args:
            configs_per_env_algo: Number of config variants to generate per (environment, algorithm) pair
        
        Returns:
            dict: Summary of generated configs
        """
        total_generated = 0
        summary = {}

        print(f"🔧 Generating {configs_per_env_algo} configs per (environment, algorithm) pair...")
        print(f"📁 Output directory: {self.output_dir}/\n")

        for env_name in self.environments:
            summary[env_name] = {}
            
            for algo in self.algorithms:
                # Create directory structure: configs/{env}/{algo}/
                env_algo_dir = Path(self.output_dir) / env_name / algo
                env_algo_dir.mkdir(parents=True, exist_ok=True)

                configs_generated = 0

                for config_id in range(1, configs_per_env_algo + 1):
                    config_path = env_algo_dir / f"config_{config_id:03d}.yaml"
                    
                    # Generate config based on algorithm type
                    if algo == "ppo":
                        config_dict = self._generate_ppo_config(env_name)
                    else:  # sac
                        config_dict = self._generate_sac_config(env_name)

                    # Write to file
                    with open(config_path, 'w') as f:
                        yaml.dump({"behaviors": config_dict}, f, sort_keys=False)

                    configs_generated += 1
                    total_generated += 1

                summary[env_name][algo] = configs_generated
                print(f"✅ {env_name}/{algo}: Generated {configs_generated} configs")

        print(f"\n🎉 Total configs generated: {total_generated}")
        print(f"📊 Breakdown: {len(self.environments)} envs × {len(self.algorithms)} algos × {configs_per_env_algo} configs = {total_generated}")
        
        return summary

    def _generate_ppo_config(self, behaviour):
        """Generate a PPO config with random hyperparameters"""
        main_dict = {}
        
        # Handle special cases
        if behaviour == "WallJump":
            # WallJump has two sub-behaviors
            self._add_ppo_behavior("BigWallJump", main_dict)
            self._add_ppo_behavior("SmallWallJump", main_dict)
        else:
            self._add_ppo_behavior(behaviour, main_dict)

        return main_dict

    def _add_ppo_behavior(self, behaviour, main_dict):
        """Add a PPO behavior configuration to the main dict"""
        behaviour_dict = {}
        
        # Randomized hyperparameters
        batch_size = random.choice([64, 128, 256, 512, 1024, 2048])
        max_steps = random.randint(500000, 1000000)  
        buffer_size = random.choice([2048, 4096, 8192, 10240])
        
        hyperparameters_dict = {
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "learning_rate": 10 ** random.uniform(-4, -3),  # 1e-4 to 1e-3
            "beta": random.uniform(0.001, 0.01),  # Narrower range
            "epsilon": random.uniform(0.1, 0.3),  # Standard PPO range
            "lambd": random.uniform(0.9, 0.99),  # Slightly tighter
            "num_epoch": random.randint(3, 10),  # Better minimum
            "learning_rate_schedule": "constant"
        }

        # Network settings
        network_settings_dict = {
            "normalize": random.choice([True, False]),
            "hidden_units": random.randint(64, 1024),
            "num_layers": random.randint(1, 5),
            "vis_encode_type": "simple"
        }

        # Special case: Hallway needs memory
        if behaviour == "Hallway":
            network_settings_dict["memory"] = {
                "sequence_length": 64,
                "memory_size": 128
            }

        # Reward signals
        reward_signals_dict = {
            "extrinsic": {
                "gamma": random.uniform(0.8, 1.0),
                "strength": 1.0
            }
        }

        # Special case: Pyramids with curiosity
        if behaviour == "Pyramids":
            reward_signals_dict["curiosity"] = {
                "gamma": 0.99,
                "encoding_size": 0.2,
                "network_settings": {"hidden_units": 256},
                "learning_rate": 0.0003
            }

        # Build behavior dict
        behaviour_dict["trainer_type"] = "ppo"
        behaviour_dict["hyperparameters"] = hyperparameters_dict
        behaviour_dict["network_settings"] = network_settings_dict
        behaviour_dict["reward_signals"] = reward_signals_dict
        behaviour_dict["keep_checkpoints"] = 5
        behaviour_dict["max_steps"] = max_steps
        behaviour_dict["time_horizon"] = random.choice([64, 128, 256, 512, 1024])
        behaviour_dict["summary_freq"] = 10000

        main_dict[behaviour] = behaviour_dict

    def _generate_sac_config(self, behaviour):
        """Generate a SAC config with random hyperparameters"""
        main_dict = {}
        
        # Handle special cases
        if behaviour == "WallJump":
            self._add_sac_behavior("BigWallJump", main_dict)
            self._add_sac_behavior("SmallWallJump", main_dict)
        else:
            self._add_sac_behavior(behaviour, main_dict)

        return main_dict

    def _add_sac_behavior(self, behaviour, main_dict):
        """Add a SAC behavior configuration to the main dict"""
        behaviour_dict = {}
        
        # Randomized hyperparameters
        batch_size = random.choice([128, 256, 512, 1024])
        max_steps = random.randint(200000, 600000) 
        buffer_size = random.randint(100000, 1000000)
        
        hyperparameters_dict = {
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "learning_rate": 10 ** random.uniform(-4, -3.5),  # 1e-4 to 3e-4
            "buffer_init_steps": random.choice([1000, 5000, 10000]),
            "tau": random.uniform(0.003, 0.01),  # Tighter around 0.005
            "steps_per_update": random.choice([1, 5, 10, 20]),
            "save_replay_buffer": False,
            "init_entcoef": random.uniform(0.01, 1.0),  # Higher minimum
            "reward_signal_steps_per_update": random.choice([5, 10, 20]),
            "learning_rate_schedule": "constant"
        }

        # Network settings
        network_settings_dict = {
            "normalize": random.choice([True, False]),
            "hidden_units": random.randint(64, 1024),
            "num_layers": random.randint(1, 5),
            "vis_encode_type": "simple"
        }

        # Special case: Hallway needs memory
        if behaviour == "Hallway":
            network_settings_dict["memory"] = {
                "sequence_length": 64,
                "memory_size": 128
            }

        # Reward signals
        reward_signals_dict = {
            "extrinsic": {
                "gamma": random.uniform(0.8, 1.0),
                "strength": 1.0
            }
        }

        # Special case: Pyramids with curiosity
        if behaviour == "Pyramids":
            reward_signals_dict["curiosity"] = {
                "gamma": 0.99,
                "strength": 0.01,
                "learning_rate": 0.0003,
                "use_actions": True,
                "use_vail": False,
                "demo_path": "Project/Assets/ML-Agents/Examples/Pyramids/Demos/ExpertPyramid.demo"
            }

        # Build behavior dict
        behaviour_dict["trainer_type"] = "sac"
        behaviour_dict["hyperparameters"] = hyperparameters_dict
        behaviour_dict["network_settings"] = network_settings_dict
        behaviour_dict["reward_signals"] = reward_signals_dict
        behaviour_dict["keep_checkpoints"] = 5
        behaviour_dict["max_steps"] = max_steps
        behaviour_dict["time_horizon"] = random.randint(1, 2048)
        behaviour_dict["summary_freq"] = random.randint(5000, 100000)

        main_dict[behaviour] = behaviour_dict


# ============ USAGE EXAMPLE ============

if __name__ == "__main__":
    # Create generator
    generator = ConfigGenerator(output_dir="data-collection/configs")
    
    # Generate 10 config variants for each (environment, algorithm) pair
    # Total: 11 environments × 2 algorithms × 10 configs = 220 config files
    summary = generator.generate_all_configs(configs_per_env_algo=10)
    
    print("\nSummary by environment:")
    for env, algos in summary.items():
        total = sum(algos.values())
        print(f"  {env}: {total} configs ({algos})")
