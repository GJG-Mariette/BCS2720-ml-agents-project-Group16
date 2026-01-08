"""
Sequential Data Collection - Single Env/Algo
Runs all configs for ONE environment and ONE algorithm sequentially
Ensures accurate timing and resource tracking for each training run
"""

import subprocess
import time
import os
import sys
from pathlib import Path
import yaml
from datetime import datetime
import psutil

# ============ CONFIGURATION ============

TARGET_ENV = "GridWorld"      # Which environment to run
TARGET_ALGO = "ppo"        # Which algorithm (ppo or sac)

NUM_ENVS = 16       # Each training uses 16 Unity env copies

CONFIGS_DIR = "data-collection/configs"
BUILDS_DIR = "builds"

ENVIRONMENTS = {
    "3DBall": f"{BUILDS_DIR}/3DBall/3DBall.exe",
    "Basic": f"{BUILDS_DIR}/Basic/Basic.exe",
    "Crawler": f"{BUILDS_DIR}/Crawler/Crawler.exe",
    "FoodCollector": f"{BUILDS_DIR}/FoodCollector/FoodCollector.exe",
    "GridWorld": f"{BUILDS_DIR}/GridWorld/GridWorld.exe",
    "Hallway": f"{BUILDS_DIR}/Hallway/Hallway.exe",
    "PushBlock": f"{BUILDS_DIR}/PushBlock/PushBlock.exe",
    "Pyramids": f"{BUILDS_DIR}/Pyramids/Pyramids.exe",
    "Walker": f"{BUILDS_DIR}/Walker/Walker.exe",
    "WallJump": f"{BUILDS_DIR}/WallJump/WallJump.exe",
    "Worm": f"{BUILDS_DIR}/Worm/Worm.exe",
}

BASE_PORT = 5005

# Training timeout (2 hours)
TRAINING_TIMEOUT = 7200

# ============ FUNCTIONS ============

def collect_configs_for_env_algo(env_name, algo):
    """Get all config files for a specific env/algo combination"""
    config_dir = Path(CONFIGS_DIR) / env_name / algo

    if not config_dir.exists():
        print(f"[ERROR] Directory not found: {config_dir}")
        return []

    configs = list(config_dir.glob("config_*.yaml"))
    return sorted([str(c) for c in configs])


def kill_unity_processes(env_exe_name):
    """Kill any hanging Unity processes"""
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if env_exe_name in proc.info['name']:
                try:
                    proc.kill()
                except:
                    pass
    except:
        pass


def run_single_training(config_path, env_name, env_exe, run_num, total_runs):
    """Run one training session sequentially"""

    run_name = f"{env_name}_{TARGET_ALGO}_run_{run_num:03d}"

    # Kill any existing Unity processes before starting
    exe_name = Path(env_exe).stem
    kill_unity_processes(exe_name)

    cmd = [
        "mlagents-learn",
        config_path,
        f"--run-id={run_name}",
        f"--env={env_exe}",
        "--no-graphics",
        f"--base-port={BASE_PORT}",
        f"--num-envs={NUM_ENVS}",
        "--force"
    ]

    config_name = Path(config_path).name
    print(f"\n{'='*70}")
    print(f"[START] Run {run_num}/{total_runs}: {config_name}")
    print(f"{'='*70}")

    start_time = time.time()
    training_completed = False
    last_step = 0
    process = None

    try:
        # Run training
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Track progress from output
        for line in process.stdout:
            if line.strip():
                print(line.strip())  # Show all output

                # Extract step number
                if "Step:" in line:
                    try:
                        step_str = line.split("Step:")[1].split(".")[0].strip().replace(',', '')
                        last_step = int(step_str)
                    except:
                        pass

                # Check for completion
                if "Exported" in line or "Copied" in line:
                    training_completed = True

        process.wait(timeout=TRAINING_TIMEOUT)
        elapsed = time.time() - start_time

        # Check success indicators
        results_dir = Path(f"results/{run_name}")

        is_success = (
            process.returncode == 0 or
            (results_dir.exists() and any(results_dir.glob("*.onnx"))) or
            (training_completed and last_step > 1000)
        )

        if is_success:
            print(f"\n[SUCCESS] Run {run_num} completed!")
            print(f"   Steps: {last_step:,}")
            print(f"   Duration: {elapsed/60:.1f} minutes")
            print(f"   [INFO] Data automatically saved by ML-Agents to data/data.csv")

            return True, elapsed, last_step
        else:
            print(f"\n[FAILED] Run {run_num} failed!")
            print(f"   Return code: {process.returncode}")
            print(f"   Last step: {last_step}")
            return False, elapsed, last_step

    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] Run {run_num} timeout after {TRAINING_TIMEOUT/60:.0f} minutes")
        if process:
            process.kill()
            kill_unity_processes(exe_name)
        return False, TRAINING_TIMEOUT, last_step

    except Exception as e:
        print(f"\n[ERROR] Run {run_num} error: {e}")
        import traceback
        traceback.print_exc()
        if process:
            try:
                process.kill()
            except:
                pass
        kill_unity_processes(exe_name)
        return False, 0, 0

    finally:
        # Cleanup: ensure Unity processes are killed
        time.sleep(1)
        kill_unity_processes(exe_name)


def main():
    """Main function - run all configs sequentially"""

    print(f"""
{'='*70}
     SEQUENTIAL DATA COLLECTION - SINGLE ENV/ALGO
{'='*70}
  Environment: {TARGET_ENV}
  Algorithm:   {TARGET_ALGO.upper()}
  Unity envs per training: {NUM_ENVS}
  Port: {BASE_PORT}
{'='*70}
    """)

    # Validate environment
    if TARGET_ENV not in ENVIRONMENTS:
        print(f"[ERROR] Unknown environment: {TARGET_ENV}")
        return 1

    env_exe = ENVIRONMENTS[TARGET_ENV]
    if not os.path.exists(env_exe):
        print(f"[ERROR] Environment build not found: {env_exe}")
        print(f"        Expected: {env_exe}")
        return 1

    print(f"[OK] Environment build: {env_exe}")

    # Collect configs
    print(f"\n[INFO] Collecting configs for {TARGET_ENV}/{TARGET_ALGO}...")
    configs = collect_configs_for_env_algo(TARGET_ENV, TARGET_ALGO)

    if not configs:
        print(f"[ERROR] No config files found!")
        return 1

    print(f"[OK] Found {len(configs)} config files")

    # Check max_steps
    try:
        with open(configs[0], 'r') as f:
            first_config = yaml.safe_load(f)
            behavior_name = list(first_config['behaviors'].keys())[0]
            max_steps = first_config['behaviors'][behavior_name].get('max_steps', 'unknown')
            print(f"     Max steps: {max_steps:,}")
    except:
        pass

    # Estimate time
    avg_time_per_run = 5  # minutes (rough estimate for sequential)
    estimated_total = len(configs) * avg_time_per_run

    print(f"\n[INFO] Will run {len(configs)} trainings sequentially")
    print(f"       Estimated time: {estimated_total:.0f} minutes ({estimated_total/60:.1f} hours)")

    response = input(f"\n[PROMPT] Ready to run {len(configs)} trainings sequentially.\n         Press ENTER to start (or Ctrl+C to cancel): ")

    print(f"\n{'='*70}")
    print(f"[START] Beginning sequential data collection at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    # Run sequentially
    overall_start_time = time.time()
    successes = 0
    failures = 0
    results = []

    try:
        for i, config_path in enumerate(configs, 1):
            success, elapsed, steps = run_single_training(config_path, TARGET_ENV, env_exe, i, len(configs))

            if success:
                successes += 1
            else:
                failures += 1

            results.append({
                'run_num': i,
                'config': Path(config_path).name,
                'success': success,
                'elapsed': elapsed,
                'steps': steps
            })

            # Show progress
            total_elapsed = time.time() - overall_start_time
            remaining_runs = len(configs) - i
            avg_time = total_elapsed / i
            estimated_remaining = remaining_runs * avg_time

            print(f"\n{'='*70}")
            print(f"[PROGRESS] {i}/{len(configs)} runs ({i/len(configs)*100:.0f}%)")
            print(f"           Success: {successes} | Failed: {failures}")
            print(f"           Total elapsed: {total_elapsed/60:.1f} min")
            print(f"           Estimated remaining: {estimated_remaining/60:.1f} min")
            print(f"           Average time per run: {avg_time/60:.1f} min")
            print(f"{'='*70}")

    except KeyboardInterrupt:
        print(f"\n\n[INTERRUPT] Data collection interrupted by user!")
        # Kill all Unity processes
        for env in ENVIRONMENTS.values():
            exe_name = Path(env).stem
            kill_unity_processes(exe_name)

    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        total_time = time.time() - overall_start_time

        print(f"\n{'='*70}")
        print(f"[COMPLETE] SEQUENTIAL DATA COLLECTION FINISHED")
        print(f"{'='*70}")
        print(f"Environment: {TARGET_ENV}")
        print(f"Algorithm: {TARGET_ALGO}")
        print(f"Success: {successes}/{len(configs)} ({successes/len(configs)*100 if len(configs) > 0 else 0:.0f}%)")
        print(f"Failed: {failures}/{len(configs)}")
        print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")

        if successes > 0:
            avg_time = total_time / successes
            print(f"Average successful run: {avg_time/60:.1f} min")

        # Show failed runs if any
        if failures > 0:
            failed_runs = [r for r in results if not r['success']]
            print(f"\n[FAILED RUNS]:")
            for r in failed_runs:
                print(f"  - Run {r['run_num']:02d}: {r['config']}")

        print(f"\n[OUTPUT]")
        print(f"  Results: results/")
        print(f"  Data CSV: data/data.csv")
        print(f"{'='*70}\n")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
