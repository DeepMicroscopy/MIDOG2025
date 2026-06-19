import os
import sys
import json
import time
import random
import select
import subprocess
from pathlib import Path

# --- 1. UTILS ---

def get_gpu_usage():
    try:
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode('utf-8')
        return int(output.strip())
    except: return 0

def wait_for_idle_gpu(baseline):
    while get_gpu_usage() > baseline:
        print(f"⏳ Waiting for GPU idle ({get_gpu_usage()}MB > {baseline}MB)...", end="\r")
        time.sleep(5)

def run_container(cmd):
    peak_mem = 0
    all_logs = []
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    while True:
        while select.select([proc.stdout], [], [], 0)[0]:
            line = proc.stdout.readline()
            if line: all_logs.append(line.strip())
            else: break
        peak_mem = max(peak_mem, get_gpu_usage())
        if proc.poll() is not None: break
        time.sleep(1.0)
    
    if proc.returncode != 0:
        print(f"\n❌ FAILURE! Logs:\n" + "\n".join(all_logs[-20:]))
        sys.exit(1)
    return peak_mem

# --- 2. ABLATION DISCOVERY ---

def get_ablation_tasks(root_dir):
    """Finds all leaf directories in the ablations folder."""
    tasks = []
    for root, dirs, files in os.walk(root_dir):
        if not dirs and files: # It's a leaf node with files
            tasks.append(root)
    return tasks

# --- 3. MAIN LOGIC ---

def execute_ablation(ablation_path, img_id, username, idle_limit):
    # Determine naming (e.g., ensembling/1 -> ensembling_1)
    rel_path = os.path.relpath(ablation_path, start="./code/ablations")
    ablation_name = rel_path.replace("/", "_")
    
    print(f"\n🚀 STARTING ABLATION: {ablation_name}")
    
    # Generate Drop-in Mounts
    # Only mount the files present in the ablation folder over /opt/app/
    drop_in_mounts = ""
    for filename in os.listdir(ablation_path):
        local_file = os.path.abspath(os.path.join(ablation_path, filename))
        if os.path.isfile(local_file):
            drop_in_mounts += f" -v {local_file}:/opt/app/{filename}:rw"

    random.seed(42)
    stats = {}

    for k in range(238):
        out_dir = os.path.abspath(f'../../outputs/{username}/ablations/{ablation_name}/{k+1}')
        
        if os.path.exists(out_dir+os.sep+'multiple-mitotic-figure-classification.json'):
            continue
        os.makedirs(out_dir, exist_ok=True)
        os.system(f"chmod -R 777 {out_dir}")

        cmd = f"""podman run --rm \
            -v ../../inputs/{k+1}:/input:ro \
            -v {out_dir}:/output:rw \
            {drop_in_mounts} \
            --device nvidia.com/gpu=all \
            --shm-size 1024M \
            {img_id}"""


        print(f"🔄 [{ablation_name}] Case {k+1}/238", end="\r")

        start_time = time.time()
        peak_vram = run_container(cmd)
        duration = time.time() - start_time


    # Save Stats
    with open(f'../../outputs/{username}/ablations/{ablation_name}/stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\n✅ Finished Ablation: {ablation_name}")

if __name__ == "__main__":
    # 1. Load Env
    if not os.path.exists("env"):
        print("❌ Error: 'env' file missing.")
        sys.exit(1)
    
    env = dict(line.strip().split('=') for line in open("env") if '=' in line)
    img_id, user = env['CONTAINER_NAME'], env['USERNAME']
    
    # 2. Detect Baseline
    IDLE_LIMIT = get_gpu_usage() + 100
    
    # 3. Find all ablations
    ablation_root = "./code/ablations"
    all_tasks = get_ablation_tasks(ablation_root)
    
    print(f"🔍 Found {len(all_tasks)} ablation tasks to run.")
    
    for task_path in sorted(all_tasks):
        execute_ablation(task_path, img_id, user, IDLE_LIMIT)

    print("\n🏁 ALL ABLATIONS COMPLETE.")
