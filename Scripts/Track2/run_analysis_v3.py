import os
import re
import glob
import random
import time
import subprocess
import json
import shutil
import sys
import select
from pathlib import Path

# --- 1. SETUP & UTILITIES ---

def get_context():
    """Load latest image and detect ID via Repository name filtering."""
    tars = glob.glob("*.tar.gz")
    if not tars:
        print("❌ Error: No .tar.gz image found in the current directory.")
        sys.exit(1)
    
    latest_tar = max(tars, key=os.path.getsize)
    print(f"📦 Loading Image: {latest_tar}...")
    
    # 1. Load the image and capture output to find the Loaded image name
    result = subprocess.check_output(f"podman load -i {latest_tar}", shell=True, stderr=subprocess.STDOUT).decode()
    print(result)

    # 2. Extract the Image Name from the load output
    # Usually: "Loaded image: localhost/my-image:latest" or "Loaded image ID: ..."
    name_match = re.search(r"Loaded image: (.*)", result)
    
    if name_match:
        image_ref = name_match.group(1).strip()
        print(f"🔍 Found Image Name: {image_ref}")
        cmd = f"podman images --format '{{{{.ID}}}}' {image_ref} | head -n 1"
    else:
        # Fallback: Just get the absolute newest by the 'last pulled/loaded' logic
        # Some podman versions support --sort-by, but 'head -n 1' on a fresh load is risky
        # if the created date is old. We'll try to find the ID directly if it was printed.
        id_match = re.search(r"Loaded image ID: ([a-f0-9]+)", result)
        if id_match:
            img_id = id_match.group(1).strip()
        else:
            print("⚠️ Could not parse name, falling back to newest ID (might be wrong if created dates vary).")
            cmd = "podman images --format '{{{{.ID}}}}' | head -n 1"
            img_id = subprocess.check_output(cmd, shell=True).decode().strip()
            
    if 'cmd' in locals():
        img_id = subprocess.check_output(cmd, shell=True).decode().strip()

    if not img_id:
        print("❌ Error: Could not resolve Image ID.")
        sys.exit(1)

    # Username is the current folder name
    username = Path(os.getcwd()).name 
    return img_id, username

def prepare_code(image_id):
    """Extract code and inject the timing template."""
    print(f"📂 Extracting code from Image: {image_id}...")
    os.makedirs("./code", exist_ok=True)
    os.system("chmod 777 ./code")
    
    extract_cmd = f"podman run --rm -v ./code:/xchange:rw --entrypoint bash {image_id} -c 'cp -r /opt/app/* /xchange/'"
    if os.system(extract_cmd) != 0:
        print("❌ Error: Code extraction failed.")
        sys.exit(1)

    orig_path = "./code/inference.py"
    timed_path = "./code/inference_timed.py"
    
    template = """import time
# === TIMING TEMPLATE ===
# t = time.time()
# ... (inference) ...
# with open("/output/time.csv", 'w') as f:
#     f.write('%.04f' % (time.time() - t))
# =======================
"""
    if os.path.exists(orig_path) and not os.path.exists(timed_path):
        with open(orig_path, 'r') as f: content = f.read()
        with open(timed_path, 'w') as f: f.write(template + "\n" + content)

    editor = "micro" if shutil.which("micro") else "nano"
    print(f"⌨️  Opening {timed_path}...")
    subprocess.call([editor, timed_path])
    input("\n✅ Press [ENTER] to start profiling...")

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

# --- 2. EXECUTION ENGINE ---

def run_container(cmd, is_timing):
    peak_mem = 0
    all_logs = []
    
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    while True:
        while select.select([proc.stdout], [], [], 0)[0]:
            line = proc.stdout.readline()
            if line:
                all_logs.append(line.strip())
            else:
                break
        
        peak_mem = max(peak_mem, get_gpu_usage())
        if proc.poll() is not None:
            break
        time.sleep(0.1 if is_timing else 1.0)

    # Final read
    remaining = proc.stdout.read()
    if remaining: all_logs.extend(remaining.splitlines())

    if proc.returncode != 0:
        print(f"\n❌ FAILURE on Case! Code: {proc.returncode}")
        print("--- FULL LOGS ---")
        for l in all_logs: print(f"| {l}")
        sys.exit(1)

    return peak_mem

# --- 3. MAIN ---

if __name__ == "__main__":
    if os.path.exists('env'):
        for line in open('env','r').readlines():
            k,v = line.strip().split('=')
            if k=='CONTAINER_NAME':
                img_id=v
            elif k=='USERNAME':
                username=v
            else:
                print('Error: Unknown key in env file: ',k)
                sys.exit(1)
    else:
        img_id, username = get_context()
    
    with open("env", "w") as f:
        f.write(f"CONTAINER_NAME={img_id}\nUSERNAME={username}\n")

    IDLE_USAGE = get_gpu_usage()
    IDLE_THRESHOLD = 400 # firefox, X11, etc.

    print(f"✅ User: {username} | Image: {img_id}")
    print(f"📊 Idle VRAM: {get_gpu_usage()}MB. Threshold: {IDLE_THRESHOLD}MB")

    prepare_code(img_id)

    random.seed(42)
    all_cases = list(range(238))
    timing_subset = set(random.sample(all_cases, 50))
    stats = {}

    for k in all_cases:
        out_dir = os.path.abspath(f'../../outputs/{username}/{k+1}')
        is_timing = k in timing_subset
        
        if os.path.exists(f"{out_dir}/multiple-mitotic-figure-classification.json") and not is_timing:
            continue

        os.makedirs(out_dir, exist_ok=True)
        os.system(f"chmod -R 777 {out_dir}")

        mount_script = f"-v {os.path.abspath('./code/inference_timed.py')}:/opt/app/inference.py" if is_timing else ""
        
        if is_timing:
            print(f"\n⏱️  Profiling Case {k+1}/238...")
            wait_for_idle_gpu(IDLE_THRESHOLD)
        else:
            print(f"🔄 Case {k+1}/238...")

        cmd = f"podman run --rm -v ../../inputs/{k+1}:/input:ro -v {out_dir}:/output:rw {mount_script} --device nvidia.com/gpu=all --shm-size 1024M {img_id}"

        start_time = time.time()
        peak_vram = run_container(cmd, is_timing) - IDLE_USAGE
        duration = time.time() - start_time

        if is_timing:
            print(f"📈 Result: Peak {peak_vram}MB | Time {duration:.2f}s")
            stats[k+1] = {'peak_mem': peak_vram, 'duration': duration}

    with open(f'../../outputs/{username}/inference_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\n🏁 Finished! Stats in ../../outputs/{username}/inference_stats.json")
