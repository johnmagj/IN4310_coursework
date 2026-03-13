# UiO ML Nodes: End-to-End Workflow (Connection → Upload Code → Run Training)

This repository is a **cluster-running template**. It does **not** explain how to solve any assignment tasks.  
It only explains **how to connect to UiO ML nodes, move your code, access the course dataset path, and run training**.

Official documentation (read this first):
- https://www.uio.no/tjenester/it/forskning/kompetansehuber/uio-ai-hub-node-project/it-resources/ml-nodes/#toc2

---

## Table of Contents

1. What you need on your laptop
2. Connect to UiO (Jump Host) and then to an ML node
3. Pick a free GPU (single GPU, no DDP)
4. Load Python via `module` and activate your virtual environment
5. Upload your project folder (recommended workflows)
6. Run training (foreground and background)
7. Monitor logs and stop jobs
8. Access the course dataset on the ML nodes (paths + notes)
9. Common pitfalls and fixes
10. Quick command checklist

---

## 1) What you need on your laptop

### Windows
- PowerShell (built-in)
- OpenSSH client (usually installed; check with `ssh -V`)
- Microsoft Authenticator for UiO 2FA

Optional but helpful:
- Git Bash / WSL for `rsync` (faster incremental uploads).  
  If you do not have it, you can still use `sftp` (works well with UiO 2FA).

---

## 2) Connect to UiO (Jump Host) → then to an ML node

UiO ML nodes live in a protected network, so you first connect to the login host.

### Step A: Connect to `login.uio.no` (Jump Host)

From **Windows PowerShell**:

```bash
ssh -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive <USERNAME>@login.uio.no
```

You will be prompted for:
- the UiO password
- and a Microsoft Authenticator confirmation / OTP

After login you should see a UiO login host (e.g. `gothmog`).

### Step B: Connect from login host to an ML node

On the login host:

```bash
ssh ml1.hpc.uio.no
```

**Important:** do not run heavy computations on the login host. Only use it for file management and as a jump host.

---

## 3) Pick a free GPU (single GPU, no DDP)

On the ML node (e.g. `ml1`), check GPU status:

```bash
nvidia-smi
```

Look for a GPU with:
- low `GPU-Util`
- low `Memory-Usage`
- few/no running processes

### Reserve one GPU for your process (single GPU)

If GPU #2 is free:

```bash
export CUDA_VISIBLE_DEVICES=2
```

**Why this works:** your Python process will only "see" the GPU(s) listed in `CUDA_VISIBLE_DEVICES`.  
This is the simplest way to use **one GPU** without DDP.

Verify in Python:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('visible gpus:', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

If `visible gpus: 1`, your masking works.

---

## 4) Load Python via modules + activate your virtual environment

On UiO ML nodes, software is managed using **modules**.

Example (matches this template):

```bash
module purge
module load Python/3.10.4-GCCcore-11.3.0
```

If you do not know what versions exist:

```bash
module spider Python
```

### Create / activate a virtual environment

Create (one-time):

```bash
python3 -m venv ~/my_env
```

Activate (every session):

```bash
source ~/my_env/bin/activate
```

Upgrade pip (recommended):

```bash
python -m pip install --upgrade pip
```

Install dependencies for this template:

```bash
pip install -r requirements.txt
```

---

## 5) Upload your project folder to UiO

### Folder layout in this template

```
Core/
  requirements.txt
  src/
    train.py
    utils.py
  scripts/
    run_mlnode.sh
    sync_to_cluster.sh   (example only)
```

You need to copy the **entire folder** `Core/` to your UiO home directory.

### Recommended workflow for UiO 2FA: use SFTP to the login host

Some `scp` workflows can be unreliable with 2FA prompts. `sftp` is often more robust.

On Windows PowerShell, go to the directory **that contains** the `Core` folder (e.g. Desktop):

```powershell
cd C:\Users\<YOU>\Desktop
sftp -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive <USERNAME>@login.uio.no
```

Inside the `sftp>` prompt:

```text
mkdir projects
mkdir projects/Core
cd projects/Core
put -r Core/* .
bye
```

Now your code is on the login host at:

```
~/projects/Core/
```

### Copy from login host to ML node (fast, internal network)

On the login host:

```bash
ssh ml1.hpc.uio.no "mkdir -p ~/projects/Core"
rsync -av --delete ~/projects/Core/ <USERNAME>@ml1.hpc.uio.no:~/projects/Core/
```

**Why two-step upload is common:**
- Windows → login host (2FA)
- login host → ML node (fast internal transfer)

Official UiO file transfer page:
- https://www.uio.no/tjenester/it/forskning/kompetansehuber/uio-ai-hub-node-project/it-resources/ml-nodes/file-transfer.html

---

## 6) Run training

### Option A: Run in the foreground (simple; stops if SSH disconnects)

On the ML node:

```bash
cd ~/projects/Core
export CUDA_VISIBLE_DEVICES=2
source ~/my_env/bin/activate
python src/train.py --epochs 5 --batch-size 64 --lr 1e-3 --outdir runs/exp0
```

### Option B: Run in the background with logs (recommended)

This is what `scripts/run_mlnode.sh` does. You can run it directly:

```bash
cd ~/projects/Core
bash scripts/run_mlnode.sh
```

It will:
- select a GPU via `CUDA_VISIBLE_DEVICES=...`
- load modules
- activate your venv
- run training with `nohup ... &`
- write logs to `logs/out.log` and `logs/err.log`

You can also launch manually:

```bash
mkdir -p logs runs/exp0
nohup python -u src/train.py   --epochs 5   --batch-size 64   --lr 1e-3   --outdir runs/exp0   > logs/out.log 2> logs/err.log &
```

`-u` makes Python print output immediately (useful for `tail -f`).

---

## 7) Monitor logs and stop jobs

### Watch logs
```bash
tail -f logs/out.log
tail -f logs/err.log
```

### Check your running processes
```bash
ps -u $USER | grep -i python
```

### Stop a job
Find the PID from `ps`, then:

```bash
kill -9 <PID>
```

### Confirm GPU usage
```bash
nvidia-smi
```

You should see your Python process listed under the chosen GPU.

---

## 8) Access the course dataset on the ML nodes (paths + notes)

This section only explains **where the data is** and how to reference it in your code.  
It does not explain how to solve the assignment.

### Dataset location (already on the ML nodes)
The images are stored here:

```text
/itf-fi-ml/shared/courses/IN3310/mandatory1_data
```

Inside that directory, the **subdirectories correspond to class labels**.  
So your code should treat each subdirectory as a class folder.

### If you need the ZIP (copying off the ML nodes)
A ZIP archive is available here:

```text
/itf-fi-ml/shared/courses/IN3310/mandatory1_data.zip
```

Use this ZIP if you need to copy the dataset elsewhere (e.g., to your own machine).

### How to reference the dataset path in Python (example)
Hard-code the root path as a CLI argument or config value:

```bash
python src/train.py --data-root /itf-fi-ml/shared/courses/IN3310/mandatory1_data
```

Inside Python, you would read from that folder using `pathlib.Path` or standard OS tools.

---

## 9) Common pitfalls and fixes

### A) “Too many authentication failures” (Windows)
Use:
- `-o PubkeyAuthentication=no`
so SSH does not try many local keys before asking for password/2FA.

### B) DataLoader worker crashes
If you see errors like “DataLoader worker exited unexpectedly”:
- set `num_workers=0` temporarily to see the real error
- check file paths carefully

### C) `persistent_workers=True` error
`persistent_workers=True` requires `num_workers > 0`.

### D) GPU still busy / wrong GPU used
- always set `export CUDA_VISIBLE_DEVICES=<id>` **before** running Python
- re-check with `nvidia-smi`
- verify with `torch.cuda.device_count()`

### E) Connection drops during training
Use `nohup ... &` and write logs to files (as shown above).

---

## 10) Quick command checklist

### Connect
```bash
ssh -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive <USERNAME>@login.uio.no
ssh ml1.hpc.uio.no
```

### Choose GPU
```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=2
```

### Environment
```bash
module purge
module load Python/3.10.4-GCCcore-11.3.0
source ~/my_env/bin/activate
pip install -r requirements.txt
```

### Run
```bash
cd ~/projects/Core
bash scripts/run_mlnode.sh
tail -f logs/out.log
```

### Stop
```bash
ps -u $USER | grep -i python
kill -9 <PID>
```

---

## References
- UiO ML nodes documentation: https://www.uio.no/tjenester/it/forskning/kompetansehuber/uio-ai-hub-node-project/it-resources/ml-nodes/#toc2
- UiO ML nodes file transfer: https://www.uio.no/tjenester/it/forskning/kompetansehuber/uio-ai-hub-node-project/it-resources/ml-nodes/file-transfer.html
