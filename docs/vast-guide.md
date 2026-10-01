# Vast.ai: not paying for idle time

## What you pay for

| Instance state | GPU + CPU (the hourly price) | Disk storage |
|---|---|---|
| Running | charged, even if idle | charged |
| Stopped | **not charged** | charged |
| Destroyed | not charged | **not charged**, disk is wiped |

The hourly price in the listing already bundles the GPU, CPU and RAM; there is no
separate CPU meter to turn off. Some hosts also charge per GB downloaded or uploaded
(shown in the listing).

**Catch with Stop:** a stopped instance can only restart on the *same* machine. If
someone else rents that GPU in the meantime, you cannot restart until it is free again,
and your files are stuck on its disk. So get results off the machine *before* stopping,
and treat Stop as a pause for the same day, not as storage.

## One-time setup on each new instance

So the script can push results without a password prompt:

1. On GitHub: Settings → Developer settings → Fine-grained tokens → generate a token
   with access to **only** `sycophancy-dpo`, permission *Contents: Read and write*,
   and a short expiry (e.g. 30 days).
2. On the instance (type the token yourself; don't paste it into chats or files in the repo):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   git config --global credential.helper store
   git push   # enter your GitHub username and the token as the password, once
   ```
   The token is saved on the instance disk and is wiped when you destroy it.

## Automatic: run, push, stop

Instead of `python run.py ...`, run the same arguments through the script:

```bash
bash scripts/run_then_stop.sh stage=evaluate eval.split=test
```

When `run.py` finishes (or crashes), the script commits everything new under
`artifacts/`, pushes it, and stops the instance using the instance's own restricted
API key (`$CONTAINER_API_KEY`, preinstalled by Vast, can only control this instance).
You can detach (Ctrl+B, D), close the laptop and go to bed.

Model checkpoints (`artifacts/checkpoints/`) are not committed. For training runs, copy
them off before destroying (see below), or push the adapter to the Hugging Face Hub.

## Afterwards (manual)

1. Check the results arrived: on the laptop, `git pull`.
2. If you need files that are not in git (checkpoints), start the instance again from the
   console, then from the laptop:
   ```bash
   scp -P <port> -r root@<host>:/workspace/sycophancy-dpo/artifacts/checkpoints ./artifacts/
   ```
3. **Destroy** the instance in the console (Instances → trash icon) when you have
   everything. Stopped instances keep billing storage indefinitely.

## Manual stop/destroy

- Web console: Instances → **Stop** (pause) or the trash icon (**Destroy**).
- From the laptop with the CLI (`pip install vastai`, `vastai set api-key <key>` once):
  ```bash
  vastai show instances
  vastai stop instance <ID>
  vastai destroy instance <ID>
  ```

## Safety nets

- Keep autobilling off and only a few dollars of credit, so a forgotten instance
  cannot run up a large bill.
- Once a week, check Instances in the console for anything stopped but not destroyed.
