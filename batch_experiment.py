#!/usr/bin/env python3
"""Run multiple experiments described by a metaconfig JSONC file.

The metaconfig should list `configs` (list of config filenames or paths)
and a set of seed lists. Each seed-list length defines how many runs to do
per config. Example:

{
  "configs": ["debug.jsonc", "example_config.jsonc"],
  "training": [10, 12, 22],
  "dataset": [2, 3, 7],
  "partition": [92, 23, 4]
}

This will run each config 3 times (one per seed-group).

Logs are written to `logs/<metaconfig>_<timestamp>.txt` and discord
notifications are posted to the webhook url stored in `hook.txt`.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Dict
import urllib.request


SEED_KEYS = [
    "global",
    "dataset",
    "partition",
    "loader",
    "model",
    "training",
    "pretrain",
    "baseline",
    "replay",
]


def read_jsonc(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = re.sub(r"//.*", "", raw)
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    return json.loads(raw)


def send_discord(hook_url: str, message: str):
    if not hook_url:
        return
    payload = json.dumps({"content": message}).encode("utf-8")
    try:
        req = urllib.request.Request(hook_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read()
    except Exception as e:
        # ignore discord errors but return None
        return None


def resolve_config_path(entry: str, configs_dir: str):
    if os.path.isabs(entry) and os.path.exists(entry):
        return entry
    # try relative to workspace
    if os.path.exists(entry):
        return entry
    # try inside configs_dir
    candidate = os.path.join(configs_dir, entry)
    if os.path.exists(candidate):
        return candidate
    # otherwise return candidate (it may fail later)
    return candidate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metaconfig", required=True, help="Path to metaconfig JSONC file (meta_configs/...)")
    p.add_argument("--configs-dir", default="configs", help="Folder for config files (default: configs)")
    p.add_argument("--dil-script", default="DIL_Experiment.py", help="Path to DIL_Experiment.py script to execute")
    p.add_argument("--hook-file", default="hook.txt", help="File containing Discord webhook URL")
    p.add_argument("--logs-dir", default="logs", help="Directory where run logs are stored")
    args = p.parse_args()

    meta = read_jsonc(args.metaconfig)

    configs = meta.get("configs")
    if configs is None or not isinstance(configs, list) or len(configs) == 0:
        raise SystemExit("metaconfig must contain a non-empty 'configs' list")

    # gather seed lists either under a 'seeds' dict or as top-level lists
    seeds_map: Dict[str, list] = {}
    if "seeds" in meta and isinstance(meta["seeds"], dict):
        seeds_map = meta["seeds"]
    else:
        for k in SEED_KEYS:
            if k in meta and isinstance(meta[k], list):
                seeds_map[k] = meta[k]

    num_runs = 1
    if seeds_map:
        lengths = {k: len(v) for k, v in seeds_map.items()}
        lens = set(lengths.values())
        if len(lens) != 1:
            raise SystemExit(f"Seed lists have mismatched lengths: {lengths}")
        num_runs = next(iter(lens))

    total_runs = num_runs * len(configs)

    os.makedirs(args.logs_dir, exist_ok=True)
    meta_base = os.path.splitext(os.path.basename(args.metaconfig))[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.logs_dir, f"{meta_base}_{stamp}.txt")

    # read hook url
    hook_url = None
    if os.path.exists(args.hook_file):
        try:
            with open(args.hook_file, "r", encoding="utf-8") as f:
                hook_url = f.read().strip()
        except Exception:
            hook_url = None

    runs_done = 0
    failures = 0

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"Meta-config run started: {args.metaconfig} at {datetime.now().isoformat()}\n")
        logf.write(f"Total tests: {total_runs} (configs={len(configs)}, runs_per_config={num_runs})\n")
        logf.flush()

        for run_i in range(num_runs):
            for cfg_entry in configs:
                cfg_path = resolve_config_path(cfg_entry, args.configs_dir)
                runs_done += 1
                logf.write(f"\nSTART RUN {runs_done}/{total_runs} - config={cfg_path} - seed_group={run_i} - time={datetime.now().isoformat()}\n")
                print(f"START RUN {runs_done}/{total_runs} - config={cfg_path}")

                # assemble command
                cmd = [sys.executable, args.dil_script, "--config", cfg_path]
                # add per-seed flags
                for k, lst in seeds_map.items():
                    val = lst[run_i]
                    # skip None entries
                    if val is None:
                        cmd += [f"--{k}-seed", "random"]
                    else:
                        cmd += [f"--{k}-seed", str(val)]

                # run by importing and calling the DIL_Experiment module directly
                try:
                    # ensure workspace root is importable
                    if os.getcwd() not in sys.path:
                        sys.path.insert(0, os.getcwd())
                    import importlib
                    import io
                    import contextlib

                    mod = importlib.import_module(os.path.splitext(os.path.basename(args.dil_script))[0])
                    importlib.reload(mod)

                    # prepare argv for the target main()
                    saved_argv = sys.argv
                    saved_stdout = sys.stdout
                    saved_stderr = sys.stderr
                    buf_out = io.StringIO()
                    buf_err = io.StringIO()
                    seed_flags = []
                    for k, lst in seeds_map.items():
                        val = lst[run_i]
                        if val is None:
                            seed_flags += [f"--{k}-seed", "random"]
                        else:
                            seed_flags += [f"--{k}-seed", str(val)]
                    sys.argv = [args.dil_script, "--config", cfg_path] + seed_flags
                    try:
                        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                            try:
                                mod.main()
                                retcode = 0
                            except SystemExit as se:
                                # argparse may call SystemExit; treat non-zero as failure
                                retcode = se.code if isinstance(se.code, int) else 1
                            except Exception as ex:
                                retcode = 2
                                # re-raise after capturing
                                raise
                    finally:
                        sys.argv = saved_argv
                        sys.stdout = saved_stdout
                        sys.stderr = saved_stderr

                    stdout_val = buf_out.getvalue()
                    stderr_val = buf_err.getvalue()
                    logf.write(f"STDOUT:\n{stdout_val}\n")
                    logf.write(f"STDERR:\n{stderr_val}\n")
                    if retcode != 0:
                        failures += 1
                        logf.write(f"RUN FAILED at {datetime.now().isoformat()} returncode={retcode}\n")
                        logf.flush()
                        msg = f"Metaconfig {args.metaconfig} - run {runs_done}/{total_runs} failed on config {cfg_path}. Return code {retcode}. See logs."
                        send_discord(hook_url, msg)
                    else:
                        logf.write(f"RUN COMPLETED at {datetime.now().isoformat()}\n")
                        logf.flush()
                except Exception as e:
                    failures += 1
                    logf.write(f"EXCEPTION during run at {datetime.now().isoformat()}: {e}\n")
                    logf.flush()
                    msg = f"Metaconfig {args.metaconfig} - exception during run {runs_done}/{total_runs} on config {cfg_path}: {e}"
                    send_discord(hook_url, msg)

                # progress update
                frac = runs_done / total_runs
                print(f"Progress: {runs_done}/{total_runs} ({frac:.2%})")
                logf.write(f"Progress: {runs_done}/{total_runs} ({frac:.2%})\n")
                logf.flush()

        # finished all runs
        summary = f"Metaconfig finished at {datetime.now().isoformat()}: completed={runs_done}, failures={failures}\n"
        logf.write(summary)
        logf.flush()

    # final discord notification
    if failures == 0:
        send_discord(hook_url, f"Metaconfig {args.metaconfig} finished successfully. {runs_done} runs completed.")
    else:
        send_discord(hook_url, f"Metaconfig {args.metaconfig} finished with {failures} failures out of {runs_done} runs.")

    print(summary)


if __name__ == '__main__':
    main()
