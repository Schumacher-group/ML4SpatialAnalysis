#!/usr/bin/env python3
import yaml
import subprocess
import copy
from pathlib import Path

# 1) Paths
BASE_CFG = Path("configs/config.yaml")
LOG_DIR  = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# 2) Load base config
with BASE_CFG.open() as f:
    base_cfg = yaml.safe_load(f)

# 3) Parameter grids
seeds         = list(range(42, 51))
gmethods      = ["knn", "atmostk", "delaunay"]
node_features = ["expressions", "celltypes", "expressions_celltypes"]

# 4) Loop & launch
for seed in seeds:
    for gmethod in gmethods:
        for node_f in node_features:
            # 4a) Make a fresh copy
            cfg = copy.deepcopy(base_cfg)
            # 4b) Overwrite the three fields
            cfg["seed"] = seed
            cfg.setdefault("model", {})\
               .setdefault("gnn", {})["gmethod"] = gmethod
            cfg["model"]["gnn"]["node_f"] = node_f

            # 4c) Dump to a per‐run YAML
            tag = f"s{seed}_{gmethod}_{node_f}"
            cfg_path = LOG_DIR / f"cfg_{tag}.yaml"
            with cfg_path.open("w") as out:
                yaml.safe_dump(cfg, out)

            # 4d) Run main.py with that config, catching errors
            log_path = LOG_DIR / f"run_{tag}.log"
            print(f"=> Running {tag}   (cfg → {cfg_path}, log → {log_path})")
            with log_path.open("w") as logf:
                try:
                    subprocess.run(
                        ["python", "main.py", "--config", str(cfg_path)],
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        check=True
                    )
                except subprocess.CalledProcessError as e:
                    print(f"!! Run {tag} failed (exit {e.returncode}). See {log_path} for details.")

# all done
print("All configurations have been processed.")
