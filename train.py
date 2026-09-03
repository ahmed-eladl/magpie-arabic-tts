"""
Local / Standard Server Fine-Tuning Pipeline for Magpie-TTS.

Runs the complete fine-tuning lifecycle on any standard GPU environment (e.g., local server, RunPod, AWS)
assuming NVIDIA NeMo 3.0+ is installed (or running inside the `nvcr.io/nvidia/nemo-speech` Docker container).

Steps automated:
1. Downloads base Magpie-TTS multilingual model (357M) and official NeMo scripts.
2. Extracts checkpoint dictionaries to guarantee the 3,359-row text embedding table matches.
3. Patches Hydra YAML configuration and prepares training/validation manifests.
4. Executes NeMo training loop with gradient accumulation and FP32 precision.
"""

import argparse
import os
import subprocess
import tarfile
import urllib.request
import yaml


def setup_workspace_environment(workspace_dir: str):
    """
    Downloads the base Magpie model, extracts dictionary files,
    and configures magpietts.yaml for fine-tuning.
    """
    os.makedirs(workspace_dir, exist_ok=True)
    conf_dir = os.path.join(workspace_dir, "conf")
    os.makedirs(conf_dir, exist_ok=True)

    script_path = os.path.join(workspace_dir, "magpietts.py")
    conf_path = os.path.join(conf_dir, "magpietts.yaml")
    base_model_path = os.path.join(workspace_dir, "magpie_tts_multilingual_357m.nemo")

    # 1. Download official training script
    if not os.path.exists(script_path):
        print("⬇️ Downloading magpietts.py from NVIDIA/NeMo repository...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/tts/magpietts.py",
            script_path
        )

    # Patch Hydra allowed targets if running older NeMo
    with open(script_path, "r", encoding="utf-8") as f:
        script_code = f.read()
    patch = (
        "import nemo.core.classes.common\n"
        "if hasattr(nemo.core.classes.common, 'ALLOWED_TARGET_PREFIXES') and 'transformers.' not in nemo.core.classes.common.ALLOWED_TARGET_PREFIXES:\n"
        "    nemo.core.classes.common.ALLOWED_TARGET_PREFIXES.append('transformers.')\n"
    )
    if "ALLOWED_TARGET_PREFIXES" not in script_code:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(patch + script_code)

    # 2. Download training configuration
    if not os.path.exists(conf_path):
        print("⬇️ Downloading magpietts.yaml...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/tts/conf/magpietts/magpietts.yaml",
            conf_path
        )

    # 3. Download base Magpie-TTS checkpoint
    if not os.path.exists(base_model_path):
        print("⬇️ Downloading base checkpoint: nvidia/magpie_tts_multilingual_357m...")
        urllib.request.urlretrieve(
            "https://huggingface.co/nvidia/magpie_tts_multilingual_357m/resolve/main/magpie_tts_multilingual_357m.nemo",
            base_model_path
        )

    # 4. Extract model_config.yaml and dataset dictionaries from .nemo checkpoint
    dict_dir = os.path.join(workspace_dir, "scripts", "tts_dataset_files")
    os.makedirs(dict_dir, exist_ok=True)
    ckpt_config_path = os.path.join(workspace_dir, "model_config.yaml")

    if not os.path.exists(ckpt_config_path):
        print("📦 Extracting dictionary files and model config from .nemo archive...")
        with tarfile.open(base_model_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith("model_config.yaml"):
                    f = tar.extractfile(member)
                    with open(ckpt_config_path, "wb") as out:
                        out.write(f.read())
                elif member.name.endswith((".txt", ".dict")) or "heteronym" in member.name:
                    f = tar.extractfile(member)
                    out_name = os.path.join(dict_dir, os.path.basename(member.name))
                    with open(out_name, "wb") as out:
                        out.write(f.read())

    # 5. Patch train configuration with checkpoint tokenizer settings
    with open(ckpt_config_path, "r", encoding="utf-8") as f:
        ckpt_config = yaml.safe_load(f)
    with open(conf_path, "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)

    if "model" in train_config:
        print("⚙️ Aligning tokenizer vocabulary table (3,359 rows)...")
        def fix_nemo_paths(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and v.startswith("nemo:"):
                        obj[k] = f"scripts/tts_dataset_files/{v[5:]}"
                    else:
                        fix_nemo_paths(v)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    if isinstance(v, str) and v.startswith("nemo:"):
                        obj[i] = f"scripts/tts_dataset_files/{v[5:]}"
                    else:
                        fix_nemo_paths(v)

        fix_nemo_paths(ckpt_config)
        for k, v in ckpt_config.items():
            if k not in ["train_ds", "validation_ds", "optim"]:
                train_config["model"][k] = v

        with open(conf_path, "w", encoding="utf-8") as f:
            yaml.dump(train_config, f, default_flow_style=False, sort_keys=False)

    print("✅ Environment setup and config patch complete.\n")


def run_training(
    dialect: str = "saudi",
    batch_size: int = 2,
    max_epochs: int = 10,
    learning_rate: float = 5e-6,
    data_dir: str = "./data",
):
    """Executes the fine-tuning loop."""
    import torch
    if torch.cuda.is_available():
        print(f"✅ Active GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ WARNING: No CUDA GPU detected! Training will be extremely slow or fail.")

    setup_workspace_environment(data_dir)

    # Dialect configuration
    if dialect.lower() in ["saudi", "ar-sa"]:
        dataset_name = "ar_sa_sft"
        tokenizer_name = "arabic_SA_chartokenizer"
        exp_dir = "./magpie_saudi_output"
    else:
        dataset_name = "ar_ae_sft"
        tokenizer_name = "arabic_SA_chartokenizer"
        exp_dir = "./magpie_emirati_output"

    cmd = [
        "python", "magpietts.py",
        "--config-path=conf",
        "--config-name=magpietts",
        "+init_from_nemo_model=./magpie_tts_multilingual_357m.nemo",
        f"exp_manager.exp_dir={exp_dir}",
        f"+train_ds_meta.{dataset_name}.manifest_path=./train.json",
        f"+train_ds_meta.{dataset_name}.audio_dir=./",
        f"+train_ds_meta.{dataset_name}.feature_dir=./",
        f"+train_ds_meta.{dataset_name}.sample_weight=1.0",
        f"+train_ds_meta.{dataset_name}.tokenizer_names=[{tokenizer_name}]",
        f"+val_ds_meta.{dataset_name}_val.manifest_path=./val.json",
        f"+val_ds_meta.{dataset_name}_val.audio_dir=./",
        f"+val_ds_meta.{dataset_name}_val.feature_dir=./",
        f"+val_ds_meta.{dataset_name}_val.sample_weight=1.0",
        f"+val_ds_meta.{dataset_name}_val.tokenizer_names=[{tokenizer_name}]",
        "model.codecmodel_path=nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
        "model.context_duration_min=3.0",
        "model.context_duration_max=10.0",
        "model.alignment_loss_scale=0.0",
        "model.prior_scaling_factor=null",
        f"model.optim.lr={learning_rate}",
        "~model.optim.sched",
        "model.load_cached_codes_if_available=false",
        "trainer.precision=32",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        f"batch_size={batch_size}",
        "trainer.accumulate_grad_batches=2",
        f"max_epochs={max_epochs}"
    ]

    print("\n🚀 Executing NeMo Fine-Tuning...\n")
    print(" ".join(cmd))
    print("\n")

    proc = subprocess.Popen(
        cmd,
        cwd=data_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in iter(proc.stdout.readline, ''):
        print(line, end="")
    proc.stdout.close()
    rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"NeMo training exited with error code {rc}")

    print("✅ Fine-tuning finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard GPU fine-tuning for Magpie-TTS.")
    parser.add_argument("--dialect", type=str, default="saudi", choices=["saudi", "emirati"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory with train.json and audio")
    
    args = parser.parse_args()
    
    run_training(
        dialect=args.dialect,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        data_dir=args.data_dir
    )
