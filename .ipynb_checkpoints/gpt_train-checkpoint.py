import os
import yaml
import subprocess

def train_gpt(
    exp_name,
    batch_size=6,
    total_epochs=4,
    if_dpo=False,
    save_only_latest=True,
    save_small_model=True,
    save_every_epoch=4,
    gpu_numbers="0",
    pretrained_gpt="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    version="v2",
    exp_root="../autodl-tmp/log",
    python_exec="python",
    is_half=True
):
    GPT_weight_root=["../autodl-tmp","GPT_weights"]
    # Load configuration
    config_file = "GPT_SoVITS/configs/s1longer-v2.yaml" if version == "v2" else "GPT_SoVITS/configs/s1longer.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Set up directories
    s1_dir = os.path.join(exp_root, exp_name)
    os.makedirs(os.path.join(s1_dir, "logs_s1"), exist_ok=True)

    # Update configuration
    if not is_half:
        config["train"]["precision"] = "32"
        batch_size = max(1, batch_size // 2)

    config["train"].update({
        "batch_size": batch_size,
        "epochs": total_epochs,
        "save_every_n_epoch": save_every_epoch,
        "if_save_every_weights": save_small_model,
        "if_save_latest": save_only_latest,
        "if_dpo": if_dpo,
        "exp_name": exp_name,
        "half_weights_save_dir": "../autodl-tmp/" +exp_name
    })
    
    config["pretrained_s1"] = pretrained_gpt
    config["train_semantic_path"] = os.path.join(s1_dir, "6-name2semantic.tsv")
    config["train_phoneme_path"] = os.path.join(s1_dir, "2-name2text.txt")
    config["output_dir"] = os.path.join(s1_dir, "logs_s1")

    # Set environment variables
    os.environ["_CUDA_VISIBLE_DEVICES"] = gpu_numbers.replace("-", ",")
    os.environ["hz"] = "25hz"

    # Write temporary configuration file
    tmp_config_path = os.path.join("tmp", f"{exp_name}_s1.yaml")
    with open(tmp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Prepare and execute training command
    cmd = f'"{python_exec}" GPT_SoVITS/s1_train.py --config_file "{tmp_config_path}"'
    print(f"Starting GPT training: {cmd}")

    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    print(f"Main output directory: {config['output_dir']}")
    print(f"Checkpoint directory: {os.path.join(config['output_dir'], 'ckpt')}")
    if save_small_model:
        print(f"Model weights directory: {config['train']['half_weights_save_dir']}")
    print(f"Training semantic file: {config['train_semantic_path']}")
    print(f"Training phoneme file: {config['train_phoneme_path']}")
    print(f"Temporary config file: {tmp_config_path}")
    print("GPT training completed")