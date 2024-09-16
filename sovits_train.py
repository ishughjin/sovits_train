import os
import json
import subprocess

def train_sovits(
    exp_name,
    batch_size=1,
    total_epochs=1,
    text_low_lr_rate=0.4,
    save_every_epoch=1,
    save_only_latest=True,
    save_small_model=False,
    gpu_numbers="0",
    pretrained_s2G="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    pretrained_s2D="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth",
    pretrained_gpt="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    version="v2",
    exp_root="path/to/experiment/root",
    python_exec="python"
):
    # Create experiment directory
    s2_dir = os.path.join(exp_root, exp_name)
    os.makedirs(os.path.join(s2_dir, "logs_s2"), exist_ok=True)

    # Load and modify configuration
    with open("GPT_SoVITS/configs/s2.json") as f:
        config = json.load(f)

    config["train"].update({
        "batch_size": batch_size,
        "epochs": total_epochs,
        "text_low_lr_rate": text_low_lr_rate,
        "pretrained_s2G": pretrained_s2G,
        "pretrained_s2D": pretrained_s2D,
        "if_save_latest": save_only_latest,
        "if_save_every_weights": save_small_model,
        "save_every_epoch": save_every_epoch,
        "gpu_numbers": gpu_numbers
    })
    
    config["model"]["version"] = version
    config["data"]["exp_dir"] = s2_dir
    config["s2_ckpt_dir"] = s2_dir
    config["name"] = exp_name
    config["version"] = version

    # Write temporary configuration file
    tmp_config_path = os.path.join("tmp", f"{exp_name}_s2.json")
    with open(tmp_config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Prepare and execute training command
    cmd = f'"{python_exec}" GPT_SoVITS/s2_train.py --config "{tmp_config_path}"'
    print(f"Starting SoVITS training: {cmd}")

    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    print("SoVITS training completed")