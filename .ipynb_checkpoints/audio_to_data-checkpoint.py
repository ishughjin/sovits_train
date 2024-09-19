import os
import traceback
from subprocess import Popen

from tools import my_utils
def check_for_existence(paths):
    """
    Check if all the provided paths exist.
    
    Args:
    paths (list): A list of file or directory paths to check.
    
    Returns:
    bool: True if all paths exist, False otherwise.
    """
    for path in paths:
        if not os.path.exists(path):
            print(f"Error: Path does not exist: {path}")
            return False
    return True
def process_audio_data(
    audio_file_list,
    audio_folder_path,
    experiment_name,
    gpu_numbers="0-0",
    bert_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    ssl_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-hubert-base",
    pretrained_s2G_path="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    pretrained_s2D_path="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth",
    pretrained_gpt_path="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    version="v2",
    gpu_info="0"
):
    global processes
    processes = []

    audio_file_list = my_utils.clean_path(audio_file_list)
    audio_folder_path = my_utils.clean_path(audio_folder_path)

    if not check_for_existence([audio_file_list, audio_folder_path]):
        return "Error: Input files do not exist"

    exp_root = "../temp/log/"  # Define this appropriately
    opt_dir = f"{exp_root}/{experiment_name}"

    try:
        # Step 1: Process text
        text_output_path = f"{opt_dir}/2-name2text.txt"  # Changed from 2-name2text-0.txt
        if not os.path.exists(text_output_path) or os.path.getsize(text_output_path) < 2:
            config = {
                "inp_text": audio_file_list,
                "inp_wav_dir": audio_folder_path,
                "exp_name": experiment_name,
                "opt_dir": opt_dir,
                "bert_pretrained_dir": bert_pretrained_dir,
                "is_half": "True",  # Adjust as needed
                "version": version,
                "gpu_info": gpu_info
            }
            _run_subprocess(config, gpu_numbers, "GPT_SoVITS/prepare_datasets/1-get-text.py")

        # Step 2: Process audio
        config = {
            "inp_text": audio_file_list,
            "inp_wav_dir": audio_folder_path,
            "exp_name": experiment_name,
            "opt_dir": opt_dir,
            "cnhubert_base_dir": ssl_pretrained_dir,
            "version": version,
            "gpu_info": gpu_info
        }
        _run_subprocess(config, gpu_numbers, "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py")

        # Step 3: Generate semantics
        semantic_output_path = f"{opt_dir}/6-name2semantic.tsv"  # Changed from 6-name2semantic-0.tsv
        if not os.path.exists(semantic_output_path) or os.path.getsize(semantic_output_path) < 31:
            config = {
                "inp_text": audio_file_list,
                "exp_name": experiment_name,
                "opt_dir": opt_dir,
                "pretrained_s2G": pretrained_s2G_path,
                "pretrained_s2D": pretrained_s2D_path,
                "pretrained_gpt": pretrained_gpt_path,
                "s2config_path": "GPT_SoVITS/configs/s2.json",
                "version": version,
                "gpu_info": gpu_info
            }
            _run_subprocess(config, gpu_numbers, "GPT_SoVITS/prepare_datasets/3-get-semantic.py")
        old_text_path = f"{opt_dir}/2-name2text-0.txt"
        new_text_path = f"{opt_dir}/2-name2text.txt"
        old_semantic_path = f"{opt_dir}/6-name2semantic-0.tsv"
        new_semantic_path = f"{opt_dir}/6-name2semantic.tsv"

        # Rename text file
        if os.path.exists(old_text_path):
            os.rename(old_text_path, new_text_path)
            print(f"Renamed {old_text_path} to {new_text_path}")
        else:
            print(f"Warning: {old_text_path} not found, skipping rename")

        # Rename semantic file
        if os.path.exists(old_semantic_path):
            os.rename(old_semantic_path, new_semantic_path)
            print(f"Renamed {old_semantic_path} to {new_semantic_path}")
        else:
            print(f"Warning: {old_semantic_path} not found, skipping rename")

        return "Audio processing completed successfully, files renamed as requested"


    except Exception as e:
        traceback.print_exc()
        _terminate_processes()
        return f"Error occurred during processing: {str(e)}"
def _run_subprocess(config, gpu_numbers, script_path):
    gpu_list = gpu_numbers.split("-")
    for i, gpu in enumerate(gpu_list):
        config.update({
            "i_part": str(i),
            "all_parts": str(len(gpu_list)),
            "_CUDA_VISIBLE_DEVICES": gpu,
        })
        os.environ.update(config)
        cmd = f'python {script_path}'
        print(f"Running command: {cmd}")
        process = Popen(cmd, shell=True)
        processes.append(process)
    
    for process in processes:
        process.wait()

def _terminate_processes():
    for process in processes:
        if process.poll() is None:
            process.terminate()
    processes.clear()