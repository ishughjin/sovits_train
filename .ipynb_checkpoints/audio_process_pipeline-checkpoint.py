import os
import subprocess
from typing import List, Union, Optional

from train.voice_processor import extract_semantic_tokens, open_denoise, open_slice

def audio_processing_pipeline(
    input_path: str,
    output_root: str,
    exp_name: str,
    python_exec: str,
    is_half: bool
) -> str:
    # Fixed parameters
    threshold = -34
    min_length = 4000
    min_interval = 300
    hop_size = 300
    max_sil_kept = 500
    _max = 0.9
    alpha = 0.35
    n_parts = 4
    gpu_numbers = "0"
    pretrained_s2G_path = "../GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    # Step 1: Audio Slicing
    slice_output = open_slice(
        inp=input_path,
        opt_root=output_root,
        threshold=threshold,
        min_length=min_length,
        min_interval=min_interval,
        hop_size=hop_size,
        max_sil_kept=max_sil_kept,
        _max=_max,
        alpha=alpha,
        n_parts=n_parts,
        python_exec=python_exec
    )
    
    if slice_output != "切割结束":
        return f"Audio slicing failed: {slice_output}"

    # Step 2: Denoising
    denoise_input_dir = os.path.join(output_root, "0_gt_wavs")
    denoise_output_dir = os.path.join(output_root, "1_denoise_wavs")
    denoise_output = open_denoise(
        denoise_inp_dir=denoise_input_dir,
        denoise_opt_dir=denoise_output_dir,
        python_exec=python_exec,
        is_half=is_half
    )
    
    if not denoise_output.startswith("语音降噪任务完成"):
        return f"Denoising failed: {denoise_output}"

    # Step 3: Semantic Token Extraction
    text_labelling_file = os.path.join(output_root, "2_txt_label.txt")
    semantic_output = extract_semantic_tokens(
        inp_text=text_labelling_file,
        exp_name=exp_name,
        gpu_numbers=gpu_numbers,
        pretrained_s2G_path=pretrained_s2G_path,
        exp_root=output_root,
        python_exec=python_exec,
        is_half=is_half
    )
    
    if semantic_output != "语义token提取进程成功完成":
        return f"Semantic token extraction failed: {semantic_output}"

    # Return the paths to the text labelling file and denoised audio directory
    print( f"Processing complete. Text labelling file: {text_labelling_file}, Denoised audio directory: {denoise_output_dir}")
    return f"Processing complete. Text labelling file: {text_labelling_file}, Denoised audio directory: {denoise_output_dir}"

# Helper functions (unchanged from the original code)
def clean_path(path: str) -> str:
    return os.path.normpath(path)

def fix_gpu_number(gpu_number: str) -> str:
    return gpu_number
