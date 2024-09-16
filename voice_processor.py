import os
import subprocess
from typing import List, Union, Optional
def open_slice(
    inp: str,
    opt_root: str,
    threshold: float,
    min_length: float,
    min_interval: float,
    hop_size: float,
    max_sil_kept: float,
    _max: float,
    alpha: float,
    n_parts: int,
    python_exec: str
) -> str:
    inp = clean_path(inp)
    opt_root = clean_path(opt_root)
    
    if not os.path.exists(inp):
        return "输入路径不存在"
    
    if os.path.isfile(inp):
        n_parts = 1
    elif not os.path.isdir(inp):
        return "输入路径存在但既不是文件也不是文件夹"
    
    ps_slice: List[subprocess.Popen] = []
    
    for i_part in range(n_parts):
        cmd = f'"{python_exec}" tools/slice_audio.py "{inp}" "{opt_root}" {threshold} {min_length} {min_interval} {hop_size} {max_sil_kept} {_max} {alpha} {i_part} {n_parts}'
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        ps_slice.append(p)
    
    for p in ps_slice:
        p.wait()
    
    return "切割结束"

def clean_path(path: str) -> str:
    # Implement the clean_path function from my_utils here
    # This is a placeholder implementation
    return os.path.normpath(path)


def open_denoise(
    denoise_inp_dir: str,
    denoise_opt_dir: str,
    python_exec: str,
    is_half: bool
) -> str:
    denoise_inp_dir = clean_path(denoise_inp_dir)
    denoise_opt_dir = clean_path(denoise_opt_dir)
    
    if not os.path.exists(denoise_inp_dir):
        return "输入目录不存在"
    
    precision = "float16" if is_half else "float32"
    cmd = f'"{python_exec}" tools/cmd-denoise.py -i "{denoise_inp_dir}" -o "{denoise_opt_dir}" -p {precision}'
    
    print(f"语音降噪任务开启：{cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        return f"语音降噪任务完成，输出目录：{denoise_opt_dir}"
    except subprocess.CalledProcessError:
        return "语音降噪任务失败"

def clean_path(path: str) -> str:
    # 实现 clean_path 函数，这里使用一个简单的实现
    return os.path.normpath(path)

def extract_semantic_tokens(
    inp_text: str,
    exp_name: str,
    gpu_numbers: str,
    pretrained_s2G_path: str,
    exp_root: str,
    python_exec: str,
    is_half: bool
) -> str:
    try:
        inp_text = clean_path(inp_text)
        
        if not os.path.exists(inp_text):
            raise FileNotFoundError(f"输入文件不存在: {inp_text}")
        
        if not os.path.exists(pretrained_s2G_path):
            raise FileNotFoundError(f"预训练模型文件不存在: {pretrained_s2G_path}")
        
        opt_dir = os.path.join(exp_root, exp_name)
        os.makedirs(opt_dir, exist_ok=True)
        
        config = {
            "inp_text": inp_text,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "pretrained_s2G": pretrained_s2G_path,
            "s2config_path": "GPT_SoVITS/configs/s2.json",
            "is_half": str(is_half)
        }
        
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        processes: List[subprocess.Popen] = []
        
        for i_part in range(all_parts):
            part_config = config.copy()
            part_config.update({
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
            })
            
            env = os.environ.copy()
            env.update(part_config)
            
            cmd = f'"{python_exec}" GPT_SoVITS/prepare_datasets/3-get-semantic.py'
            print(f"执行命令: {cmd}")
            p = subprocess.Popen(cmd, shell=True, env=env)
            processes.append(p)
        
        for p in processes:
            p.wait()
        
        # Combine results
        opt = ["item_name\tsemantic_audio"]
        path_semantic = os.path.join(opt_dir, "6-name2semantic.tsv")
        for i_part in range(all_parts):
            semantic_path = os.path.join(opt_dir, f"6-name2semantic-{i_part}.tsv")
            if not os.path.exists(semantic_path):
                raise FileNotFoundError(f"未找到预期的输出文件: {semantic_path}")
            with open(semantic_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(semantic_path)
        
        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        
        return "语义token提取进程成功完成"
    
    except FileNotFoundError as e:
        return f"错误: {str(e)}"
    except subprocess.CalledProcessError as e:
        return f"命令执行失败: {e.cmd}. 返回码: {e.returncode}"
    except Exception as e:
        return f"发生未预期的错误: {str(e)}"

def clean_path(path: str) -> str:
    # Implement the clean_path function from my_utils here
    return os.path.normpath(path)

def fix_gpu_number(gpu_number: str) -> str:
    # Implement the fix_gpu_number function here
    # This is a placeholder implementation
    return gpu_number
