import os
import sys
import torch
import librosa
import soundfile as sf
from tools.uvr5.vr import AudioPre, AudioPreDeEcho
from tools.uvr5.mdxnet import MDXNetDereverb
from tools.uvr5.bsroformer import BsRoformer_Loader
def process_audio(model_name, input_file, output_vocal, output_instrumental, agg=10, format='flac'):
    weight_uvr5_root = "./tools/uvr5/uvr5_weights"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_half = True if device == "cuda" else False

    is_hp3 = "HP3" in model_name

    if model_name == "onnx_dereverb_By_FoxJoy":
        pre_fun = MDXNetDereverb(15)
    elif model_name == "Bs_Roformer" or "bs_roformer" in model_name.lower():
        pre_fun = BsRoformer_Loader(
            model_path=os.path.join(weight_uvr5_root, model_name + ".ckpt"),
            device=device,
            is_half=is_half
        )
    else:
        func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
        pre_fun = func(
            agg=int(agg),
            model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
            device=device,
            is_half=is_half,
        )

    try:
        pre_fun._path_audio_(input_file, output_instrumental, output_vocal, format, is_hp3)
        print(f"Successfully processed {input_file}")
        print(f"Vocal output: {output_vocal}")
        print(f"Instrumental output: {output_instrumental}")
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <model_name> <input_file> <output_vocal> <output_instrumental>")
        sys.exit(1)

    model_name = sys.argv[1]
    input_file = sys.argv[2]
    output_vocal = sys.argv[3]
    output_instrumental = sys.argv[4]

    process_audio(model_name, input_file, output_vocal, output_instrumental)