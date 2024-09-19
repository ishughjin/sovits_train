import os
from subprocess import Popen, CalledProcessError
import my_utils
from tools.asr.config import asr_dict

# Custom exception classes
class ASRError(Exception):
    """Base exception class for ASR processing errors"""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details

class ASRInputError(ASRError):
    """Exception raised for errors in the input"""
    def __init__(self, message, input_type, input_value):
        super().__init__(message, {"input_type": input_type, "input_value": input_value})

class ASRProcessError(ASRError):
    """Exception raised for errors during ASR processing"""
    def __init__(self, message, cmd=None, returncode=None):
        super().__init__(message, {"cmd": cmd, "returncode": returncode})

class ASROutputError(ASRError):
    """Exception raised for errors related to ASR output"""
    def __init__(self, message, output_path):
        super().__init__(message, {"output_path": output_path})

# Global variable to track the ASR process
asr_process = None


def run_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision, python_exec):
    """
    Run ASR (Automatic Speech Recognition) task.

    Parameters:
    asr_inp_dir (str): Directory of input audio files
    asr_opt_dir (str): Directory for ASR output
    asr_model (str): Name of the ASR model to use
    asr_model_size (str): Size of the model
    asr_lang (str): Language for recognition
    asr_precision (str): Computation precision
    python_exec (str): Path to Python executor

    Returns:
    str: Full path to the output file (including filename)

    Raises:
    ASRInputError: When input parameters are invalid
    ASRProcessError: When an error occurs during ASR processing
    ASROutputError: When the ASR output file does not exist
    """
    global asr_process

    if asr_process is not None:
        raise ASRProcessError("An ASR task is already in progress. Terminate the current task before starting a new one.")

    asr_inp_dir = my_utils.clean_path(asr_inp_dir)
    asr_opt_dir = my_utils.clean_path(asr_opt_dir)

    if not os.path.exists(asr_inp_dir):
        raise ASRInputError(f"Input directory does not exist: {asr_inp_dir}", "input_directory", asr_inp_dir)

    if asr_model not in asr_dict:
        raise ASRInputError(f"Unsupported ASR model: {asr_model}", "asr_model", asr_model)

    cmd = f'"{python_exec}" tools/asr/{asr_dict[asr_model]["path"]}'
    cmd += f' -i "{asr_inp_dir}"'
    cmd += f' -o "{asr_opt_dir}"'
    cmd += f' -s {asr_model_size}'
    cmd += f' -l {asr_lang}'
    cmd += f" -p {asr_precision}"

    # Fixed output file name
    output_file_name = "slicer.list"
    output_folder = asr_opt_dir or "output/asr_opt"
    output_file_path = os.path.join(output_folder, output_file_name)

    print(f"Starting ASR task: {cmd}")
    try:
        asr_process = Popen(cmd, shell=True)
        asr_process.wait()
        if asr_process.returncode != 0:
            raise ASRProcessError(f"ASR process returned non-zero exit code: {asr_process.returncode}", cmd, asr_process.returncode)
    except CalledProcessError as e:
        raise ASRProcessError(f"ASR process execution failed: {str(e)}", cmd, e.returncode)
    finally:
        asr_process = None

    if not os.path.exists(output_file_path):
        raise ASROutputError(f"ASR task completed, but output file not found: {output_file_path}", output_file_path)

    return output_file_path

def stop_asr():
    """
    Stop the currently running ASR process.

    Returns:
    str: Message indicating the result of the operation

    Raises:
    ASRProcessError: When attempting to stop a non-existent process
    """
    global asr_process
    if asr_process is not None:
        try:
            my_utils.kill_process(asr_process.pid)
            asr_process = None
            return "ASR process has been terminated"
        except Exception as e:
            raise ASRProcessError(f"Error occurred while terminating ASR process: {str(e)}")
    else:
        raise ASRProcessError("No running ASR process to terminate")