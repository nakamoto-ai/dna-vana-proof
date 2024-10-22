import json
import logging
import os
import sys
import traceback
import zipfile
from typing import Dict, Any

from my_proof.proof import Proof

INPUT_DIR, OUTPUT_DIR, SEALED_DIR = '/input', '/output', '/sealed'

logging.basicConfig(level=logging.INFO, format='%(message)s')


def load_config() -> Dict[str, Any]:
    """Load proof configuration from environment variables."""
    config = {
        'dlp_id': 22,  # Set your own DLP ID here
        'use_sealing': os.path.isdir(SEALED_DIR),
        'input_dir': INPUT_DIR,
        'user_email': os.environ.get('USER_EMAIL', None),
        'token': os.environ.get('TOKEN', None),
        'key': os.environ.get('KEY', None),
        'verify': os.environ.get('VERIFY', None),
        'endpoint': os.environ.get('ENDPOINT', None)
    }
    logging.info(f"Using config: {json.dumps(config, indent=2)}")
    return config


def run() -> None:
    """Generate proofs for all input files."""
    config = load_config()
    input_files_exist = os.path.isdir(INPUT_DIR) and bool(os.listdir(INPUT_DIR))

    if not input_files_exist:
        raise FileNotFoundError(f"No input files found in {INPUT_DIR}")

    for c_val in ['token', 'key', 'verify', 'endpoint']:
        if config[c_val] is not None:
            print(f"Config '{c_val}' Found: {config[c_val]}")
        else:
            print(f"Config '{c_val}' Not Found")

    input_file = os.path.join(config['input_dir'], os.listdir(config['input_dir'])[0])
    new_input_file = change_and_delete_file_extension(input_file, '.txt')

    proof = Proof(config)
    proof_response = proof.generate()

    output_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_path, 'w') as f:
        json.dump(proof_response.dict(), f, indent=2)
    logging.info(f"Proof generation complete: {proof_response}")


def change_and_delete_file_extension(file_path, new_extension):
    # Split the file path into name and extension
    base = os.path.splitext(file_path)[0]
    new_file_path = base + new_extension

    # Rename the file with the new extension
    os.rename(file_path, new_file_path)

    # Remove (delete) the old file
    if os.path.exists(file_path):
        os.remove(file_path)

    return new_file_path


def extract_input() -> None:
    """
    If the input directory contains any zip files, extract them
    :return:
    """
    for input_filename in os.listdir(INPUT_DIR):
        input_file = os.path.join(INPUT_DIR, input_filename)

        if zipfile.is_zipfile(input_file):
            with zipfile.ZipFile(input_file, 'r') as zip_ref:
                zip_ref.extractall(INPUT_DIR)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logging.error(f"Error during proof generation: {e}")
        traceback.print_exc()
        sys.exit(1)
