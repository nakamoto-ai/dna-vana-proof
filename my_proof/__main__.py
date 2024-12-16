import json
import logging
import os
import sys
import traceback
import zipfile
import copy
from typing import Dict, Any

from my_proof.proof import Proof

INPUT_DIR, OUTPUT_DIR = '/input', '/output'

logging.basicConfig(level=logging.INFO, format='%(message)s')


def reverse(s):
    return ''.join(list(s)[::-1])


def load_config() -> Dict[str, Any]:
    config = {
        'dlp_id': 2,
        'input_dir': INPUT_DIR,
        'user_email': os.environ.get('USER_EMAIL', None),
        'token': os.environ.get('TOKEN', None),
        'key': os.environ.get('KEY', None),
        'verify': os.environ.get('VERIFY', None),
        'endpoint': os.environ.get('ENDPOINT', None)
    }
    print_config = copy.deepcopy(config)
    for c, v in print_config.items():
        print_config[c] = reverse(str(v))
    print(f"Configs Reversed: {print_config}")
    return config


def run() -> None:
    config = load_config()
    input_files_exist = os.path.isdir(INPUT_DIR) and bool(os.listdir(INPUT_DIR))

    if not input_files_exist:
        raise FileNotFoundError(f"No input files found in {INPUT_DIR}")

    proof = Proof(config)
    proof_response = proof.generate()

    output_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_path, 'w') as f:
        json.dump(proof_response.dict(), f, indent=2)
    logging.info(f"Proof generation complete: {proof_response}")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logging.error(f"Error during proof generation: {e}")
        traceback.print_exc()
        sys.exit(1)
