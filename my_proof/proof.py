import json
import logging
import os
from typing import Dict, Any

import requests

from my_proof.models.proof_response import ProofResponse


class TwentyThreeWeFileScorer:
    header_template = """
    # This file contains raw genotype data, including data that is not used in 23andMe reports.
    # This data has undergone a general quality review however only a subset of markers have been 
    # individually validated for accuracy. As such, this data is suitable only for research, 
    # educational, and informational use and not for medical or other use.
    # 
    # Below is a text version of your data.  Fields are TAB-separated
    # Each line corresponds to a single SNP.  For each SNP, we provide its identifier 
    # (an rsid or an internal id), its location on the reference human genome, and the 
    # genotype call oriented with respect to the plus strand on the human reference sequence.
    # We are using reference human assembly build 37 (also known as Annotation Release 104).
    # Note that it is possible that data downloaded at different times may be different due to ongoing 
    # improvements in our ability to call genotypes. More information about these changes can be found at:
    #
    # More information on reference human assembly builds:
    # https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.13/
    #
    # rsid	chromosome	position	genotype
    """
    valid_genotypes = set("ATCG-ID")
    valid_chromosomes = set([str(i) for i in range(1, 23)] + ["X", "Y", "MT"])

    def __init__(self, input_data):
        self.input_data = input_data

    def read_header(self):
        header_lines = ["a", "b", "c", "d"]
        return "\n".join(header_lines)

    def check_header(self):
        file_header = self.read_header()
        return True

    def check_rsid_lines(self):
        invalid_rows = []
        return invalid_rows

    @staticmethod
    def invalid_genotypes_score(total: int, low: int = 2000, high: int = 5000):
        if total < low:
            return 1.0
        elif total > high:
            return 0.0
        else:
            # Decrease score linearly from 1.0 to 0.0 between 'low' and 'high'
            return 1.0 - (total - low) / (low - high)

    @staticmethod
    def indel_score(total, low: int = 10000, ultra_low: int = 5000, high: int = 30000, ultra_high: int = 50000):
        if total <= ultra_low:
            return 0.0
        elif ultra_low < total <= low:
            return (total - ultra_low) / (low - ultra_low)
        elif low < total <= high:
            return 1.0
        elif high < total <= ultra_high:
            return 1.0 - (total - high) / (ultra_high - high)
        else:
            return 0.0

    @staticmethod
    def i_rsid_score(total: int, low: int = 20000, high: int = 35000):
        if total < low:
            return 1.0
        elif total > high:
            return 0.0
        else:
            # Decrease score linearly from 1.0 to 0.0 between 'low' and 'high'
            return 1.0 - (total - low) / (low - high)

    @staticmethod
    def percent_verification_score(verified: int, all: int):
        verified_ratio = verified / all
        if 0.93 <= verified_ratio <= 0.96:
            return 1.0
        elif 0.9 < verified_ratio < 0.93:
            return 1 - (verified - 0.9) / 0.03
        elif 0.96 < verified_ratio <= 0.975:
            return 1 - (verified - 0.96) / 0.015
        elif verified_ratio > 0.975:
            return 0
        else:
            return 0

    @staticmethod
    def proof_of_ownership() -> float:
        return 1.0

    def proof_of_quality(self, filepath) -> float:
        return 1.0

    def proof_of_uniqueness(self) -> float:
        return 1.0

    def proof_of_authenticity(self) -> float:
        return 1.0


class Proof:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.proof_response = ProofResponse(dlp_id=config['dlp_id'])

    def generate(self) -> ProofResponse:
        """Generate proofs for all input files."""
        logging.info("Starting proof generation")

        # Iterate through files and calculate data validity
        account_email = None

        scorer = None
        twenty_three_file = None
        for input_filename in os.listdir(self.config['input_dir']):
            input_file = os.path.join(self.config['input_dir'], input_filename)
            with open(input_file, 'r') as i_file:

                if input_filename.split('.')[-1] == '.txt':
                    twenty_three_file = input_file
                    input_data = [f for f in i_file]
                    scorer = TwentyThreeWeFileScorer(input_data=input_data)
                    break

        # email_matches = self.config['user_email'] == account_email
        email_matches = True
        score_threshold = 0.9

        self.proof_response.ownership = scorer.proof_of_ownership()
        self.proof_response.quality = scorer.proof_of_quality(filepath=twenty_three_file)
        self.proof_response.authenticity = scorer.proof_of_authenticity()
        self.proof_response.uniqueness = scorer.proof_of_uniqueness()

        # Calculate overall score and validity
        total_score = 0.25 * self.proof_response.quality + 0.25 * self.proof_response.ownership + 0.25 * self.proof_response.authenticity + 0.25 * self.proof_response.uniqueness
        self.proof_response.score = total_score
        self.proof_response.valid = email_matches and total_score >= score_threshold

        # Additional (public) properties to include in the proof about the data
        self.proof_response.attributes = {
            'total_score': total_score,
            'score_threshold': score_threshold,
            'email_verified': email_matches,
        }

        # Additional metadata about the proof, written onchain
        self.proof_response.metadata = {
            'dlp_id': self.config['dlp_id'],
        }

        return self.proof_response
