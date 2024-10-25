import sqlite3
import pandas as pd
import requests
import time
from functools import wraps, lru_cache
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import List, Tuple, Dict, Any
import os
import random
from collections import defaultdict
import gc
import json

SAMPLE_GENOME_RESPONSE = {
  "valid": [
    {
      "rsid": "rs12345",
      "genotype": ["A", "T"]
    },
    {
      "rsid": "rs67890",
      "genotype": ["G", "C"]
    },
    {
      "rsid": "rs13579",
      "genotype": ["C", "G"]
    }
  ],
  "invalid": [
    {
      "rsid": "rs12345",
      "genotype": ["A", "T"]
    },
    {
      "rsid": "rs67890",
      "genotype": ["G", "C"]
    },
    {
      "rsid": "rs13579",
      "genotype": ["C", "G"]
    }
  ]
}


import sqlite3
import time
from functools import wraps
import re
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


class DbSNPHandler:

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @staticmethod
    def is_i_rsid(rsid: str) -> bool:
        """Checks if the rsid starts with 'i' and is followed by digits."""
        return rsid.startswith('i') and rsid[1:].isdigit()

    @staticmethod
    def is_indel(genotype: str) -> bool:
        """Checks if the genotype is an indel."""
        return genotype == '--' or any(special in genotype.upper() for special in ['I', 'D'])

    def handle_special_cases(self, rsid_array: np.ndarray, genotype_array: np.ndarray, invalid_genotypes: List[str],
                             indels: List[str], i_rsids: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        This method combines the logic for handling both i-rsid and indel cases
        by checking the rsid and genotype arrays in a single function.
        """

        # Vectorize the checks for i-rsid and indels
        i_rsid_mask = np.vectorize(lambda x: self.is_i_rsid(x))(rsid_array)
        indel_mask = np.isin(genotype_array, ['--', 'II', 'DD'])

        # Handle indels
        indels.extend(rsid_array[indel_mask & np.isin(rsid_array, invalid_genotypes)].tolist())
        invalid_genotypes = list(set(invalid_genotypes) - set(rsid_array[indel_mask & np.isin(rsid_array, invalid_genotypes)]))

        # Handle i-rsids
        i_rsids.extend(rsid_array[i_rsid_mask & np.isin(rsid_array, invalid_genotypes)].tolist())
        invalid_genotypes = list(set(invalid_genotypes) - set(rsid_array[i_rsid_mask & np.isin(rsid_array, invalid_genotypes)]))

        return indels, i_rsids, invalid_genotypes

    def verify_snp(self, rsid: str | None, genotype: str) -> Tuple[None | str, None | str, None | str]:
        """
        Verifies the SNP by checking if it is a special rsid (e.g., i-rsid) or indel.
        """
        if rsid is None:
            return rsid, None, None

        if self.is_i_rsid(rsid):
            return None, None, rsid

        if self.is_indel(genotype):
            return None, genotype, None

        return rsid, None, None

    def check_indels_and_i_rsids(self, rsid_list: List[str], genotype_list: List[str], invalid_genotypes: List[str],
                                 indels: List[str], i_rsids: List[str]) -> Dict[str, int | List[Any]]:
        """
        Checks for both indels and i-rsids by leveraging the `handle_special_cases` function.
        """

        # Convert lists to arrays for vectorized operations
        rsid_array = np.array(rsid_list)
        genotype_array = np.array(genotype_list)

        # Check for special cases and update the indels, i_rsids, and invalid genotypes
        indels, i_rsids, invalid_genotypes = self.handle_special_cases(rsid_array, genotype_array, invalid_genotypes, indels, i_rsids)

        # Return the summary information
        dna_info = {
            'indels': len(indels),
            'i_rsids': len(i_rsids),
            'invalid_genotypes': len(invalid_genotypes),
            'all': len(indels) + len(i_rsids) + len(invalid_genotypes),
        }

        return dna_info

    def verify_snps(self, df: pd.DataFrame) -> Tuple[List[str | None]]:
        """
        Verifies SNPs by checking for invalid, indels, and i-rsids cases.
        """
        sampled_rsids = self.get_sampled_rsids(df)
        genome_response = self.verify_genome(sampled_rsids)

        invalid_list = genome_response.get('invalid', [])
        rsid_list = [item['rsid'] for item in invalid_list]
        genotype_list = [''.join(item['genotype']) for item in invalid_list]

        results = genome_response.get('valid', [])

        skipped_rsids = []
        indels = []
        i_rsids = []

        for rsid, genotype in zip(rsid_list, genotype_list):
            skipped_rsid, indel, i_rsid = self.verify_snp(rsid, genotype)
            if skipped_rsid:
                skipped_rsids.append(skipped_rsid)
            if indel:
                indels.append(indel)
            if i_rsid:
                i_rsids.append(i_rsid)

        return results, skipped_rsids, indels, i_rsids

    def get_sampled_rsids(self, df: pd.DataFrame) -> List[Dict[str, str | List[str]]]:
        """
        Samples RSIDs by selecting up to 10 items for each chromosome.
        """
        rsid_list = df['rsid'].tolist()
        genotype_list = df['genotype'].tolist()
        chromosomes = df['chromosome'].tolist()

        # Step 1: Group items by chromosome
        grouped_data = defaultdict(list)
        for rsid, genotype, chrom in zip(rsid_list, genotype_list, chromosomes):
            grouped_data[chrom].append((rsid, genotype))

        # Step 2: For each chromosome (1-23, X, Y, MT), select up to 10 items
        sampled_rsids = []
        chromosome_names = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

        for chrom in chromosome_names:
            if chrom in grouped_data:
                selected_items = random.sample(grouped_data[chrom], min(10, len(grouped_data[chrom])))

                for rsid, genotype in selected_items:
                    allele_list = list(set(genotype))  # Convert genotype to a set to remove duplicates

                    item_dict = {
                        'rsid': rsid,
                        'genotype': allele_list
                    }
                    sampled_rsids.append(item_dict)
        return sampled_rsids

    def verify_genome(self, final_list: List[Dict[str, str | List[str]]]) -> Dict[str, List[Dict[str, str | List[str]]]]:
        """
        Sends the genome data for verification via a POST request.
        """
        token = self.config['token']
        endpoint = self.config['endpoint']

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": 'application/json'
        }

        data = json.dumps({'genomes': final_list})

        response = requests.get(url=endpoint, data=data,
                                headers=headers)
        genome_response = response.json()
        return genome_response

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads the data from the file into a pandas DataFrame.
        """
        return pd.read_csv(filepath, comment='#', sep='\s+', names=['rsid', 'chromosome', 'position', 'genotype'],
                           dtype={'rsid': str, 'chromosome': str, 'position': int, 'genotype': str})

    def filter_valid_chromosomes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Filters the data to only include valid chromosomes.
        """
        valid_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

        df['chromosome'] = df['chromosome'].astype(str).str.strip()

        df_valid = df[df['chromosome'].isin(valid_chromosomes)]
        invalid_chromosomes = df[~df['chromosome'].isin(valid_chromosomes)]

        unique_invalid_chromosomes = []
        if not invalid_chromosomes.empty:
            unique_invalid_chromosomes += invalid_chromosomes['chromosome'].unique().tolist()
            print(f"Invalid chromosomes found: {', '.join(unique_invalid_chromosomes)}")

        unique_chromosomes_in_df = df['chromosome'].unique()
        missing_chromosomes = list(set(valid_chromosomes) - set(unique_chromosomes_in_df))

        return df_valid, unique_invalid_chromosomes, missing_chromosomes

    def check_genotypes(self, df_valid: pd.DataFrame) -> Dict[str, int | List[Any]]:
        """
        Runs checks on genotypes for SNPs.
        """
        dbsnp_verified, invalid_genotypes, indels, i_rsids = self.verify_snps(df_valid)

        rsid_list = df_valid['rsid'].tolist()
        genotype_list = df_valid['genotype'].tolist()

        dna_info = self.check_indels_and_i_rsids(rsid_list, genotype_list, invalid_genotypes, indels, i_rsids)

        dna_info['dbsnp_verified'] = len(dbsnp_verified)
        dna_info['all'] += len(dbsnp_verified)

        return dna_info

    def dbsnp_verify(self, filepath: str) -> Dict[str, Any]:
        """
        Verifies the SNPs in the provided file.
        """
        df = self.load_data(filepath)
        df_valid, invalid_chromosomes, missing_chromosomes = self.filter_valid_chromosomes(df)
        dna_info = self.check_genotypes(df_valid)
        dna_info['invalid_chromosomes'] = invalid_chromosomes
        dna_info['missing_chromosomes'] = missing_chromosomes

        del df, df_valid
        gc.collect()

        return dna_info


if __name__ == '__main__':
    handler = DbSNPHandler()

    filepath = '23andme_raw_data.txt'
    print(handler.dbsnp_verify(filepath))
    print_cumulative_times()

    verifier.close_connection()
