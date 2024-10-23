
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd


@dataclass
class ConfigEnvVars:
    user_email: Optional[str]
    token: Optional[str]
    expected_user_email: Optional[str]
    expected_token: Optional[str]


@dataclass
class ProfileIdInputData:
    data: List[str]
    expected_profile_id: Optional[str]


@dataclass
class HeaderTestData:
    data: List[str]
    header_included: bool
    expected_check_header: bool


@dataclass
class RSIDTestData:
    data: List[str]
    expected_check_rsid_lines: bool


@dataclass
class ProfileVerifyTestData:
    profile_id: str
    mock_response: Dict[str, bool]
    expected_verification: bool


@dataclass
class HashVerifyTestData:
    genome_hash: str
    mock_response: Dict[str, bool]
    expected_hash_verification: bool


@dataclass
class InvalidGenotypesScoreTestData:
    total: int
    low: int
    high: int
    expected_score: float


@dataclass
class IndelScoreTestData:
    total: float
    low: int
    ultra_low: int
    high: int
    ultra_high: int
    expected_score: float


@dataclass
class IRsidScoreTestData:
    total: int
    low: int
    high: int
    expected_score: float


@dataclass
class PercentVerifyScoreTestData:
    verified: int
    all: int
    low: float
    ultra_low: float
    high: float
    ultra_high: float
    expected_score: float


@dataclass
class IsIRsidTestData:
    rsid: str
    expected_is_i_rsid: bool


@dataclass
class IsIndelTestData:
    genotype: str
    expected_is_indel: bool


@dataclass
class HandleSpecialCasesTestData:
    rsid_array: np.ndarray
    genotype_array: np.ndarray
    invalid_genotypes: List[str]
    indels: List[str]
    i_rsids: List[str]
    expected_indels: List[str]
    expected_i_rsids: List[str]
    expected_invalid_genotypes: List[str]


@dataclass
class VerifySNPTestData:
    rsid: str | None
    genotype: str
    expected_rsid: str | None
    expected_indel: str | None
    expected_i_rsid: str | None


@dataclass
class ProofResponse:
    authenticity: float
    ownership: float
    uniqueness: float
    quality: float
    attributes: Dict[str, float]
    valid: bool
    score: Optional[float]
    dlp_id: Optional[int]


@dataclass
class HashSaveDataTestData:
    proof_response: Dict[str, any]
    expected_hash_save_data: Dict[str, any]


@dataclass
class ProofOfOwnershipTestData:
    mock_verify_profile_response: bool
    expected_ownership_score: float


@dataclass
class ProofOfQualityTestData:
    dbsnp_verify_result: Dict[str, int]
    expected_quality_score: float


@dataclass
class ProofOfUniquenessTestData:
    mock_hash_23andme_file_response: str
    mock_verify_hash_response: bool
    expected_uniqueness_score: float


@dataclass
class ProofOfAuthenticityTestData:
    mock_check_header_response: bool
    mock_check_rsid_lines_response: bool
    expected_authenticity_score: float


@dataclass
class SampledRSIDTestData:
    df: pd.DataFrame
    expected_sampled_rsids: List[Dict[str, str | List[str]]]


@dataclass
class VerifyGenomeTestData:
    final_list: List[Dict[str, str | List[str]]]
    expected_response: Dict[str, List[Dict[str, str | List[str]]]]


@dataclass
class LoadDataTestData:
    sample_data: str
    expected_df: pd.DataFrame


@dataclass
class FilterChromosomesTestData:
    input_df: pd.DataFrame
    expected_df_valid: pd.DataFrame
    expected_invalid_chromosomes: List[str]
    expected_missing_chromosomes: List[str]


@dataclass
class CheckGenotypesTestData:
    input_df_valid: pd.DataFrame
    mock_verify_snps_output: Tuple[List[str], List[str], List[str], List[str]]
    mock_check_indels_and_i_rsids_output: Dict[str, int | List[Any]]
    expected_output: Dict[str, int | List[Any]]


@dataclass
class DbsnpVerifyTestData:
    filepath: str
    mock_load_data_output: pd.DataFrame
    mock_filter_valid_chromosomes_output: Tuple[pd.DataFrame, List[str], List[str]]
    mock_check_genotypes_output: Dict[str, Any]
    expected_output: Dict[str, Any]


@dataclass
class CheckIndelsAndIRsidsTestData:
    rsid_list: List[str]
    genotype_list: List[str]
    invalid_genotypes: List[str]
    indels: List[str]
    i_rsids: List[str]
    mock_handle_special_cases_output: Tuple[List[str], List[str], List[str]]
    expected_output: Dict[str, int | List[Any]]


@dataclass
class VerifySNPsTestData:
    input_df: pd.DataFrame
    mock_get_sampled_rsids_output: List[Dict[str, str | List[str]]]
    mock_verify_genome_output: Dict[str, List[Dict[str, str | List[str]]]]
    mock_verify_snp_output: List[Tuple[None | str, None | str, None | str]]
    expected_output: Tuple[List[str | None], List[str], List[str], List[str]]


@dataclass
class ReadHeaderTestData:
    input_data: list[str]
    expected_header: str


@dataclass
class Hash23AndMeFileTestData:
    input_data: pd.DataFrame
    expected_concatenated_string: str
    expected_hash: str


@dataclass
class SaveHashTestData:
    proof_response: dict
    expected_hash_data: dict
    mock_response: dict
    expected_success: bool


@dataclass
class FileChangeConfig:
    file_path: str
    new_extension: str
    expected_new_file_path: str


@dataclass
class RunTestData:
    input_files_exist: bool
    file_list: list
    expected_exception: Exception | None


@dataclass
class GenerateTestData:
    file_list: list
    expected_uniqueness: float
    expected_ownership: float
    expected_authenticity: float
    expected_quality: float
    expected_valid: bool

