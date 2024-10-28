
import pytest
from unittest import mock
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from io import StringIO
import numpy as np
import numpy.testing as npt
import pandas as pd
import json

from my_proof.proof import DbSNPHandler


class TestDbSNPHandler:

    @dataclass
    class IsIRsidTestData:
        rsid: str
        expected_is_i_rsid: bool

    @pytest.mark.parametrize("input_data", [
        IsIRsidTestData(rsid="i12345", expected_is_i_rsid=True),
        IsIRsidTestData(rsid="rs12345", expected_is_i_rsid=False),
        IsIRsidTestData(rsid="iabcde", expected_is_i_rsid=False),
        IsIRsidTestData(rsid="i00001", expected_is_i_rsid=True),
        IsIRsidTestData(rsid="x12345", expected_is_i_rsid=False)
    ])
    def test_is_i_rsid(self, input_data: IsIRsidTestData) -> None:
        result = DbSNPHandler.is_i_rsid(input_data.rsid)
        assert result == input_data.expected_is_i_rsid

    @dataclass
    class IsIndelTestData:
        genotype: str
        expected_is_indel: bool

    @pytest.mark.parametrize("input_data", [
        IsIndelTestData(genotype="--", expected_is_indel=True),
        IsIndelTestData(genotype="ID", expected_is_indel=True),
        IsIndelTestData(genotype="DI", expected_is_indel=True),
        IsIndelTestData(genotype="AA", expected_is_indel=False),
        IsIndelTestData(genotype="AC", expected_is_indel=False),
        IsIndelTestData(genotype="Ii", expected_is_indel=True),
        IsIndelTestData(genotype="Dd", expected_is_indel=True)
    ])
    def test_is_indel(self, input_data: IsIndelTestData) -> None:
        result = DbSNPHandler.is_indel(input_data.genotype)
        assert result == input_data.expected_is_indel

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

    @pytest.mark.parametrize("input_data", [
        HandleSpecialCasesTestData(
            rsid_array=np.array(["rs1", "i123", "rs3"]),
            genotype_array=np.array(["AA", "TT", "DD"]),
            invalid_genotypes=["rs1", "rs3"],
            indels=[],
            i_rsids=[],
            expected_indels=["rs3"],
            expected_i_rsids=["i123"],
            expected_invalid_genotypes=["rs1"]
        ),
        HandleSpecialCasesTestData(
            rsid_array=np.array(["rs2", "i456", "rs5"]),
            genotype_array=np.array(["GG", "CC", "AA"]),
            invalid_genotypes=["rs2", "i456"],
            indels=[],
            i_rsids=[],
            expected_indels=[],
            expected_i_rsids=["i456"],
            expected_invalid_genotypes=["rs2"]
        ),
        HandleSpecialCasesTestData(
            rsid_array=np.array(["i789", "rs6", "rs7"]),
            genotype_array=np.array(["TT", "DD", "--"]),
            invalid_genotypes=["rs6", "rs7"],
            indels=[],
            i_rsids=[],
            expected_indels=["rs6", "rs7"],
            expected_i_rsids=["i789"],
            expected_invalid_genotypes=[]
        )
    ])
    def test_handle_special_cases(self, input_data: HandleSpecialCasesTestData) -> None:
        scorer = DbSNPHandler(config={})

        indels, i_rsids, invalid_genotypes = scorer.handle_special_cases(
            rsid_array=input_data.rsid_array,
            genotype_array=input_data.genotype_array,
            invalid_genotypes=input_data.invalid_genotypes,
            indels=input_data.indels,
            i_rsids=input_data.i_rsids
        )

        assert indels == input_data.expected_indels
        assert i_rsids == input_data.expected_i_rsids
        assert invalid_genotypes == input_data.expected_invalid_genotypes

    @dataclass
    class VerifySNPTestData:
        rsid: str | None
        genotype: str
        expected_rsid: str | None
        expected_indel: str | None
        expected_i_rsid: str | None

    @pytest.mark.parametrize("input_data", [
        VerifySNPTestData(
            rsid="rs1", genotype="AA",
            expected_rsid="rs1", expected_indel=None, expected_i_rsid=None
        ),
        VerifySNPTestData(
            rsid="i123", genotype="GG",
            expected_rsid=None, expected_indel=None, expected_i_rsid="i123"
        ),
        VerifySNPTestData(
            rsid="rs3", genotype="--",
            expected_rsid=None, expected_indel="rs3", expected_i_rsid=None
        ),
        VerifySNPTestData(
            rsid=None, genotype="AA",
            expected_rsid=None, expected_indel=None, expected_i_rsid=None
        ),
        VerifySNPTestData(
            rsid="rs5", genotype="DD",
            expected_rsid=None, expected_indel="rs5", expected_i_rsid=None
        )
    ])
    def test_verify_snp(self, input_data: VerifySNPTestData) -> None:
        scorer = DbSNPHandler(config={})

        result_rsid, result_indel, result_i_rsid = scorer.verify_snp(
            rsid=input_data.rsid, genotype=input_data.genotype
        )

        assert result_rsid == input_data.expected_rsid
        assert result_indel == input_data.expected_indel
        assert result_i_rsid == input_data.expected_i_rsid

    @dataclass
    class SampledRSIDTestData:
        df: pd.DataFrame
        expected_sampled_rsids: List[Dict[str, str | List[str]]]

    @pytest.mark.parametrize("input_data", [
        SampledRSIDTestData(
            df=pd.DataFrame({
                'rsid': ['rs1', 'rs2', 'rs3', 'rs4'],
                'genotype': ['AA', 'GG', 'TT', 'CC'],
                'chromosome': ['1', '1', '2', 'X']
            }),
            expected_sampled_rsids=[
                {'rsid': 'rs1', 'genotype': ['A']},
                {'rsid': 'rs2', 'genotype': ['G']},
                {'rsid': 'rs3', 'genotype': ['T']},
                {'rsid': 'rs4', 'genotype': ['C']}
            ]
        ),
        SampledRSIDTestData(
            df=pd.DataFrame({
                'rsid': ['rs5', 'rs6', 'rs7', 'rs8', 'rs9'],
                'genotype': ['AA', 'CC', 'GG', 'TT', 'CC'],
                'chromosome': ['1', '1', '1', '1', '1']
            }),
            expected_sampled_rsids=[
                {'rsid': 'rs5', 'genotype': ['A']},
                {'rsid': 'rs6', 'genotype': ['C']},
                {'rsid': 'rs7', 'genotype': ['G']},
                {'rsid': 'rs8', 'genotype': ['T']},
                {'rsid': 'rs9', 'genotype': ['C']}
            ]
        ),
        SampledRSIDTestData(
            df=pd.DataFrame({
                'rsid': ['rs10', 'rs11', 'rs12'],
                'genotype': ['AA', 'CC', 'GG'],
                'chromosome': ['X', 'Y', 'MT']
            }),
            expected_sampled_rsids=[
                {'rsid': 'rs10', 'genotype': ['A']},
                {'rsid': 'rs11', 'genotype': ['C']},
                {'rsid': 'rs12', 'genotype': ['G']}
            ]
        ),
    ])
    def test_get_sampled_rsids(self, input_data: SampledRSIDTestData) -> None:
        scorer = DbSNPHandler(config={})

        result_sampled_rsids = scorer.get_sampled_rsids(input_data.df)

        assert sorted(result_sampled_rsids, key=lambda x: x['rsid']) == sorted(input_data.expected_sampled_rsids,
                                                                               key=lambda x: x['rsid'])

    @dataclass
    class VerifyGenomeTestData:
        final_list: List[Dict[str, str | List[str]]]
        expected_response: Dict[str, List[Dict[str, str | List[str]]]]

    @pytest.mark.parametrize("input_data", [
        VerifyGenomeTestData(
            final_list=[{'rsid': 'rs1', 'genotype': ['A', 'A']}],
            expected_response={'valid': [{'rsid': 'rs1', 'genotype': ['A', 'A']}]}
        ),
        VerifyGenomeTestData(
            final_list=[{'rsid': 'rs2', 'genotype': ['G', 'G']}],
            expected_response={'invalid': [{'rsid': 'rs2', 'genotype': ['G', 'G']}]}
        ),
        VerifyGenomeTestData(
            final_list=[{'rsid': 'rs3', 'genotype': ['T']}],
            expected_response={'valid': [{'rsid': 'rs3', 'genotype': ['T']}]}
        ),
    ])
    @patch("requests.get")
    def test_verify_genome(self, mock_get: mock.MagicMock, input_data: VerifyGenomeTestData) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = input_data.expected_response
        mock_get.return_value = mock_response

        config = {
            'token': 'test-token',
            'endpoint': 'http://fake-endpoint.com/verify'
        }
        scorer = DbSNPHandler(config=config)

        result = scorer.verify_genome(input_data.final_list)

        mock_get.assert_called_once_with(
            url=config['endpoint'],
            data=json.dumps({'genomes': input_data.final_list}),
            headers={
                "Authorization": f"Bearer {config['token']}",
                "Content-Type": 'application/json'
            }
        )

        assert result == input_data.expected_response

    @dataclass
    class LoadDataTestData:
        sample_data: str
        expected_df: pd.DataFrame

    @pytest.mark.parametrize("input_data", [
        LoadDataTestData(
            sample_data="""
            rs1035804    4    175372799    CT
            rs12649767   4    175375824    AA
            rs147587276  4    175377096    --
            rs17358909   4    175381743    CC
            rs79164672   4    175385354    CC
            rs114885136  4    175387978    GG
            rs188971327  4    175392609    TT
            rs77565390   4    175402507    AA
            rs139575437  4    175407046    GG
            rs8752       4    175412477    CT
            rs13106936   4    175415991    GG
            rs181587981  4    175416710    GG
            rs2612656    4    175422289    AG
            i710260      4    175428308    TT
            rs114764767  4    175428815    AA
            i708485      4    175429879    CC
            rs45484594   4    175430278    AA
            rs45437095   4    175431488    AA
            """,
            expected_df=pd.DataFrame({
                'rsid': [
                    'rs1035804', 'rs12649767', 'rs147587276', 'rs17358909', 'rs79164672',
                    'rs114885136', 'rs188971327', 'rs77565390', 'rs139575437', 'rs8752',
                    'rs13106936', 'rs181587981', 'rs2612656', 'i710260', 'rs114764767',
                    'i708485', 'rs45484594', 'rs45437095'
                ],
                'chromosome': ['4'] * 18,
                'position': [
                    175372799, 175375824, 175377096, 175381743, 175385354, 175387978,
                    175392609, 175402507, 175407046, 175412477, 175415991, 175416710,
                    175422289, 175428308, 175428815, 175429879, 175430278, 175431488
                ],
                'genotype': [
                    'CT', 'AA', '--', 'CC', 'CC', 'GG', 'TT', 'AA', 'GG', 'CT',
                    'GG', 'GG', 'AG', 'TT', 'AA', 'CC', 'AA', 'AA'
                ]
            })
        )
    ])
    def test_load_data(self, input_data: LoadDataTestData) -> None:
        file_like_object = StringIO(input_data.sample_data.strip())

        scorer = DbSNPHandler(config={})
        result_df = scorer.load_data(filepath=file_like_object)

        pd.testing.assert_frame_equal(result_df, input_data.expected_df)

    @dataclass
    class FilterChromosomesTestData:
        input_df: pd.DataFrame
        expected_df_valid: pd.DataFrame
        expected_invalid_chromosomes: List[str]
        expected_missing_chromosomes: List[str]

    @pytest.mark.parametrize("input_data", [
        FilterChromosomesTestData(
            input_df=pd.DataFrame({
                'rsid': ['rs1', 'rs2', 'rs3', 'rs4', 'rs5'],
                'chromosome': ['1', '2', 'MT', 'X', '25'],
                'position': [1000, 2000, 3000, 4000, 5000],
                'genotype': ['AA', 'GG', 'CC', 'TT', 'AA']
            }),
            expected_df_valid=pd.DataFrame({
                'rsid': ['rs1', 'rs2', 'rs3', 'rs4'],
                'chromosome': ['1', '2', 'MT', 'X'],
                'position': [1000, 2000, 3000, 4000],
                'genotype': ['AA', 'GG', 'CC', 'TT']
            }),
            expected_invalid_chromosomes=['25'],
            expected_missing_chromosomes=['3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                                          '17', '18', '19', '20', '21', '22', 'Y']
        ),
        FilterChromosomesTestData(
            input_df=pd.DataFrame({
                'rsid': ['rs6', 'rs7', 'rs8'],
                'chromosome': ['Y', 'Z', 'MT'],
                'position': [6000, 7000, 8000],
                'genotype': ['AG', 'CT', 'GG']
            }),
            expected_df_valid=pd.DataFrame({
                'rsid': ['rs6', 'rs8'],
                'chromosome': ['Y', 'MT'],
                'position': [6000, 8000],
                'genotype': ['AG', 'GG']
            }),
            expected_invalid_chromosomes=['Z'],
            expected_missing_chromosomes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                                          '15', '16', '17', '18', '19', '20', '21', '22', 'X']
        )
    ])
    def test_filter_valid_chromosomes(self, input_data: FilterChromosomesTestData) -> None:
        scorer = DbSNPHandler(config={})

        df_valid, invalid_chromosomes, missing_chromosomes = scorer.filter_valid_chromosomes(input_data.input_df)

        df_valid = df_valid.reset_index(drop=True)
        expected_df_valid = input_data.expected_df_valid.reset_index(drop=True)

        pd.testing.assert_frame_equal(df_valid, expected_df_valid)
        assert invalid_chromosomes == input_data.expected_invalid_chromosomes
        assert sorted(missing_chromosomes) == sorted(input_data.expected_missing_chromosomes)

    @dataclass
    class CheckGenotypesTestData:
        input_df_valid: pd.DataFrame
        mock_verify_snps_output: Tuple[List[str], List[str], List[str], List[str]]
        mock_check_indels_and_i_rsids_output: Dict[str, int | List[Any]]
        expected_output: Dict[str, int | List[Any]]

    @pytest.mark.parametrize("input_data", [
        CheckGenotypesTestData(
            input_df_valid=pd.DataFrame({
                'rsid': ['rs1', 'rs2', 'rs3'],
                'genotype': ['AA', '--', 'i123'],
                'chromosome': ['1', '1', '1']
            }),
            mock_verify_snps_output=(
                    ['rs1'],
                    ['rs2'],
                    ['rs2'],
                    ['rs3']
            ),
            mock_check_indels_and_i_rsids_output={
                'indels': 1,
                'i_rsids': 1,
                'invalid_genotypes': 1,
                'all': 3
            },
            expected_output={
                'indels': 1,
                'i_rsids': 1,
                'invalid_genotypes': 1,
                'all': 4,
                'dbsnp_verified': 1
            }
        ),
        CheckGenotypesTestData(
            input_df_valid=pd.DataFrame({
                'rsid': ['rs4', 'rs5'],
                'genotype': ['TT', 'GG'],
                'chromosome': ['X', 'Y']
            }),
            mock_verify_snps_output=(
                    ['rs4', 'rs5'],
                    [],
                    [],
                    []
            ),
            mock_check_indels_and_i_rsids_output={
                'indels': 0,
                'i_rsids': 0,
                'invalid_genotypes': 0,
                'all': 0
            },
            expected_output={
                'indels': 0,
                'i_rsids': 0,
                'invalid_genotypes': 0,
                'all': 2,
                'dbsnp_verified': 2
            }
        )
    ])
    @patch.object(DbSNPHandler, 'verify_snps')
    @patch.object(DbSNPHandler, 'check_indels_and_i_rsids')
    def test_check_genotypes(self, mock_check_indels_and_i_rsids: mock.MagicMock, mock_verify_snps: mock.MagicMock,
                             input_data: CheckGenotypesTestData) -> None:

        scorer = DbSNPHandler(config={})

        mock_verify_snps.return_value = input_data.mock_verify_snps_output
        mock_check_indels_and_i_rsids.return_value = input_data.mock_check_indels_and_i_rsids_output

        result = scorer.check_genotypes(input_data.input_df_valid)

        mock_verify_snps.assert_called_once_with(input_data.input_df_valid)
        mock_check_indels_and_i_rsids.assert_called_once_with(
            input_data.input_df_valid['rsid'].tolist(),
            input_data.input_df_valid['genotype'].tolist(),
            input_data.mock_verify_snps_output[1],
            input_data.mock_verify_snps_output[2],
            input_data.mock_verify_snps_output[3]
        )

        assert result == input_data.expected_output

    @dataclass
    class DbsnpVerifyTestData:
        filepath: str
        mock_load_data_output: pd.DataFrame
        mock_filter_valid_chromosomes_output: Tuple[pd.DataFrame, List[str], List[str]]
        mock_check_genotypes_output: Dict[str, Any]
        expected_output: Dict[str, Any]

    @pytest.mark.parametrize("input_data", [
        DbsnpVerifyTestData(
            filepath="test_data_1.txt",
            mock_load_data_output=pd.DataFrame({
                'rsid': ['rs1', 'rs2', 'rs3'],
                'chromosome': ['1', '25', 'MT'],
                'position': [1000, 2000, 3000],
                'genotype': ['AA', 'GG', '--']
            }),
            mock_filter_valid_chromosomes_output=(
                    pd.DataFrame({
                        'rsid': ['rs1', 'rs3'],
                        'chromosome': ['1', 'MT'],
                        'position': [1000, 3000],
                        'genotype': ['AA', '--']
                    }),
                    ['25'],
                    ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                     '20', '21', '22', 'X', 'Y']
            ),
            mock_check_genotypes_output={
                'indels': 1,
                'i_rsids': 0,
                'invalid_genotypes': 0,
                'all': 1,
                'dbsnp_verified': 1
            },
            expected_output={
                'indels': 1,
                'i_rsids': 0,
                'invalid_genotypes': 0,
                'all': 1,
                'dbsnp_verified': 1,
                'invalid_chromosomes': ['25'],
                'missing_chromosomes': ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                                        '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']
            }
        ),
        DbsnpVerifyTestData(
            filepath="test_data_2.txt",
            mock_load_data_output=pd.DataFrame({
                'rsid': ['rs4', 'rs5'],
                'chromosome': ['X', 'Y'],
                'position': [4000, 5000],
                'genotype': ['TT', 'GG']
            }),
            mock_filter_valid_chromosomes_output=(
                    pd.DataFrame({
                        'rsid': ['rs4', 'rs5'],
                        'chromosome': ['X', 'Y'],
                        'position': [4000, 5000],
                        'genotype': ['TT', 'GG']
                    }),
                    [],
                    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                     '19', '20', '21', '22', 'MT']
            ),
            mock_check_genotypes_output={
                'indels': 0,
                'i_rsids': 0,
                'invalid_genotypes': 0,
                'all': 2,
                'dbsnp_verified': 2
            },
            expected_output={
                'indels': 0,
                'i_rsids': 0,
                'invalid_genotypes': 0,
                'all': 2,
                'dbsnp_verified': 2,
                'invalid_chromosomes': [],
                'missing_chromosomes': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                                        '16', '17', '18', '19', '20', '21', '22', 'MT']
            }
        )
    ])
    @patch.object(DbSNPHandler, 'load_data')
    @patch.object(DbSNPHandler, 'filter_valid_chromosomes')
    @patch.object(DbSNPHandler, 'check_genotypes')
    @patch('gc.collect')
    def test_dbsnp_verify(self, mock_gc_collect: mock.MagicMock, mock_check_genotypes: mock.MagicMock,
                          mock_filter_valid_chromosomes: mock.MagicMock, mock_load_data: mock.MagicMock,
                          input_data: DbsnpVerifyTestData) -> None:
        scorer = DbSNPHandler(config={})

        mock_load_data.return_value = input_data.mock_load_data_output
        mock_filter_valid_chromosomes.return_value = input_data.mock_filter_valid_chromosomes_output
        mock_check_genotypes.return_value = input_data.mock_check_genotypes_output

        result = scorer.dbsnp_verify(input_data.filepath)

        mock_load_data.assert_called_once_with(input_data.filepath)
        mock_filter_valid_chromosomes.assert_called_once_with(input_data.mock_load_data_output)
        mock_check_genotypes.assert_called_once_with(input_data.mock_filter_valid_chromosomes_output[0])
        mock_gc_collect.assert_called_once()

        assert result == input_data.expected_output

    @dataclass
    class CheckIndelsAndIRsidsTestData:
        rsid_list: List[str]
        genotype_list: List[str]
        invalid_genotypes: List[str]
        indels: List[str]
        i_rsids: List[str]
        mock_handle_special_cases_output: Tuple[List[str], List[str], List[str]]
        expected_output: Dict[str, int | List[Any]]

    @pytest.mark.parametrize("input_data", [
        CheckIndelsAndIRsidsTestData(
            rsid_list=['rs1', 'rs2', 'rs3'],
            genotype_list=['AA', '--', 'i123'],
            invalid_genotypes=['rs2'],
            indels=['rs2'],
            i_rsids=['rs3'],
            mock_handle_special_cases_output=(
                    ['rs2'],
                    ['rs3'],
                    ['rs2']
            ),
            expected_output={
                'indels': 1,
                'i_rsids': 1,
                'invalid_genotypes': 1,
                'all': 3
            }
        ),
        CheckIndelsAndIRsidsTestData(
            rsid_list=['rs4', 'rs5'],
            genotype_list=['TT', 'GG'],
            invalid_genotypes=[],
            indels=[],
            i_rsids=[],
            mock_handle_special_cases_output=(
                    [],
                    [],
                    []
            ),
            expected_output={
                'indels': 0,
                'i_rsids': 0,
                'invalid_genotypes': 0,
                'all': 0
            }
        )
    ])
    @patch.object(DbSNPHandler, 'handle_special_cases')
    def test_check_indels_and_i_rsids(self, mock_handle_special_cases: mock.MagicMock,
                                      input_data: CheckIndelsAndIRsidsTestData) -> None:
        scorer = DbSNPHandler(config={})

        mock_handle_special_cases.return_value = input_data.mock_handle_special_cases_output

        result = scorer.check_indels_and_i_rsids(
            rsid_list=input_data.rsid_list,
            genotype_list=input_data.genotype_list,
            invalid_genotypes=input_data.invalid_genotypes,
            indels=input_data.indels,
            i_rsids=input_data.i_rsids
        )

        rsid_array = np.array(input_data.rsid_list)
        genotype_array = np.array(input_data.genotype_list)

        called_args = mock_handle_special_cases.call_args[0]

        npt.assert_array_equal(called_args[0], rsid_array)
        npt.assert_array_equal(called_args[1], genotype_array)

        assert called_args[2] == input_data.invalid_genotypes
        assert called_args[3] == input_data.indels
        assert called_args[4] == input_data.i_rsids

        assert result == input_data.expected_output

    @dataclass
    class VerifySNPsTestData:
        input_df: pd.DataFrame
        mock_get_sampled_rsids_output: List[Dict[str, str | List[str]]]
        mock_verify_genome_output: Dict[str, List[Dict[str, str | List[str]]]]
        mock_verify_snp_output: List[Tuple[None | str, None | str, None | str]]
        expected_output: Tuple[List[str | None], List[str], List[str], List[str]]

    @pytest.mark.parametrize("input_data", [
        VerifySNPsTestData(
            input_df=pd.DataFrame({
                'rsid': ['rs1', 'rs2', 'rs3'],
                'chromosome': ['1', '1', '1'],
                'position': [1000, 2000, 3000],
                'genotype': ['AA', 'GG', '--']
            }),
            mock_get_sampled_rsids_output=[
                {'rsid': 'rs1', 'genotype': ['A']},
                {'rsid': 'rs2', 'genotype': ['G']},
                {'rsid': 'rs3', 'genotype': ['--']}
            ],
            mock_verify_genome_output={
                'valid': [{'rsid': 'rs1', 'genotype': ['A']}],
                'invalid': [{'rsid': 'rs3', 'genotype': ['--']}]
            },
            mock_verify_snp_output=[
                (None, 'rs3', None)
            ],
            expected_output=(
                    [{'rsid': 'rs1', 'genotype': ['A']}],
                    [],
                    ['rs3'],
                    []
            )
        ),
        VerifySNPsTestData(
            input_df=pd.DataFrame({
                'rsid': ['rs4', 'rs5', 'i123'],
                'chromosome': ['X', 'Y', '1'],
                'position': [4000, 5000, 6000],
                'genotype': ['TT', 'GG', 'TT']
            }),
            mock_get_sampled_rsids_output=[
                {'rsid': 'rs4', 'genotype': ['T']},
                {'rsid': 'rs5', 'genotype': ['G']},
                {'rsid': 'i123', 'genotype': ['T']}
            ],
            mock_verify_genome_output={
                'valid': [{'rsid': 'rs4', 'genotype': ['T']}],
                'invalid': [{'rsid': 'i123', 'genotype': ['T']}]
            },
            mock_verify_snp_output=[
                (None, None, 'i123')
            ],
            expected_output=(
                    [{'rsid': 'rs4', 'genotype': ['T']}],
                    [],
                    [],
                    ['i123']
            )
        )
    ])
    @patch.object(DbSNPHandler, 'get_sampled_rsids')
    @patch.object(DbSNPHandler, 'verify_genome')
    @patch.object(DbSNPHandler, 'verify_snp')
    def test_verify_snps(self, mock_verify_snp: mock.MagicMock, mock_verify_genome: mock.MagicMock,
                         mock_get_sampled_rsids: mock.MagicMock, input_data: VerifySNPsTestData) -> None:

        scorer = DbSNPHandler(config={})

        mock_get_sampled_rsids.return_value = input_data.mock_get_sampled_rsids_output
        mock_verify_genome.return_value = input_data.mock_verify_genome_output
        mock_verify_snp.side_effect = input_data.mock_verify_snp_output

        result = scorer.verify_snps(input_data.input_df)

        mock_get_sampled_rsids.assert_called_once_with(input_data.input_df)
        mock_verify_genome.assert_called_once_with(input_data.mock_get_sampled_rsids_output)

        rsid_list = [item['rsid'] for item in input_data.mock_verify_genome_output['invalid']]
        genotype_list = [''.join(item['genotype']) for item in input_data.mock_verify_genome_output['invalid']]

        for rsid, genotype in zip(rsid_list, genotype_list):
            mock_verify_snp.assert_any_call(rsid, genotype)

        assert result == input_data.expected_output
