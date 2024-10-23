
import pytest
import os
import json
import logging
from unittest import mock
from unittest.mock import patch, MagicMock, mock_open

from my_proof.__main__ import load_config, change_and_delete_file_extension
from my_proof.__main__ import run as main_run
from my_proof.models.proof_response import ProofResponse

from .data_tables import ConfigEnvVars, FileChangeConfig, RunTestData


@pytest.fixture
def mock_config():
    """Fixture to provide a mock config."""
    return {
        'dlp_id': 1234,
        'use_sealing': False,
        'input_dir': '/mock/input',
        'user_email': 'test@example.com',
        'token': 'test_token',
        'key': 'test_key',
        'verify': 'test_verify',
        'endpoint': 'test_endpoint'
    }


@pytest.mark.parametrize("env_vars", [
    ConfigEnvVars(user_email="test@example.com", token="test_token", expected_user_email="test@example.com", expected_token="test_token"),
    ConfigEnvVars(user_email="user@example.com", token="secret_token", expected_user_email="user@example.com", expected_token="secret_token"),
    ConfigEnvVars(user_email=None, token=None, expected_user_email=None, expected_token=None)
])
def test_env_vars(env_vars):
    assert env_vars.user_email == env_vars.expected_user_email
    assert env_vars.token == env_vars.expected_token


@pytest.mark.parametrize("config", [
    FileChangeConfig(file_path="test.txt", new_extension=".md", expected_new_file_path="test.md"),
    FileChangeConfig(file_path="document.pdf", new_extension=".docx", expected_new_file_path="document.docx"),
    FileChangeConfig(file_path="notes", new_extension=".txt", expected_new_file_path="notes.txt")
])
@mock.patch('os.rename')
@mock.patch('os.remove')
@mock.patch('os.path.exists')
def test_change_and_delete_file_extension(mock_exists, mock_remove, mock_rename, config):
    mock_exists.side_effect = [True, False]

    new_file_path = change_and_delete_file_extension(config.file_path, config.new_extension)

    mock_rename.assert_called_once_with(config.file_path, config.expected_new_file_path)
    mock_remove.assert_called_once_with(config.file_path)

    assert new_file_path == config.expected_new_file_path


@pytest.mark.parametrize("test_data", [
    RunTestData(input_files_exist=False, file_list=[], expected_exception=FileNotFoundError),  # No input files
    RunTestData(input_files_exist=True, file_list=['file1.raw'], expected_exception=None),  # Input files exist
])
@mock.patch('os.path.exists')
@mock.patch('os.remove')
@mock.patch('os.rename')
@mock.patch('os.listdir')
@mock.patch('os.path.isdir')
@mock.patch('builtins.open', new_callable=mock.mock_open)
@mock.patch('json.dump')
@mock.patch('my_proof.proof.Proof.generate')
def test_run(mock_generate, mock_json_dump, mock_open, mock_isdir, mock_listdir, mock_rename, mock_remove, mock_exists,
             test_data: RunTestData):
    # Setup mocks
    mock_isdir.return_value = test_data.input_files_exist
    mock_listdir.return_value = test_data.file_list

    # Simulate the file exists before the deletion, and does not exist after deletion
    mock_exists.side_effect = [True, False]

    if test_data.expected_exception:
        with pytest.raises(test_data.expected_exception):
            main_run()
    else:
        mock_generate.return_value.dict.return_value = {"valid": True, "score": 0.95}

        # Run the function
        main_run()

        # Assertions
        if test_data.input_files_exist:
            input_file = os.path.join('/input', test_data.file_list[0])
            new_input_file = os.path.splitext(input_file)[0] + '.txt'

            # Ensure rename was called
            mock_rename.assert_called_once_with(input_file, new_input_file)

            # Ensure remove was called
            mock_remove.assert_called_once_with(input_file)

            # Ensure generate was called and result was written to the output
            mock_generate.assert_called_once()
            mock_open.assert_called_once_with('/output/results.json', 'w')
            mock_json_dump.assert_called_once_with({"valid": True, "score": 0.95}, mock.ANY, indent=2)
        else:
            mock_rename.assert_not_called()
            mock_remove.assert_not_called()
            mock_generate.assert_not_called()
            mock_open.assert_not_called()
            mock_json_dump.assert_not_called()
