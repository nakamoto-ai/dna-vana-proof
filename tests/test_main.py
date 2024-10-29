
import pytest
import os
from unittest import mock
from typing import Optional, Dict, Any
from dataclasses import dataclass

from my_proof.__main__ import change_and_delete_file_extension
from my_proof.__main__ import run as main_run


@pytest.fixture
def mock_config() -> Dict[str, Any]:
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


@dataclass
class ConfigEnvVars:
    user_email: Optional[str]
    token: Optional[str]
    expected_user_email: Optional[str]
    expected_token: Optional[str]


@pytest.mark.parametrize("env_vars", [
    ConfigEnvVars(user_email="test@example.com", token="test_token", expected_user_email="test@example.com", expected_token="test_token"),
    ConfigEnvVars(user_email="user@example.com", token="secret_token", expected_user_email="user@example.com", expected_token="secret_token"),
    ConfigEnvVars(user_email=None, token=None, expected_user_email=None, expected_token=None)
])
def test_env_vars(env_vars: ConfigEnvVars) -> None:
    assert env_vars.user_email == env_vars.expected_user_email
    assert env_vars.token == env_vars.expected_token


@dataclass
class FileChangeConfig:
    file_path: str
    new_extension: str
    expected_new_file_path: str


@pytest.mark.parametrize("config", [
    FileChangeConfig(file_path="test.txt", new_extension=".md", expected_new_file_path="test.md"),
    FileChangeConfig(file_path="document.pdf", new_extension=".docx", expected_new_file_path="document.docx"),
    FileChangeConfig(file_path="notes", new_extension=".txt", expected_new_file_path="notes.txt")
])
@mock.patch('os.rename')
@mock.patch('os.remove')
@mock.patch('os.path.exists')
def test_change_and_delete_file_extension(mock_exists: mock.MagicMock, mock_remove: mock.MagicMock,
                                          mock_rename: mock.MagicMock, config: FileChangeConfig) -> None:
    mock_exists.side_effect = [True, False]

    new_file_path = change_and_delete_file_extension(config.file_path, config.new_extension)

    mock_rename.assert_called_once_with(config.file_path, config.expected_new_file_path)
    mock_remove.assert_called_once_with(config.file_path)

    assert new_file_path == config.expected_new_file_path


@dataclass
class RunTestData:
    input_files_exist: bool
    file_list: list
    expected_exception: Exception | None


@pytest.mark.parametrize("test_data", [
    RunTestData(input_files_exist=False, file_list=[], expected_exception=FileNotFoundError),
    RunTestData(input_files_exist=True, file_list=['file1.raw'], expected_exception=None),
])
@mock.patch('os.path.exists')
@mock.patch('os.remove')
@mock.patch('os.rename')
@mock.patch('os.listdir')
@mock.patch('os.path.isdir')
@mock.patch('builtins.open', new_callable=mock.mock_open)
@mock.patch('json.dump')
@mock.patch('my_proof.proof.Proof.generate')
def test_run(mock_generate: mock.MagicMock, mock_json_dump: mock.MagicMock,
             mock_open: mock.MagicMock, mock_isdir: mock.MagicMock,
             mock_listdir: mock.MagicMock, mock_rename: mock.MagicMock,
             mock_remove: mock.MagicMock, mock_exists: mock.MagicMock,
             test_data: RunTestData) -> None:

    mock_isdir.return_value = test_data.input_files_exist
    mock_listdir.return_value = test_data.file_list

    mock_exists.side_effect = [True, False]

    if test_data.expected_exception:
        with pytest.raises(test_data.expected_exception):
            main_run()
    else:
        mock_generate.return_value.dict.return_value = {"valid": True, "score": 0.95}

        main_run()

        if test_data.input_files_exist:
            input_file = os.path.join('/input', test_data.file_list[0])
            new_input_file = os.path.splitext(input_file)[0] + '.txt'

            mock_rename.assert_called_once_with(input_file, new_input_file)

            mock_remove.assert_called_once_with(input_file)

            mock_generate.assert_called_once()
            mock_open.assert_called_once_with('/output/results.json', 'w')
            mock_json_dump.assert_called_once_with({"valid": True, "score": 0.95}, mock.ANY, indent=2)
        else:
            mock_rename.assert_not_called()
            mock_remove.assert_not_called()
            mock_generate.assert_not_called()
            mock_open.assert_not_called()
            mock_json_dump.assert_not_called()
