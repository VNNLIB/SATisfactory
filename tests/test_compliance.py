import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from vnntest.cli.main import app
from vnntest.core.compliance import (
    check_opset_versions, validate_opset_versions,
    check_vnnlib_versions, validate_vnnlib_versions,
    check_serialise_assignments, validate_serialise_assignments,
)
from vnntest.core.runner import run_solver_command, run_solver_on_test
from vnntest.models.test_case import TestCase, Verdict


@pytest.mark.parametrize('output', ['13\n19', '13\n19\n', '13\r\n19\r\n', '19\n19\n'])
def test_valid_opset_pairs(output):
    assert validate_opset_versions(output) is None


@pytest.mark.parametrize('output', [
    '', '13', '13 19', '13\n19\n20', '13\n\n19', '13\n19\n\n',
    'log\n13\n19', '13.0\n19.0', '-1\n19', '0\n19', '19\n13',
    ' 13\n19', '13\ntrue', '１３\n１９',
])
def test_invalid_opset_pairs(output):
    assert validate_opset_versions(output) is not None


@pytest.mark.parametrize('separator', ['\n', '\r\n'])
@pytest.mark.parametrize('final_newline', [False, True])
def test_opset_line_endings(separator, final_newline):
    output = f'13{separator}19' + (separator if final_newline else '')
    assert validate_opset_versions(output) is None


@pytest.mark.parametrize('separator', [
    '\v', '\f', '\r', '\x1c', '\x1d', '\x1e', '\x85', '\u2028', '\u2029',
])
@pytest.mark.parametrize('position', ['between', 'trailing'])
def test_rejects_non_newline_separators(separator, position):
    output = f'13{separator}19' if position == 'between' else f'13\n19{separator}'
    assert validate_opset_versions(output) is not None


def test_query_uses_shared_runner_and_ignores_stderr_logs():
    result = subprocess.CompletedProcess([], 0, '13\n19\n', 'diagnostic log\n')
    with patch('vnntest.core.compliance.run_solver_command', return_value=result) as run:
        assert check_opset_versions('solver', 3)[0]
    run.assert_called_once_with('solver', ['supports', '--onnx-opset-versions'], 3, preserve_newlines=True)


def test_query_nonzero_exit_does_not_pass_with_valid_stdout():
    result = subprocess.CompletedProcess([], 2, '13\n19\n', 'unknown command')
    with patch('vnntest.core.compliance.run_solver_command', return_value=result):
        passed, message = check_opset_versions('solver')
    assert not passed
    assert 'exit code 2' in message


@pytest.mark.parametrize('error', [FileNotFoundError('missing'), subprocess.TimeoutExpired('solver', 1)])
def test_query_execution_errors(error):
    with patch('vnntest.core.compliance.run_solver_command', side_effect=error):
        assert not check_opset_versions('solver', 1)[0]


def test_shared_runner_preserves_process_output():
    result = run_solver_command(sys.executable, ['-c',
        'import sys; print("13\\n19"); print("log", file=sys.stderr); sys.exit(2)'])
    assert result.stdout == '13\n19\n'
    assert result.stderr == 'log\n'
    assert result.returncode == 2


def test_verify_invocation_and_output_are_unchanged():
    test_case = TestCase(
        name='query', category='model', onnx_path=Path('model.onnx'),
        onnx_paths={'N': Path('model.onnx')},
        vnnlib_path=Path('query.vnnlib'), expected=Verdict.UNSAT,
    )
    result = subprocess.CompletedProcess([], 0, 'unsat\n', 'log\n')
    with patch('vnntest.core.runner.subprocess.run', return_value=result) as run:
        run_result = run_solver_on_test('solver', test_case, 3)
    assert run_result.verdict == Verdict.UNSAT
    assert run_result.stdout == 'unsat\n'
    assert run_result.stderr == 'log\n'
    run.assert_called_once_with(
        ['solver', 'verify', 'query.vnnlib', '--network', 'N=model.onnx', '--timeout', '3'],
        capture_output=True, text=True, timeout=3,
    )


@pytest.mark.parametrize(('stdout', 'returncode', 'expected_exit'), [
    ('13\n19\n', 0, 0), ('invalid', 0, 1), ('13\n19\n', 2, 1),
])
def test_compliance_cli(stdout, returncode, expected_exit):
    result = subprocess.CompletedProcess([], returncode, stdout, '')
    with patch('vnntest.core.compliance.run_solver_command', return_value=result):
        response = CliRunner().invoke(app, ['compliance', 'opset', '--solver', 'solver'])
    assert response.exit_code == expected_exit, response.output
    assert ('PASS:' if expected_exit == 0 else 'FAIL:') in response.output


@pytest.mark.parametrize('separator', ['\n', '\r\n'])
@pytest.mark.parametrize('final_newline', [False, True])
def test_vnnlib_line_endings(separator, final_newline):
    output = f'2.0{separator}2.3' + (separator if final_newline else '')
    assert validate_vnnlib_versions(output) is None


@pytest.mark.parametrize('output', [
    '2.0\n2.0', '2.9\n2.10', '2.99\n3.0', '02.09\n2.10',
])
def test_vnnlib_numeric_ordering(output):
    assert validate_vnnlib_versions(output) is None


@pytest.mark.parametrize('output', [
    '', '2.0', '2.0 2.3', '2.0\n2.3\n3.0', '2.0\n\n2.3',
    '2.0\n2.3\n\n', '2.0\r\n2.3\r\n\r\n',
    'log\n2.0\n2.3', ' 2.0\n2.3', '2.0\n2.3 ', '2.0\n2.3 log',
    'v2.0\n2.3', '2\n3', '2.\n2.3', '.0\n2.3', '-2.0\n2.3',
    '2.0.1\n2.3', '2.0-beta\n2.3', '２.０\n2.3',
    '2.10\n2.9', '3.0\n2.99', '\n2.0\n2.3', '2.0\n2.3\x00',
])
def test_invalid_vnnlib_pairs(output):
    assert validate_vnnlib_versions(output) is not None


@pytest.mark.parametrize('separator', [
    '\v', '\f', '\r', '\x1c', '\x1d', '\x1e', '\x85', '\u2028', '\u2029',
])
@pytest.mark.parametrize('position', ['between', 'trailing'])
def test_vnnlib_rejects_invalid_separators(separator, position):
    output = f'2.0{separator}2.3' if position == 'between' else f'2.0\n2.3{separator}'
    assert validate_vnnlib_versions(output) is not None


def test_vnnlib_query_uses_runner_and_ignores_stderr():
    result = subprocess.CompletedProcess([], 0, '2.0\n2.3\n', 'diagnostic log')
    with patch('vnntest.core.compliance.run_solver_command', return_value=result) as run:
        assert check_vnnlib_versions('solver', 3)[0]
    run.assert_called_once_with('solver', ['supports', '--vnnlib-versions'], 3, preserve_newlines=True)


@pytest.mark.parametrize(('error', 'message'), [
    (FileNotFoundError('missing'), 'Could not execute capability query:'),
    (PermissionError('denied'), 'Could not execute capability query:'),
    (UnicodeError('invalid encoding'), 'Could not execute capability query:'),
    (subprocess.TimeoutExpired('solver', 1), 'Capability query timed out after 1 seconds.'),
])
def test_vnnlib_execution_errors(error, message):
    with patch('vnntest.core.compliance.run_solver_command', side_effect=error):
        passed, reason = check_vnnlib_versions('solver', 1)
    assert not passed
    assert message in reason


@pytest.mark.parametrize(('stdout', 'returncode', 'expected_exit'), [
    ('2.0\n2.3\n', 0, 0), ('invalid', 0, 1), ('2.0\n2.3\n', 2, 1),
])
def test_vnnlib_cli(stdout, returncode, expected_exit):
    result = subprocess.CompletedProcess([], returncode, stdout, 'diagnostic')
    with patch('vnntest.core.compliance.run_solver_command', return_value=result) as run:
        response = CliRunner().invoke(app, ['compliance', 'vnnlib', '--solver', 'solver', '--timeout', '3'])
    run.assert_called_once_with('solver', ['supports', '--vnnlib-versions'], 3, preserve_newlines=True)
    assert response.exit_code == expected_exit, response.output
    assert 'Compliance: supports --vnnlib-versions' in response.output
    assert ('PASS:' if expected_exit == 0 else 'FAIL:') in response.output
    if returncode:
        assert 'exit code 2: diagnostic' in response.output


@pytest.mark.parametrize('value', ['true', 'false'])
@pytest.mark.parametrize('ending', ['', '\n', '\r\n'])
def test_valid_serialise_assignments(value, ending):
    assert validate_serialise_assignments(value + ending) is None


@pytest.mark.parametrize('output', [
    '', '\n', 'TRUE', 'FALSE', 'True', 'False', '1', '0', 'yes', 'null',
    ' true', 'true ', '\tfalse', 'false\t', 'true \n', '\ntrue',
    'true\nfalse', 'false\r\ntrue', 'true\n\n', 'false\r\n\r\n',
    'true\n\r\n', 'false\r\n\n', 'true false', 'true log', 'log\nfalse',
    'true\x00', 'true\u00a0', '"true"',
])
def test_invalid_serialise_assignments(output):
    assert validate_serialise_assignments(output) is not None


@pytest.mark.parametrize('separator', [
    '\v', '\f', '\r', '\x1c', '\x1d', '\x1e', '\x85', '\u2028', '\u2029',
])
@pytest.mark.parametrize('output', ['true{}', '{}false', 'true{}false'])
def test_serialise_assignments_rejects_invalid_separators(separator, output):
    assert validate_serialise_assignments(output.format(separator)) is not None


@pytest.mark.parametrize('value', ['true', 'false'])
def test_serialise_assignments_query_uses_runner_and_ignores_stderr(value):
    result = subprocess.CompletedProcess([], 0, value + '\n', 'diagnostic log')
    with patch('vnntest.core.compliance.run_solver_command', return_value=result) as run:
        assert check_serialise_assignments('solver', 3)[0]
    run.assert_called_once_with('solver', ['supports', '--serialise-assignments'], 3, preserve_newlines=True)


@pytest.mark.parametrize(('error', 'message'), [
    (FileNotFoundError('missing'), 'Could not execute capability query:'),
    (PermissionError('denied'), 'Could not execute capability query:'),
    (UnicodeError('invalid encoding'), 'Could not execute capability query:'),
    (subprocess.TimeoutExpired('solver', 1), 'Capability query timed out after 1 seconds.'),
])
def test_serialise_assignments_cli_execution_errors(error, message):
    with patch('vnntest.core.compliance.run_solver_command', side_effect=error):
        response = CliRunner().invoke(app, [
            'compliance', 'serialise-assignments', '--solver', 'solver', '--timeout', '1',
        ])
    assert response.exit_code == 1, response.output
    assert 'FAIL:' in response.output
    assert message in response.output


@pytest.mark.parametrize(('stdout', 'returncode', 'expected_exit'), [
    ('true\n', 0, 0), ('false\r\n', 0, 0), ('TRUE', 0, 1),
    ('true\nfalse', 0, 1), ('true\n', 2, 1), ('false', 2, 1),
])
def test_serialise_assignments_cli(stdout, returncode, expected_exit):
    result = subprocess.CompletedProcess([], returncode, stdout, 'diagnostic')
    with patch('vnntest.core.compliance.run_solver_command', return_value=result) as run:
        response = CliRunner().invoke(app, [
            'compliance', 'serialise-assignments', '--solver', 'vibecheck',
        ])
    run.assert_called_once_with('vibecheck', ['supports', '--serialise-assignments'], 10, preserve_newlines=True)
    assert response.exit_code == expected_exit, response.output
    assert 'Compliance: supports --serialise-assignments' in response.output
    assert ('PASS:' if expected_exit == 0 else 'FAIL:') in response.output
    if returncode:
        assert 'exit code 2: diagnostic' in response.output


@pytest.mark.parametrize(('check', 'output'), [
    (check_opset_versions, '13{ending}19{ending}'),
    (check_vnnlib_versions, '2.0{ending}2.3{ending}'),
    (check_serialise_assignments, 'true{ending}'),
])
@pytest.mark.parametrize(('ending', 'expected_pass'), [
    ('\n', True), ('\r\n', True), ('\r', False),
])
def test_compliance_preserves_actual_process_line_endings(check, output, ending, expected_pass):
    payload = output.format(ending=ending).encode('ascii')

    def run_python(exec_path, arguments, timeout, **kwargs):
        return run_solver_command(
            sys.executable, ['-c', f'import sys; sys.stdout.buffer.write({payload!r})'],
            timeout, **kwargs,
        )

    with patch('vnntest.core.compliance.run_solver_command', side_effect=run_python):
        assert check('solver')[0] is expected_pass


def test_default_runner_keeps_legacy_newline_normalisation():
    result = run_solver_command(sys.executable, [
        '-c', 'import sys; sys.stdout.buffer.write(b"unsat\\r\\n"); sys.stderr.buffer.write(b"log\\r")',
    ])
    assert result.stdout == 'unsat\n'
    assert result.stderr == 'log\n'
