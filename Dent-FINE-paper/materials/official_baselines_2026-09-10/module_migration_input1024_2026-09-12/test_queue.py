#!/usr/bin/env python3
"""CPU-only queue tests with fake GPU snapshots and mocked child processes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_queue as queue


def completed_fixture(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = b'fake-checkpoint-not-a-training-artifact'
    (path / 'best.pth').write_bytes(checkpoint)
    checksum = hashlib.sha256(checkpoint).hexdigest()
    rows = [{'epoch': index, 'test_coco_eval_bbox': [0.5 + index / 10000] + [0.0] * 11}
            for index in range(160)]
    (path / 'log.txt').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    best = rows[-1]['test_coco_eval_bbox'][0]
    queue.write_json(path / 'metrics_summary.json', {
        'epochs_completed': 160, 'final_epoch_zero_based': 159, 'best_epoch_zero_based': 159,
        'best_metrics': {'AP': best * 100}, 'checkpoint_sha256': checksum})
    queue.write_json(path / 'verified_coco_metrics.json', {
        'matches_selected_checkpoint': True, 'metrics_fraction': {'AP': best}, 'checkpoint_sha256': checksum})
    (path / 'COMPLETED').write_text('mock completed\n')


class FakeChild:
    def __init__(self, pid, returncode=0, on_poll=None):
        self.pid = pid
        self.returncode = returncode
        self.on_poll = on_poll
        self.wait_calls = []

    def poll(self):
        if self.on_poll is not None:
            callback, self.on_poll = self.on_poll, None
            callback()
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return self.returncode


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dent_queue_mock_')
        self.addCleanup(self.temp.cleanup)
        self.here = Path(self.temp.name)
        self.protocol = {'queue_order': ['plain', 'selected'], 'runs': {
            task: {'output': str(self.here / 'runs' / task)} for task in ['plain', 'selected']}}
        queue.write_json(self.here / 'protocol.json', self.protocol)
        self.preflight = {'passed': True,
                          'protocol_sha256': queue.sha256(self.here / 'protocol.json'),
                          'recommended_min_free_gpu_mib': 11520,
                          'memory_limit_mib': 10752}
        queue.write_json(self.here / 'preflight_validation.json', self.preflight)
        self.snapshot = {'index': 0, 'free_mib': 12000, 'used_mib': 12000, 'total_mib': 24564}

    def make_queue(self, **kwargs):
        options = dict(gpu_query=lambda: dict(self.snapshot), disk_free=lambda: 32 * 1024**3,
                       sleep=lambda seconds: None, poll_seconds=1)
        options.update(kwargs)
        return queue.SerialQueue(self.here, self.protocol, self.protocol['queue_order'], **options)

    def test_valid_finished_run_and_tampered_weight(self):
        output = self.protocol['runs']['plain']['output']
        completed_fixture(output)
        self.assertEqual(queue.verify_completed(output)['epochs_completed'], 160)
        (Path(output) / 'best.pth').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'SHA256'):
            queue.verify_completed(output)

    def test_duplicate_or_missing_epoch_is_rejected(self):
        output = Path(self.protocol['runs']['plain']['output'])
        completed_fixture(output)
        lines = (output / 'log.txt').read_text().splitlines()
        lines[50] = lines[49]
        (output / 'log.txt').write_text('\n'.join(lines) + '\n')
        with self.assertRaisesRegex(RuntimeError, 'contiguous'):
            queue.verify_completed(output)

    def test_nonbest_epoch_and_second_weight_are_rejected(self):
        output = Path(self.protocol['runs']['plain']['output'])
        completed_fixture(output)
        summary = queue.read_json(output / 'metrics_summary.json')
        summary['best_epoch_zero_based'] = 158
        queue.write_json(output / 'metrics_summary.json', summary)
        with self.assertRaisesRegex(RuntimeError, 'earliest global'):
            queue.verify_completed(output)
        completed_fixture(output)
        (output / 'last.pth').write_bytes(b'extra')
        with self.assertRaisesRegex(RuntimeError, 'Best-only'):
            queue.verify_completed(output)

    def test_serial_success_and_owned_lock_inheritance(self):
        calls, children = [], []
        def popen(command, **kwargs):
            self.assertTrue(kwargs['start_new_session'])
            self.assertEqual(len(kwargs['pass_fds']), 1)
            self.assertIn('--memory-limit-mib=10752', command)
            if children:
                self.assertTrue(children[-1].wait_calls, 'Next task launched before prior child reaped')
            task = command[command.index('--task') + 1]
            completed_fixture(self.protocol['runs'][task]['output'])
            calls.append(task)
            child = FakeChild(900001 + len(calls))
            children.append(child)
            return child
        runner = self.make_queue(popen=popen)
        self.assertEqual(runner.run(), 0)
        self.assertEqual(calls, ['plain', 'selected'])
        state = queue.read_json(self.here / 'queue_state.json')
        self.assertEqual(state['state'], 'completed')
        self.assertEqual(state['completed'], calls)
        self.assertIsNone(state['child_pid'])
        self.assertEqual(state['memory_limit_mib'], 10752)
        self.assertEqual(state['min_free_mib'], 11520)

    def test_custom_memory_limit_is_forwarded(self):
        commands = []
        def popen(command, **kwargs):
            commands.append(command)
            task = command[command.index('--task') + 1]
            completed_fixture(self.protocol['runs'][task]['output'])
            return FakeChild(900060 + len(commands))
        runner = self.make_queue(popen=popen, memory_limit_mib=11000, min_free_mib=11768)
        self.assertEqual(runner.run(), 0)
        self.assertEqual(len(commands), 2)
        self.assertTrue(all('--memory-limit-mib=11000' in command for command in commands))

    def test_preflight_recommendation_and_cli_are_lower_bounds(self):
        for cli_min, recommended, expected in [(11520, 12288, 12288),
                                                (13000, 12288, 13000),
                                                (10000, 10000, 11520)]:
            with self.subTest(cli_min=cli_min, recommended=recommended):
                self.preflight['recommended_min_free_gpu_mib'] = recommended
                queue.write_json(self.here / 'preflight_validation.json', self.preflight)
                policy = queue.preflight_resource_policy(self.here, cli_min, 10752)
                self.assertEqual(policy['min_free_mib'], expected)
                self.assertEqual(policy['memory_limit_mib'], 10752)

    def test_larger_memory_limit_requires_its_own_headroom(self):
        policy = queue.preflight_resource_policy(self.here, 11520, 12000)
        self.assertEqual(policy['min_free_mib'], 12768)
        self.assertEqual(policy['preflight_memory_limit_mib'], 10752)
        with self.assertRaisesRegex(ValueError, 'below the validated'):
            queue.preflight_resource_policy(self.here, 11520, 9216)

    def test_preflight_protocol_hash_must_match(self):
        self.preflight['protocol_sha256'] = 'old-protocol-sha256'
        queue.write_json(self.here / 'preflight_validation.json', self.preflight)
        with patch.object(queue, 'HERE', self.here), patch.object(queue, 'SerialQueue') as constructor:
            with self.assertRaisesRegex(RuntimeError, 'protocol SHA256'):
                queue.main([])
        constructor.assert_not_called()
        self.assertFalse((self.here / 'queue_state.json').exists())

    def test_preflight_must_have_passed(self):
        self.preflight['passed'] = False
        queue.write_json(self.here / 'preflight_validation.json', self.preflight)
        with self.assertRaisesRegex(RuntimeError, 'has not passed'):
            queue.preflight_resource_policy(self.here, 11520, 10752)
        (self.here / 'preflight_validation.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'Complete preflight_validation'):
            queue.preflight_resource_policy(self.here, 11520, 10752)

    def test_invalid_preflight_resource_values_are_rejected(self):
        for field in ['recommended_min_free_gpu_mib', 'memory_limit_mib']:
            for invalid in [None, 0, -1, True, '11520', 11520.0]:
                with self.subTest(field=field, invalid=invalid):
                    report = dict(self.preflight, **{field: invalid})
                    queue.write_json(self.here / 'preflight_validation.json', report)
                    with self.assertRaisesRegex(ValueError, 'positive integer'):
                        queue.preflight_resource_policy(self.here, 11520, 10752)

    def test_main_passes_validated_resource_policy_to_queue(self):
        self.preflight['recommended_min_free_gpu_mib'] = 12288
        queue.write_json(self.here / 'preflight_validation.json', self.preflight)
        with patch.object(queue, 'HERE', self.here), patch.object(queue, 'SerialQueue') as constructor:
            runner = constructor.return_value
            runner.state = {}
            runner.run.return_value = 0
            self.assertEqual(queue.main([]), 0)
            self.assertEqual(constructor.call_args.kwargs['min_free_mib'], 12288)
            self.assertEqual(constructor.call_args.kwargs['memory_limit_mib'], 10752)
            self.assertEqual(runner.state['resource_policy']['protocol_sha256'],
                             queue.sha256(self.here / 'protocol.json'))
            runner.run.assert_called_once_with()

    def test_memory_limit_must_be_below_launch_threshold(self):
        with self.assertRaisesRegex(ValueError, 'below min_free_mib'):
            self.make_queue(memory_limit_mib=11520, min_free_mib=11520)

    def test_skip_requires_full_verification(self):
        for entry in self.protocol['runs'].values():
            completed_fixture(entry['output'])
        with patch.object(subprocess, 'Popen', side_effect=AssertionError('No child expected')):
            runner = self.make_queue(popen=subprocess.Popen)
            self.assertEqual(runner.run(), 0)
        self.assertEqual(runner.state['skipped'], ['plain', 'selected'])
        (Path(self.protocol['runs']['plain']['output']) / 'best.pth').write_bytes(b'bad')
        with self.assertRaisesRegex(RuntimeError, 'SHA256'):
            self.make_queue(popen=lambda *a, **kw: self.fail('No child expected')).run()

    def test_waits_for_free_memory_not_for_idle_gpu(self):
        snapshots = iter([dict(self.snapshot, free_mib=8000, utilization_percent=99),
                          dict(self.snapshot, free_mib=12000, utilization_percent=99)])
        sleeps = []
        runner = self.make_queue(gpu_query=lambda: next(snapshots), sleep=sleeps.append)
        runner.wait_for_resources('plain')
        self.assertEqual(sleeps, [1])
        self.assertEqual(runner.state['gpu_snapshot']['utilization_percent'], 99)

    def test_low_disk_fails_before_launch(self):
        runner = self.make_queue(disk_free=lambda: 4 * 1024**3,
                                 popen=lambda *a, **kw: self.fail('Must not launch'))
        with self.assertRaisesRegex(RuntimeError, '5 GiB'):
            runner.run()
        self.assertEqual(runner.state['state'], 'failed')

    def test_nonzero_exit_stops_queue_and_only_signals_owned_group(self):
        calls = []
        def popen(command, **kwargs):
            calls.append(command)
            return FakeChild(900010, returncode=2)
        runner = self.make_queue(popen=popen)
        with patch.object(os, 'killpg') as killpg:
            with self.assertRaisesRegex(RuntimeError, 'exit status 2'):
                runner.run()
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(call.args[0] == 900010 for call in killpg.call_args_list))
        self.assertEqual(runner.state['completed'], [])

    def test_zero_exit_without_complete_artifacts_fails(self):
        calls = []
        def popen(command, **kwargs):
            calls.append(command)
            return FakeChild(900020)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete run'):
            self.make_queue(popen=popen).run()
        self.assertEqual(len(calls), 1)

    def test_signal_forwarding_reaps_only_owned_child(self):
        runner = self.make_queue()
        child = FakeChild(900030, returncode=None,
                          on_poll=lambda: runner.signal_handler(signal.SIGTERM, None))
        runner.popen = lambda *a, **kw: child
        with patch.object(os, 'killpg') as killpg:
            self.assertEqual(runner.run(), 128 + signal.SIGTERM)
        self.assertEqual([call.args[0] for call in killpg.call_args_list], [900030, 900030])
        self.assertEqual(child.wait_calls, [30])
        self.assertEqual(runner.state['state'], 'interrupted')
        self.assertIsNone(runner.state['child_pid'])

    def test_refuses_unowned_process_group(self):
        runner = self.make_queue()
        runner.child = FakeChild(900040)
        runner.child_pgid = os.getpgrp()
        with patch.object(os, 'killpg') as killpg:
            with self.assertRaisesRegex(RuntimeError, 'not created'):
                runner.terminate_owned_child()
        killpg.assert_not_called()

    def test_duplicate_queue_lock_does_not_replace_status(self):
        marker = {'state': 'existing queue'}
        queue.write_json(self.here / 'queue_state.json', marker)
        with queue.exclusive_queue_lock(self.here):
            with self.assertRaisesRegex(RuntimeError, 'owns the GPU0 queue lock'):
                self.make_queue().run()
        self.assertEqual(queue.read_json(self.here / 'queue_state.json'), marker)

    def test_termination_timeout_escalates_only_owned_group(self):
        runner = self.make_queue()
        child = FakeChild(900050)
        child.wait = unittest.mock.Mock(side_effect=[subprocess.TimeoutExpired('mock', 30), 0])
        runner.child, runner.child_pgid = child, child.pid
        with patch.object(os, 'killpg') as killpg:
            runner.terminate_owned_child()
        self.assertEqual([call.args[1] for call in killpg.call_args_list],
                         [signal.SIGTERM, signal.SIGKILL, signal.SIGKILL])
        self.assertTrue(all(call.args[0] == 900050 for call in killpg.call_args_list))


if __name__ == '__main__':
    unittest.main(verbosity=2)
