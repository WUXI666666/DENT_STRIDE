"""CPU-only orchestration, immutability, recipe and archive tests; no training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import yaml
import run_search as s


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous = s.e.read_json(s.PREVIOUS / 'manifest.json')
        cls.results = {r['job_id']: r for r in s.e.read_json(s.PREVIOUS / 'results_summary.json')}

    def test_core_is_six_unique_profiles_including_reused(self):
        for task in s.TASKS:
            recipes = s.core_recipes(task) + [s.e.augmentation('fixed640', mixup=.5)]
            self.assertEqual(len({json.dumps(r, sort_keys=True) for r in recipes}), 6)
            self.assertEqual({(r['profile'], r['mixup']) for r in recipes},
                             {(p, m) for p in ['legacy640_multiscale', 'fixed640', 'fixed1024'] for m in [0, .5]})
            self.assertTrue(all(r['crop'] == .8 and r['zoom'] == .5 and r['mosaic'] == 0 for r in recipes))

    def test_legacy_a07_uses_its_native_no_mixup(self):
        self.assertEqual(s.core_recipes('phone_a07_proto_aux')[0]['mixup'], 0)
        for task in s.TASKS:
            base = yaml.safe_load(Path(self.previous['tasks'][task]['base_config']).read_text())
            native = s.e.apply_augmentation(base, s.core_recipes(task)[0])
            self.assertEqual(base['train_dataloader']['collate_fn'], native['train_dataloader']['collate_fn'])
            self.assertEqual(native['DEIMCriterion']['weight_dict']['loss_arch'],
                             .1 if task == 'phone_a07_proto_aux' else .25)
            self.assertEqual(base['DFINETransformer'], native['DFINETransformer'])
            self.assertEqual(base['optimizer'], native['optimizer'])
            if task == 'phone_a07_proto_aux':
                self.assertTrue(native['save_best_only'])

    def test_reused_config_matches_expected(self):
        for task, jobid in s.STARTING_IDS.items():
            r = self.results[jobid]
            base = yaml.safe_load(Path(self.previous['tasks'][task]['base_config']).read_text())
            cfg = yaml.safe_load(Path(r['config']).read_text())
            self.assertEqual(s.e.config_semantics(cfg), s.e.config_semantics(s.e.apply_augmentation(base, r['augmentation'])))

    def test_shapes_and_model_invariants(self):
        for task in s.TASKS:
            base = yaml.safe_load(Path(self.previous['tasks'][task]['base_config']).read_text())
            for recipe in s.core_recipes(task):
                cfg = s.e.apply_augmentation(base, recipe)
                size = 1024 if recipe['profile'] == 'fixed1024' else 640
                self.assertEqual(cfg['eval_spatial_size'], [size, size])
                self.assertEqual(s.e.transform(cfg, 'Resize')['size'], [size, size])
                self.assertEqual(s.e.transform(cfg, 'Resize', val=True)['size'], [size, size])
                self.assertEqual(cfg['train_dataloader']['collate_fn']['base_size'], size)
                self.assertEqual(cfg['train_dataloader']['collate_fn']['base_size_repeat'],
                                 20 if recipe['profile'] == 'legacy640_multiscale' else None)
                for key in base.keys() - {'train_dataloader', 'val_dataloader', 'eval_spatial_size'}:
                    self.assertEqual(base[key], cfg[key], (task, key))
                self.assertEqual(cfg['train_dataloader']['total_batch_size'], 8)
                self.assertEqual(cfg['val_dataloader']['total_batch_size'], 16)

    def test_fractional_mosaic_only_has_one_probability_gate(self):
        with tempfile.TemporaryDirectory() as td, patch.object(s, 'CONFIGS', Path(td)), patch.object(s, 'OUTPUTS', Path(td) / 'out'):
            job = s.make_job(self.previous, 'mosaic', 'phone_dualopt',
                             s.e.augmentation('fixed640', mosaic=.25), 'test')
            cfg = yaml.safe_load(Path(job['config']).read_text())
            self.assertEqual(cfg['train_dataloader']['dataset']['transforms']['mosaic_prob'], .25)
            self.assertEqual(s.e.transform(cfg, 'Mosaic')['probability'], 1)
            self.assertEqual(s.e.transform(cfg, 'Mosaic')['output_size'], 640)
            self.assertEqual(job['mosaic_gate']['active_epochs_zero_based'], [4, 69])

    def test_refinements_are_independent_single_factor_changes(self):
        with tempfile.TemporaryDirectory() as td, patch.object(s, 'CONFIGS', Path(td)), patch.object(s, 'OUTPUTS', Path(td) / 'out'):
            anchor = {'augmentation': s.e.augmentation('fixed1024', mixup=0)}
            jobs = s.refinement_jobs(self.previous, 'phone_a07_proto_aux', anchor)
            self.assertEqual(len(jobs), 3)
            self.assertEqual({s.e.diff(anchor['augmentation'], j['augmentation'])[0]['path'] for j in jobs},
                             {'crop', 'zoom', 'mosaic'})
            self.assertTrue(all(len(s.e.diff(anchor['augmentation'], j['augmentation'])) == 1 for j in jobs))

    def test_generated_configs_are_immutable(self):
        with tempfile.TemporaryDirectory() as td, patch.object(s, 'CONFIGS', Path(td)), patch.object(s, 'OUTPUTS', Path(td) / 'out'):
            s.make_job(self.previous, 'immutable', 'phone_dualopt', s.e.augmentation('fixed640'), 'test')
            with self.assertRaises(RuntimeError):
                s.make_job(self.previous, 'immutable', 'phone_dualopt', s.e.augmentation('fixed1024'), 'test')

    def test_memory_scheduler_does_not_block_other_ready_trials(self):
        high = {'task': 'phone_a07_proto_aux', 'augmentation': s.e.augmentation('fixed1024')}
        low = {'task': 'phone_dualopt', 'augmentation': s.e.augmentation('fixed640')}
        self.assertIs(s.choose_ready([high, low], 11778), low)
        self.assertIsNone(s.choose_ready([high], 11778))
        self.assertEqual(s.minimum_memory(high), 13500)
        high['augmentation']['mosaic'] = .25
        self.assertEqual(s.minimum_memory(high), 18000)

    def test_checked_copy_never_overwrites_different_files(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / 'a', Path(td) / 'b'
            a.write_bytes(b'original'); b.write_bytes(b'different')
            with self.assertRaises(RuntimeError):
                s.checked_copy(a, b)
            self.assertEqual(b.read_bytes(), b'different')

    def test_zip_hash_verified_and_immutable(self):
        with tempfile.TemporaryDirectory() as td, patch.object(s, 'BUNDLE', Path(td)):
            root = Path(td)
            (root / 'README.md').write_text('test')
            model = root / 'starting_models' / 'test'
            model.mkdir(parents=True)
            (model / 'best.pth').write_bytes(b'test weights')
            entry = {'bundle_relative_path': 'starting_models/test',
                     'files_sha256': {'best.pth': s.e.sha(model / 'best.pth')}, 'assets': {}}
            path = s.create_zip('test.zip', [entry])
            with zipfile.ZipFile(path) as handle:
                self.assertEqual(handle.read('starting_models/test/best.pth'), b'test weights')
            self.assertTrue(s.e.read_json(Path(path).with_suffix('.zip.sha256.json'))['model_artifact_hashes_verified'])
            with self.assertRaises(RuntimeError):
                s.create_zip('test.zip', [entry])

    def test_mocked_full_queue_independent_selection_and_24_new_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            configs = root / 'configs'; configs.mkdir()
            outputs = root / 'outputs'; outputs.mkdir()
            bundle = root / 'bundle'; bundle.mkdir()
            patches = [patch.object(s, 'ROOT', root), patch.object(s, 'CONFIGS', configs),
                       patch.object(s, 'OUTPUTS', outputs), patch.object(s, 'BUNDLE', bundle),
                       patch.object(s.e, 'ROOT', root), patch.object(s.e, 'OUTPUTS', outputs),
                       patch.object(s.e, 'CONFIGS', configs), patch.object(s.e, 'verify_sources'),
                       patch.object(s, 'verify_result'), patch.object(s.e, 'gpu_free', return_value=24000),
                       patch.object(s.e, 'stop_child'), patch.object(s.signal, 'signal')]
            for item in patches:
                item.start()
            try:
                manifest = {'tasks': self.previous['tasks'], 'reference_results': {}, 'initial_jobs': []}
                for task, jobid in s.STARTING_IDS.items():
                    manifest['reference_results'][task] = copy.deepcopy(self.results[jobid])
                for i in range(5):
                    for task in s.TASKS:
                        manifest['initial_jobs'].append(s.make_job(manifest, f'{task}_C{i+1:02}', task,
                                                                  s.core_recipes(task)[i], 'core_grid'))
                s.e.write_json(root / 'manifest.json', manifest)
                called = []
                best_core = {'phone_dualopt': 'C03', 'dslr_cmfa_dualopt': 'C02', 'phone_a07_proto_aux': 'C01'}

                def fake_run(manifest, job):
                    called.append(job)
                    r = {**copy.deepcopy(manifest['reference_results'][job['task']]), **job}
                    bonus = 2 if job['job_id'].endswith(best_core[job['task']]) else -1
                    if job['job_id'].endswith('R02_zoom0'):
                        bonus = 3
                    ap = r['metrics_percent']['AP'] + bonus
                    r['metrics_percent'] = {**r['metrics_percent'], 'AP': ap}
                    r['metrics_fraction'] = {**r['metrics_fraction'], 'AP': ap / 100}
                    return r

                def fake_archive(result, metadata, collection):
                    return {'task': result['task'], 'AP': result['metrics_percent']['AP']}

                with patch.object(s.e, 'run_job', side_effect=fake_run), \
                     patch.object(s, 'archive_model', side_effect=fake_archive), \
                     patch.object(s, 'create_zip', return_value='fake_final.zip'):
                    s.run_study()
                self.assertEqual(len(called), 24)
                self.assertEqual(len({j['job_id'] for j in called}), 24)
                self.assertFalse(any(j['task'] == 'dentex_spatial_shape' for j in called))
                self.assertEqual(s.e.read_json(root / 'queue_state.json')['state'], 'completed')
                self.assertEqual(len(s.e.read_json(root / 'results_summary.json')), 27)
                for task in s.TASKS:
                    core = s.e.read_json(root / f'{task}_core_selection.json')['selected']
                    final = s.e.read_json(root / f'{task}_selected_result.json')
                    self.assertTrue(core['job_id'].endswith(best_core[task]))
                    self.assertTrue(final['selected']['job_id'].endswith('R02_zoom0'))
                    self.assertEqual(final['candidate_count'], 9)
                    self.assertEqual(core['augmentation']['profile'], final['selected']['augmentation']['profile'])
            finally:
                for item in reversed(patches):
                    item.stop()


if __name__ == '__main__':
    unittest.main()
