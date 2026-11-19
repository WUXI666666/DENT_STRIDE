from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import yaml
from common import ROOT, OLD, PROJECT, MODELS, NAMES, ANN, TRAIN_ANN, sha, write, now


def load(path):
    path = Path(path)
    obj = yaml.safe_load(path.read_text()) or {}
    result = {}
    def merge(a, b):
        for k, v in b.items():
            a[k] = merge(a.get(k, {}), v) if isinstance(v, dict) else deepcopy(v)
        return a
    for include in obj.pop('__include__', []):
        merge(result, load(path.parent/include))
    return merge(result, obj)


def main():
    if (ROOT/'protocol.json').exists():
        raise RuntimeError('Prepared experiment already exists; refuse to overwrite.')
    (ROOT/'configs').mkdir(parents=True, exist_ok=True)
    mobile = load('/home/wuxi/Dent-FINE-paper/materials/cmig_selected_configs_multiseed_2026-10-11/configs/mobile_proto_k4_tau0p50/seed_0.yml')
    manifest = {}
    for split, ann in [('train', TRAIN_ANN), ('val', ANN)]:
        data = json.loads(ann.read_text())
        assert sorted(c['id'] for c in data['categories']) == list(range(52))
        assert len({r['id'] for r in data['images']}) == len(data['images'])
        image_root = ann.parent
        missing = [r['file_name'] for r in data['images'] if not (image_root/r['file_name']).is_file()]
        assert not missing, missing[:3]
        manifest[split] = {'annotation': str(ann), 'sha256': sha(ann), 'images': len(data['images']),
                           'boxes': len(data['annotations']), 'label_map': data['categories']}
        labels = {}
        for a in data['annotations']:
            labels.setdefault(a['image_id'], []).append(a)
        assert len({Path(i['file_name']).name for i in data['images']})==len(data['images'])
        for image in data['images']:
            original_filename = Path(image['file_name'])
            filename = Path(original_filename.name)
            image_dir = ROOT/'dataset_yolo/images'/split
            label_dir = ROOT/'dataset_yolo/labels'/split
            image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
            image_link=image_dir/filename
            if image_link.is_symlink():
                assert image_link.resolve()==(image_root/original_filename).resolve()
            else:
                image_link.symlink_to(image_root/original_filename)
            rows = []
            for a in labels.get(image['id'], []):
                x, y, w, h = a['bbox']; iw, ih = image['width'], image['height']
                assert w > 0 and h > 0
                rows.append(f"{a['category_id']} {(x+w/2)/iw:.10f} {(y+h/2)/ih:.10f} {w/iw:.10f} {h/ih:.10f}")
            # Generated label files only; the canonical COCO annotations are immutable.
            (label_dir/filename.with_suffix('.txt')).write_text('\n'.join(rows)+'\n')
    assert manifest['train']['label_map'] == manifest['val']['label_map']
    names = {c['id']: c['name'] for c in manifest['train']['label_map']}
    data_yaml = {'path': str(ROOT/'dataset_yolo'), 'train': 'images/train', 'val': 'images/val', 'nc': 52, 'names': names}
    (ROOT/'configs/mobile_fdi52.yaml').write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    configurations = {}
    for family in ['deim_s', 'rtdetrv2_s', 'dome_detr_s']:
        if family == 'dome_detr_s':
            original = PROJECT/'third_party/dome_detr/configs/dome/Dome-S-ViewNet3-DSLR-FDI52.yml'
        else:
            original = OLD/'configs'/f'{family}_phone.yml'
        cfg = load(original)
        cfg.update(eval_spatial_size=[1024,1024], epoches=160, checkpoint_freq=1000000,
                   seed=0, resume=None, tuning=None, output_dir=str(ROOT/'runs'/family/'seed_0'))
        for split in ['train_dataloader','val_dataloader']:
            cfg[split]['dataset']['transforms'] = deepcopy(mobile[split]['dataset']['transforms'])
            # DOME does not implement Mosaic. It is disabled in the common recipe.
            cfg[split]['dataset']['transforms']['ops'] = [op for op in cfg[split]['dataset']['transforms']['ops'] if op['type']!='Mosaic']
            if 'policy' in cfg[split]['dataset']['transforms']:
                policy = cfg[split]['dataset']['transforms']['policy']
                policy['ops'] = [x for x in policy.get('ops',[]) if x!='Mosaic']
            cfg[split]['dataset']['transforms'].pop('mosaic_prob',None)
            cfg[split]['dataset']['ann_file'] = str(TRAIN_ANN if split=='train_dataloader' else ANN)
            cfg[split]['dataset']['img_folder'] = str((TRAIN_ANN if split=='train_dataloader' else ANN).parent)
            cfg[split]['num_workers'] = 4
            cfg[split]['total_batch_size'] = 8 if split=='train_dataloader' else 16
        collate = cfg['train_dataloader']['collate_fn']
        collate.update(base_size=1024, base_size_repeat=None, mixup_prob=0.0, stop_epoch=130)
        if family!='deim_s':
            cfg['train_dataloader']['dataset']['transforms']['policy']['epoch'] = 130
        if family=='rtdetrv2_s':
            cfg['train_dataloader']['collate_fn'] = {'type':'BatchImageCollateFunction', 'scales':None, 'stop_epoch':130}
        if family=='dome_detr_s':
            collate.pop('mixup_prob', None)
            collate['mwas_window_size'] = 16
            cfg['HGNetv2']['local_model_dir'] = str(PROJECT/'third_party/dome_detr/weight/hgnetv2')+'/'
            # Keep the original 8x8 MWAS window grid: (640/8)/10 == (1024/8)/16.
            # Native code requires feature dimensions divisible by the window.
            cfg['HybridEncoder']['mwas_window_size'] = 16
            cfg['train_dataloader']['total_batch_size'] = 4
            cfg['val_dataloader']['total_batch_size'] = 4
            cfg['DomeTransformer']['max_num_select'] = 1500
            cfg['DomeTransformer']['min_num_select'] = 300
            # DOME Compose uses an integer stop epoch rather than DEIM's list.
            cfg['train_dataloader']['dataset']['transforms']['policy']['epoch'] = 130
        path = ROOT/'configs'/f'{family}_phone.yml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        configurations[family+'_phone']={'path':str(path),'sha256':sha(path),
                                         'origin':str(original),'origin_sha256':sha(original)}
    old_protocol = json.loads((OLD/'protocol.json').read_text())
    protocol = {'created_at':now(),'seed':0,'seeds':[0],'input':[1024,1024],'epochs':160,
                'models':old_protocol['models'],'model_order':MODELS,'model_names':NAMES,
                'configs':configurations,'datasets':manifest,
                'selection':'Maximum validation COCO bbox AP, earliest epoch on exact ties',
                'storage':'One best weight per model; stage rollback snapshots in RAM; atomic temporary replacement only',
                'scope':'Eight framework baselines on the unchanged Mobile internal-validation split. No manuscript updates.',
                'augmentation':'No flips, no Mosaic, no MixUp. DEIM/RTv2/DOME share photo/crop/zoom policy; YOLO frameworks retain native color/affine transforms. Faster R-CNN uses fixed-square resize.',
                'pretraining':'Native ImageNet backbone for DEIM/RTv2/DOME; native COCO detector for YOLO/CAF/RT-DETR-L/Faster R-CNN. No prior dental fine-tuning.',
                'matching_limit':'Resolution-, split-, seed- and budget-aligned framework comparison, not an isolated module ablation; native losses/optimizers/pretraining remain different.',
                'dome_history':'Old phone tuned301 output was 40-epoch DSLR-checkpoint tuning; this run is fresh from ImageNet HGNetv2 backbone, 160 epochs, native dynamic 300..1500 queries.',
                'dome_resolution_adaptation':'Native MWAS window 10 at 640 is incompatible with feature width 128 at input 1024; set window=16 to preserve the original 8x8 window grid, without editing source or selecting against validation outcomes.',
                'dome_batch':'Batch 8 exhausted 23.53 GiB in a no-checkpoint preflight; train/validation batch=4. Native optimizer, 160 epochs and model modules remain unchanged. Batch/optimizer-step counts are not fully matched across frameworks.',
                'evaluation':'Canonical COCO JSON; native postprocessing, maximum 300 predictions. Fixed endpoints score>=0.30, localization Hungarian IoU>=0.50.',
                'disk_min_free_GiB':3,'gpu':0,'gpu1_policy':'Do not interfere with another user process.'}
    write(ROOT/'protocol.json',protocol)
    print(json.dumps({'root':str(ROOT),'models':MODELS,'train_images':manifest['train']['images'],
                      'val_images':manifest['val']['images'],'configs':configurations},indent=2))


if __name__=='__main__':main()
