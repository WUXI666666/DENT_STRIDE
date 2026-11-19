#!/usr/bin/env python3
"""Compare reused backbone files with official release bytes, without another copy."""
import hashlib
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from prepare_comparison import ROOT, sha256, write_json


if __name__ == '__main__':
    pairs = [
        (Path('/home/wuxi/RT-DETR-main/D-FINE/weight/hgnetv2/PPHGNetV2_B0_stage1.pth'),
         'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'),
        (ROOT / 'pretrained_cache/hub/checkpoints/ResNet18_vd_pretrained_from_paddle.pth',
         'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth'),
    ]
    results = []
    for path, url in pairs:
        h = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=30) as response:
            while True:
                chunk = response.read(1024*1024)
                if not chunk:
                    break
                h.update(chunk)
        local_hash = sha256(path)
        assert h.hexdigest() == local_hash, f'Official backbone checksum mismatch: {path}'
        results.append({'file': str(path), 'official_release_url': url, 'sha256': local_hash,
                        'exact_official_bytes': True, 'verified_at': datetime.now().astimezone().isoformat()})
        write_json(ROOT / 'verified_backbone_sources.json', results)
        print(json.dumps(results[-1]), flush=True)
