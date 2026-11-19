#!/usr/bin/env python3
"""Permit trusted high-resolution dental images and their native augmentation crops.

Upstream DEIM already sets this PIL option in its dataset module. RT-DETRv2
does not, causing its unmodified RandomZoomOut/RandomIoUCrop pipeline to abort
on these existing photographs. No pixels, labels, random draws or image
transform parameters are changed by this process-local compatibility setting.
"""
import argparse
from PIL import Image
from prepare_comparison import ROOT, sha256, write_json


if __name__ == '__main__':
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--model', required=True)
    p.add_argument('--cohort', required=True)
    p.add_argument('--seed', type=int, required=True)
    args, _ = p.parse_known_args()
    assert args.model == 'rtdetrv2_s'
    Image.MAX_IMAGE_PIXELS = None
    output = ROOT / 'runs' / args.model / args.cohort / f'seed_{args.seed}'
    write_json(output / 'runtime_compatibility.json', {
        'adapter': 'run_official_safe_images.py', 'sha256': sha256(__file__),
        'PIL.Image.MAX_IMAGE_PIXELS': None,
        'scope': 'this process and its forked workers only; original trusted local dataset',
        'reason': 'Native augmentation crop exceeded Pillow default guard; DEIM already disables the same guard.',
        'no_changes': ['original images', 'annotations', 'augmentation operations or parameters',
                       'model', 'loss', 'optimizer', 'seed', 'evaluation'],
    })
    import run_official
    run_official.main()
