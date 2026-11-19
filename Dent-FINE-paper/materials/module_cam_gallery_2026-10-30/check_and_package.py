"""Validate generated outputs and package only task-owned material, no weights."""
from pathlib import Path
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
FIG=HERE/'figures'
BEST=['Recommended_CMFA_HiResCAM','Recommended_HOOA_HiResCAM',
      'Recommended_Prototype_mechanism','Recommended_FinalMobile_modules_HiResCAM']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    captions=json.loads((HERE/'figure_captions.json').read_text())
    captions.update(json.loads((HERE/'final_mobile_captions.json').read_text()))
    manifest=json.loads((HERE/'figure_manifest.json').read_text())
    manifest.update(json.loads((HERE/'final_mobile_figure_manifest.json').read_text()))
    (HERE/'all_figure_captions.json').write_text(json.dumps(captions,indent=2))
    (HERE/'all_figure_manifest.json').write_text(json.dumps(manifest,indent=2))
    recommendation=json.loads((HERE/'recommendations.json').read_text())
    for group,r in recommendation.items():r['diagnostics']=[v for v in r['diagnostics'] if v['case_id'] in r['case_ids']]
    (HERE/'recommendations.json').write_text(json.dumps(recommendation,indent=2))
    checks=[]
    for name in manifest:
        png=Image.open(FIG/f'{name}.png');png.verify()
        png=Image.open(FIG/f'{name}.png');assert png.width==5010 and png.info['dpi'][0]>599
        ET.parse(FIG/f'{name}.svg')
        assert (FIG/f'{name}.pdf').read_bytes().startswith(b'%PDF-')
        assert (FIG/f'{name}_preview.png').exists()
        checks.append({'figure':name,'PNG_dimensions':[png.width,png.height],'DPI':png.info['dpi'][0]})
    plan=json.loads((HERE/'gallery_plan.json').read_text());weights=[]
    for group in plan['groups']:
        for side in ['before','after']:
            source=group[side];actual=sha(source['checkpoint']);assert actual==source['checkpoint_sha256']
            records=json.loads((HERE/f'{group["name"]}_{side}_capture.json').read_text())
            assert len(records['records'])==7 and records['passive_hook_logit_error']<1e-5
            for row in records['records']:
                assert row['bbox_max_normalized_error']<.001
                assert row['query_top1_category']==row['live_prediction']['category_id']
                assert row['live_prediction']['score']>=.30
                a=np.load(row['raw_maps'])
                assert all(np.isfinite(a[k]).all() for k in a.files)
            weights.append({'checkpoint':source['checkpoint'],'sha256':actual,'unchanged':True})
    final=json.loads((HERE/'final_mobile_capture.json').read_text())
    assert final['passive_hook_logit_error']<1e-5
    actual=sha(final['checkpoint'])
    prior=HERE.parent/'cmig_selected_seed_revision_2026-10-12/prediction'
    provenance=json.loads((prior/'mechanism_provenance.json').read_text())
    assert provenance['checkpoint_sha256']==actual
    bank=np.load(prior/'prototype_bank.npz');p=bank['normalized_prototypes'].astype(np.float64)
    p/=np.linalg.norm(p,axis=-1,keepdims=True)
    pair=p@p.swapaxes(-1,-2);v=pair[:,np.triu_indices(4,1)[0],np.triu_indices(4,1)[1]]
    v=v[bank['train_support']>0]
    caution={'checkpoint_sha256':actual,'supported_classes':int((bank['train_support']>0).sum()),
        'within_class_pair_cosine_mean':float(v.mean()),'within_class_pair_cosine_min':float(v.min()),
        'interpretation':'The four within-class prototypes are nearly collinear. This does not support distinct multimodal prototype specialization. Auxiliary-supervision learning gains and multimodal diversity are different claims.'}
    (HERE/'prototype_diversity_check.json').write_text(json.dumps(caution,indent=2))
    report={'figure_count':len(manifest),'controlled_case_count':21,'final_model_case_count':len(final['records']),
        'weights_unchanged':weights,'final_checkpoint_sha256':actual,'validated_files':checks,
        'all_native_predictions_top1_and_score_checked':True,'all_passive_hooks_forward_invariant':True,
        'CAMs_GT_masked_or_manually_painted':False,'original_source_or_manuscripts_modified':False,
        'training_started':False,'GPU_workloads_interrupted':False,'device_used':'CPU FP32'}
    (HERE/'verification_report.json').write_text(json.dumps(report,indent=2))
    concise={
    BEST[0]:(
        'Query-specific visualization of CMFA on DSLR validation images. Columns show reference annotations, control CAM, '
        'CMFA CAM, and predictions from separately trained models without and with CMFA. The paired runs use AdamW and seed 0. '
        'The examples correct FDI 55 to 16, 32 to 72, and 81 to 41, respectively. Teal boxes indicate ground truth and orange '
        'boxes indicate predictions. CAMs explain the reference-FDI logit at the finest encoder output using positive '
        'elementwise gradient-times-activation. Each map is normalized by its full-image maximum before identical cropping; '
        'colors show relative spatial attribution, not comparable magnitudes across models. Predictions use score≥0.30 and '
        'one-to-one localization matching at IoU≥0.50. The selected cases complement the controlled quantitative evaluation.'),
    BEST[1]:(
        'Effect of the high-pass and second-order mechanisms in HOOA on a Mobile validation example. The reduced model '
        'replaces the second-order interaction with the context branch and removes the high-pass contribution; other HOOA '
        'branches remain active. Both models are independently trained at seed 0 with prototype temperature 0.20. Complete '
        'HOOA corrects a primary FDI 53 prediction to the reference permanent FDI 13. CAMs target the same reference-FDI '
        'class at the independently matched native query in each model. Teal boxes denote ground truth and orange boxes '
        'denote predictions. Maps use positive elementwise gradient-times-activation with per-map full-image maximum '
        'normalization. This is an illustrative correction rather than a correction-frequency estimate.'),
    BEST[2]:(
        'Prototype auxiliary supervision and FDI identification on Mobile validation images. Separately trained seed-0 '
        'models without prototype supervision and with four prototypes per class at temperature 0.50 are compared. The '
        'examples correct FDI 11 to 51, 63 to 62, and 83 to 42. Prototype CAMs explain the actual auxiliary class score '
        'using positive elementwise gradient-times-activation; they do not explain an additional inference classification '
        'path. Prototype score fusion at inference is disabled. The final column shows the five highest auxiliary scores '
        'among classes with training support, computed by LogSumExp aggregation of temperature-scaled query–prototype '
        'cosine similarities; the reference class is teal. Scores are unnormalized and are not probabilities. These '
        'examples illustrate identity supervision, not specialization of the four prototype slots.'),
    BEST[3]:(
        'Module-specific spatial attributions in the final DENT-STRIDE-Mobile model. All panels within a row use the '
        'same image, trained EMA checkpoint (seed 2, four prototypes per class, temperature 0.50), and native query. '
        'Detection CAM, CMFA CAM, and HOOA edge CAM explain the same reference-FDI detection logit at the finest encoder '
        'output, finer top-down CMFA output, and high-pass branch tensor, respectively. Prototype CAM explains the '
        'auxiliary prototype class score. Positive elementwise gradient-times-activation is normalized separately by '
        'each full-image maximum, so colors do not compare contribution strength across modules. Orange boxes are '
        'predictions and dashed teal boxes are ground truth. These mechanism illustrations complement, but do not '
        'replace, independently retrained module ablations.')}
    common=('CAM interpretation: gradients do not traverse the native detached query-content initialization and therefore '
        'describe the remaining differentiable paths. Bilinear interpolation is for display only. CAMs are not native '
        'attention values, segmentation masks, probabilities, or proof of causal or clinical effectiveness. No target '
        'mask is used to calculate the attribution. The same limitations apply to all four figures above.')
    txt='\n\n'.join(name+'\n'+concise[name] for name in BEST)+'\n\n'+common+'\n'
    (HERE/'Recommended_captions_en.txt').write_text(txt)
    alltxt='\n\n'.join(name+'\n'+caption for name,caption in captions.items())+'\n'
    (HERE/'All_captions_en.txt').write_text(alltxt)
    # The small PNG pack is the convenient handoff; SVG/PDF remain available individually.
    for zname,formats in [('Recommended_PNG.zip',['png']),('Recommended_CAM_figures.zip',['png','pdf','svg'])]:
        with zipfile.ZipFile(HERE/zname,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in BEST:
                for ext in formats:z.write(FIG/f'{name}.{ext}',f'{name}.{ext}')
                z.write(FIG/f'{name}_preview.png',f'{name}_preview.png')
            for fname in ['Recommended_captions_en.txt','图片推荐与使用说明.txt','verification_report.json','prototype_diversity_check.json']:
                if (HERE/fname).exists():z.write(HERE/fname,fname)
    with zipfile.ZipFile(HERE/'All_CAM_previews.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in manifest:z.write(FIG/f'{name}_preview.png',f'{name}_preview.png')
        for fname in ['All_captions_en.txt','图片推荐与使用说明.txt','all_figure_manifest.json']:
            if (HERE/fname).exists():z.write(HERE/fname,fname)
    print('VERIFIED',len(manifest),'figures; controlled cases 21; final cases',len(final['records']),flush=True)
    print('PROTOTYPE_DIVERSITY',json.dumps(caution),flush=True)
    print('PACKS',[(p.name,round(p.stat().st_size/1024**2,2)) for p in HERE.glob('*.zip')],flush=True)


if __name__=='__main__':main()
