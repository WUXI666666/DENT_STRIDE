"""Render actual new-bank and new-forward diagnostics in the established style."""
from pathlib import Path
from hashlib import sha256
import importlib.util
import json
import numpy as np

DEST=Path(__file__).resolve().parent
BASE=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03/scripts/plot_mechanism_evidence.py')
spec=importlib.util.spec_from_file_location('mechanism_renderer',BASE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.DEST=DEST

def main():
    m.tensor_figures();m.prototypes();m.factor_confusion()
    prov=json.loads((DEST/'mechanism_provenance.json').read_text());iid=prov['selected_cases'][0]['image_id'];n=prov['captured_queries_total']
    bank=np.load(DEST/'prototype_bank.npz');p=bank['normalized_prototypes']
    intra=np.max(np.abs(p-p[:,:1,:]),axis=(1,2))
    support=bank['train_support'];supported=support>0
    diagnosis={'normalized_bank_shape':list(p.shape),'unsupported_FDI_codes':bank['fdi_codes'][~supported].tolist(),'maximum_absolute_within_class_prototype_difference':float(intra.max()),'within_class_prototypes_exactly_identical_all_supported':bool(np.all(intra[supported]==0)),'interpretation':'Near-unit cohesion describes nearly collinear within-class vectors, not evidence of distinct multi-modal prototypes or a demonstrated AP improvement.','checkpoint_sha256':prov['checkpoint_sha256'],'prototype_temperature':.50}
    (DEST/'prototype_representation_checks.json').write_text(json.dumps(diagnosis,indent=2))
    captions=json.loads((DEST/'figure_captions.json').read_text())
    captions[f'actual_CMFA_HOOA_image_{iid}']=f'Actual CMFA and HOOA forward-pass responses from the representative Mobile seed-2 checkpoint for the improvement image in Fig. 12A. The same image supplies every panel. CMFA gA/gB and the HOOA channel gate are 16 × 16 channel-index grids, not anatomical maps. Spatial and channel gates retain their sigmoid values on [0,1]. Local, context, high-pass, second-order, input, and output response maps summarize channel RMS, divided by their own 99th percentile and clipped to [0,1]. The tensors were captured in a CPU FP32 forward pass; passive hooks reconstruct native outputs and leave predictions unchanged. These are internal tensors, not Grad-CAM or causal performance evidence.'
    captions['current_Mobile_prototype_diagnostics']='Prototype representation diagnostics from the representative Mobile seed-2 checkpoint (K = 4, τ = 0.50). (A) Cosine similarity between renormalized averages of each class’s four normalized prototypes, ordered by FDI quadrant. (B) Within-class cohesion, defined as the mean of six pairwise prototype cosine similarities. Unsupported Mobile classes 18, 28, 38, and 48 are masked gray. Within-class cohesion is approximately one, indicating nearly collinear vectors rather than evidence of distinct subtypes or an independent detection gain. The prototype bank supplies training supervision only and is separate from fixed anatomical templates.'
    captions['current_Mobile_cohesion_all_FDI']='Per-class prototype cohesion for permanent and primary FDI classes in the representative Mobile seed-2 checkpoint. Cohesion averages the six pairwise cosine similarities among four normalized class prototypes and is approximately one across supported classes. Classes 18, 28, 38, and 48 have no Mobile training observations and are unavailable. These quantities describe representation redundancy rather than distinct multimodal prototypes; prototype scores are not fused at inference.'
    captions['current_Mobile_factor_confusions']=f'Row-normalized auxiliary quadrant, tooth-position, and dentition confusion matrices for {n:,} localized unique ordinary queries from the representative Mobile seed-2 checkpoint on all {len(prov["images_evaluated"])} Mobile validation images. Each ordinary query is eligible when its maximum sigmoid score is ≥0.30; one-to-one class-agnostic Hungarian localization matching requires IoU ≥0.50. Gray rows have no reference observations. Counts are obtained in CPU FP32 forward passes and are distinct from post-top-300 detector endpoints. The branches supervise structured identity, not objectness.'
    (DEST/'figure_captions.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2))
    manifest={'checkpoint_sha256':prov['checkpoint_sha256'],'artifact_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in DEST.iterdir() if p.is_file() and p.suffix in ['.svg','.png','.pdf','.npz','.csv']},'representation_check':diagnosis}
    (DEST/'prediction_and_mechanism_assets.json').write_text(json.dumps(manifest,indent=2))
    print('MECHANISM_FIGURES_COMPLETE',n,diagnosis,flush=True)

if __name__=='__main__':main()
