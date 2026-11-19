"""Prepare an auditable experiment design. Does NOT start training or alter splits."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import csv
import json
import shutil
import yaml

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/wuxi/Dent-FINE')
PAPER = ROOT.parent.parent
CONFIGS = ROOT/'planned_configs_unfrozen'
PRIMARY = []

def dump(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)

def csvout(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def feature(c, cmfa, hooa):
    c['HybridEncoder']['use_cmfa_fusion']=cmfa
    c['HybridEncoder']['use_hooa_attention']=hooa

def identity(c, factors, proto, fusion):
    d=c['DFINETransformer'];loss=c['DEIMCriterion']
    d['use_factor_heads']=factors;d['use_tooth_prototype']=proto
    d['factor_fusion_train']=fusion
    d['use_fdi_factor_prior']=fusion
    for k,v in zip(['factor_alpha_quad','factor_alpha_tooth','factor_alpha_dent'],[.1,.05,.03]):
        d[k]=v if fusion else 0.
    d['fdi_factor_logit_scale']=0.
    d['prototype_fusion_weight']=0.;d['prototype_fusion_train']=False
    loss['losses']=[x for x in loss['losses'] if x not in ['quad','tooth','dent','proto']]
    if factors:loss['losses']+=['quad','tooth','dent']
    if proto:loss['losses']+=['proto']

def no_prior(c):
    for k in ['use_dental_spatial_prior','use_dental_shape_prior','use_dental_view_prior','use_dental_appearance_prior','use_global_assignment']:
        c['PostProcessor'][k]=False

def add(identifier, cohort, feature_state, identity_state, optimizer, resolution, c):
    for seed in [0,1,2]:
        v=deepcopy(c)
        v['output_dir']=str(ROOT/'runs'/identifier/f'seed_{seed}')
        snapshot=ROOT/'source_snapshots'/('mobile' if cohort=='Mobile' else 'local')
        v['HGNetv2']['local_model_dir']='/home/wuxi/RT-DETR-main/D-FINE/weight/hgnetv2'
        p=CONFIGS/identifier/f'seed_{seed}.yml'
        dump(p,yaml.safe_dump(v,sort_keys=False))
        PRIMARY.append({'configuration':identifier,'cohort':cohort,'seed':seed,'features':feature_state,'identity':identity_state,'optimizer':optimizer,'input_policy':resolution,'epochs':160,'config':str(p),'implementation':str(snapshot),'status':'NOT_STARTED_AWAITING_SPLIT_DECISION_AND_PREFLIGHT','reused':False})

def main():
    source_records=[]
    for name,src in [('local',REPO),('mobile',REPO/'experiments/a07_proto_aux_ap_patch_1024')]:
        selected=list((src/'engine').rglob('*.py'))+[src/'train.py']
        for p in selected:
            dest=ROOT/'source_snapshots'/name/p.relative_to(src)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert dest.read_bytes()==p.read_bytes()
            else:shutil.copy2(p,dest)
            source_records.append({'source':str(p),'snapshot':str(dest),'sha256':sha256(p.read_bytes()).hexdigest()})
        for p in [REPO/'LICENSE', REPO/'requirements.txt']:
            dest=ROOT/'source_snapshots'/name/p.name
            if not dest.exists():shutil.copy2(p,dest)
    dump(ROOT/'source_snapshot_manifest.json',json.dumps(source_records,indent=2))
    refs={k:yaml.safe_load((ROOT/'reference_configs'/f'{k}_selected.yml').read_text()) for k in ['Mobile','Pano','DSLR']}
    for size,states in [(1024,[(0,0),(1,0),(0,1),(1,1)]),(640,[(0,0),(1,1)])]:
        for feat,ident in states:
            c=deepcopy(refs['Mobile']);feature(c,bool(feat),bool(feat));identity(c,bool(ident),bool(ident),bool(ident));no_prior(c)
            c['eval_spatial_size']=[size,size]
            for loader in ['train_dataloader','val_dataloader']:
                for op in c[loader]['dataset']['transforms']['ops']:
                    if op['type']=='Resize':op['size']=[size,size]
            c['train_dataloader']['collate_fn']['base_size']=size
            c['train_dataloader']['collate_fn']['base_size_repeat']=None
            c['DEIMCriterion']['use_arch_order_loss']=True
            c['DEIMCriterion']['weight_dict']['loss_arch']=.1
            c['DEIMCriterion']['weight_dict']['loss_nwd']=.5
            assert 'nwd' in c['DEIMCriterion']['losses']
            label={(0,0):'base',(1,0):'features',(0,1):'identity',(1,1):'full'}[(feat,ident)]
            add(f'mobile_{size}_{label}','Mobile','CMFA+HOOA' if feat else 'none','Factor+ProtoAux+factor fusion' if ident else 'none','AdamW',f'fixed{size}',c)
    for factor,proto in [(0,0),(1,0),(0,1),(1,1)]:
        c=deepcopy(refs['Pano']);feature(c,False,False);identity(c,bool(factor),bool(proto),False);no_prior(c)
        c['DEIMCriterion']['weight_dict']['loss_arch']=.25
        label={(0,0):'base',(1,0):'factor',(0,1):'prototype',(1,1):'factor_prototype'}[(factor,proto)]
        add(f'pano_{label}','Pano','none',label,'AdamW','fixed640',c)
    for cmfa,dual in [(0,0),(1,0),(0,1),(1,1)]:
        c=deepcopy(refs['DSLR']);feature(c,bool(cmfa),False);identity(c,False,False,False);no_prior(c)
        c['DEIMCriterion']['weight_dict']['loss_arch']=.25
        if not dual:
            c['optimizer']['type']='AdamW'
            for k in ['rollback_weight','realtime_decay','layer_decay_power','layer_decay_min','layer_decay_max']:c['optimizer'].pop(k,None)
        add(f'dslr_{"cmfa" if cmfa else "base"}_{"dualopt" if dual else "adamw"}','DSLR','CMFA' if cmfa else 'none','none','DualOptAdamW' if dual else 'AdamW','legacy640_multiscale',c)
    assert len(PRIMARY)==42 and len({r['configuration'] for r in PRIMARY})==14
    csvout(ROOT/'primary_experiments.csv',PRIMARY)
    internal=[
      ('SpatialShape','Pano','none / center / shape / center+shape',0,'Same pano_factor_prototype seed checkpoint; full pre-top-k logits and boxes. Repeat replay for seeds 0/1/2. No view/appearance term.'),
      ('CMFA gates','Mobile','both off / channel only / spatial only / both on',3,'Full1024 anchor. Channel off removes gated cross-residual; spatial off is identity. Keep 2C output. Requires source switch tests before training.'),
      ('HOOA','Mobile','full / no high-pass / no second-order / neither',3,'Full1024 anchor. Proposed no high-pass: zero edge contribution. Proposed no second-order: replace local*sigmoid(context) with context, preserving context branch and 3C projection. Freeze this definition before training.'),
      ('Factor supervision','Mobile','all / no quadrant / no position / no dentition',3,'Anchor: training fusion OFF and inference fusion OFF. Keep all heads; remove only specified CE term. Reuse full-supervision training-fusion-OFF model below.'),
      ('Factor fusion','Mobile','train on/off x inference on/off',1,'Two training settings, each replayed with inference on/off before query selection. Training-ON seed0 reuses primary full1024 after equivalence check.'),
      ('Prototype','Mobile','off / K=1,2,4 at tau=.20 / tau=.10,.20,.50 at K=4',5,'K denotes prototypes per FDI class, not class count. Full1024 is K=4,tau=.20 reference; no prototype inference fusion.'),
      ('DualOpt','DSLR','none / rollback only / extra decay only / both',2,'CMFA architecture fixed. None and both reuse two primary DSLR seed0 runs. AdamW ordinary weight decay stays unchanged.'),
      ('Mobile background losses','Mobile','neither / NWD only / arch only / both',3,'Full1024 modules fixed. NWD weight .50 (normalizer .08), arch weight .10. Both reuses primary full1024 seed0.'),
    ]
    rows=[dict(zip(['module','cohort','comparisons','additional_seed0_trainings','definition_and_reuse'],r)) for r in internal]
    csvout(ROOT/'internal_ablation_design.csv',rows)
    assert sum(r['additional_seed0_trainings'] for r in rows)==20
    protocol={
      'status':'PREPARATION_ONLY_NO_TRAINING_STARTED',
      'user_decisions':{'independent_test':'not available; controlled internal validation only','split_changes':'not authorized; report audit first','primary_seeds':[0,1,2],'internal_seeds':[0],'gpu':0},
      'primary_configurations':14,'primary_training_runs_before_reuse':42,'additional_internal_training_runs_planned':20,
      'total_training_runs_if_no_historical_reuse':62,
      'budget':{'epochs':160,'train_batch':8,'validation_batch':16,'AMP':True,'EMA':True},
      'initialization':{'path':'/home/wuxi/RT-DETR-main/D-FINE/weight/hgnetv2/PPHGNetV2_B0_stage1.pth','sha256':sha256(Path('/home/wuxi/RT-DETR-main/D-FINE/weight/hgnetv2/PPHGNetV2_B0_stage1.pth').read_bytes()).hexdigest()},
      'comparison_controls':'Within each cohort/family: fixed source snapshot, data/hash list, label map, pretraining, augmentation, background losses, batch/epochs/EMA/scheduler, precision and checkpoint rule, except declared variable.',
      'checkpoint_selection':'Maximum validation native COCO AP over all 160 epochs, earliest epoch on exact ties. Pano selected with all inference priors/factor fusion OFF. Freeze prior weights, thresholds and comparisons before launch.',
      'checkpoint_retention':'Final best.pth only per experiment. Keep minimal temporary stage checkpoints when required by the original training algorithm; delete only task-owned redundant files after successful verified completion. Do not change solver stage behavior solely to save disk.',
      'statistics':'Per-seed scores and sample SD (ddof=1). Paired seed differences and Student-t df=2 CI; separately paired resampling of identical validation units, grouping DSLR by patient/all five views. Mobile/Pano patient IDs unavailable: only content-group/image-level internal intervals, never patient-level or independent-test claims.',
      'evaluation':'Native AP/AP50/AP75/AR100 with no score truncation; P/R/F1 at score>=.30, IoU>=.50, one-to-one class-agnostic localization matching and correct FDI for TP. Matched FDI denominator=localized matches; localization coverage denominator=GT instances. Raw duplicate-image denominator=all evaluated images. Save raw candidate tensors before top-k for prior/factor replays and all retained native predictions.',
      'external_comparators':'Keep RT-DETR-L. Verify semantic annotation equivalence, image hashes, canonical label maps, training provenance, input/pretraining and native postprocessing before reuse. Do not label framework-native baselines fully controlled.',
      'cross_device':'Both directions; each photographic controlled base and full model uses source resolution and FDI52 without target fine-tuning. Not independent external validation.',
      'efficiency':'RTX4090 GPU0; batch1 inference; same CUDA/PyTorch, FP32 and AMP separately; warm-up then synchronized repeated timing; forward-only and end-to-end tensor pipeline separately; mean/SD/P50/P95, throughput, peak allocated/reserved memory, params and MACs with tool/unsupported operators stated. Training peak separately at batch8 AMP.',
      'figures':'Generate only from newly accepted checkpoints; same image/threshold/class colors/shared zoom across compared methods; success and failure examples; prior help and harm cases; actual seeds with SD; raw prediction PR/F1/duplicate curves and full supported/unsupported FDI confusion/count tables.',
      'history_reuse':'No old run accepted for reuse yet. Exact semantic configuration/source/initialization/data/stage-rule checks required; Pano previously selected with prior ON is not automatically reusable for prior-OFF selection.'
    }
    dump(ROOT/'protocol_preparation.json',json.dumps(protocol,ensure_ascii=False,indent=2))
    print(json.dumps({'primary_configurations':14,'primary_runs':42,'internal_additional_runs':20,'training_started':False,'source_files_snapshotted':len(source_records)},indent=2))

if __name__=='__main__':main()
