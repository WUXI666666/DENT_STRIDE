"""Select actual native-query examples; retain favorable and adverse inventory."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml

HERE = Path(__file__).resolve().parent
MAT = HERE.parent
sys.path.insert(0, str(MAT/'effectiveness_evidence_2026-10-30'))
from prepare_effectiveness_evidence import config_differences, DATA


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def run(root, name):
    job = read(root/'jobs'/f'{name}_seed0.json')
    out = Path(job['output_dir'])
    metric_file = out/'native_coco_metrics.json'
    if not metric_file.exists():metric_file = out/'metrics_summary.json'
    metrics = read(metric_file)
    ckpt = out/'best.pth'
    assert sha(ckpt) == metrics['checkpoint_sha256']
    return {'job':job, 'metrics':metrics['metrics_percent'],
            'checkpoint':str(ckpt), 'checkpoint_sha256':metrics['checkpoint_sha256'],
            'candidates':str(out/'evaluation/pre_topk_candidates.npz')}


def statuses(candidate, row, gg, W, H, mapping):
    logits = candidate['logits'][row]
    cat = logits.argmax(-1)
    scores = 1/(1+np.exp(-np.clip(logits.max(-1), -80, 80)))
    idx = np.where(scores >= .30)[0]
    box = candidate['boxes_cxcywh'][row, idx].copy()
    box[:, :2] -= box[:, 2:]/2
    box *= [W,H,W,H]
    gt = np.array([g['bbox'] for g in gg])
    lo = np.maximum(box[:,None,:2], gt[None,:,:2])
    hi = np.minimum(box[:,None,:2]+box[:,None,2:], gt[None,:,:2]+gt[None,:,2:])
    inter = np.maximum(hi-lo,0).prod(-1)
    ov = inter/np.maximum(box[:,2:].prod(-1)[:,None]+gt[:,2:].prod(-1)[None,:]-inter,1e-9)
    ans = {g['id']:{'status':'missed','prediction':None,'query':None,'IoU':None} for g in gg}
    for rr, cc in zip(*linear_sum_assignment(-ov)):
        if ov[rr,cc] < .50:
            continue
        q = int(idx[rr]); cid = int(cat[q]); g = gg[cc]
        ans[g['id']] = {'status':'correct' if cid == g['category_id'] else 'wrong',
            'prediction':{'bbox':box[rr].tolist(),'category_id':cid,'FDI':mapping[cid], 'score':float(scores[q])},
            'query':q, 'IoU':float(ov[rr,cc]), 'GT_class_logit':float(logits[q,g['category_id']])}
    return ans


def inventory(group):
    cohort = group['cohort']
    ann = read(DATA[cohort][0]); ims = {i['id']:i for i in ann['images']}
    mapping = {c['id']:int(c.get('fdi',c['name'])) for c in ann['categories']}
    a,b = [np.load(group[s]['candidates']) for s in ['before','after']]
    ai = {int(v):i for i,v in enumerate(a['image_ids'])}; bi = {int(v):i for i,v in enumerate(b['image_ids'])}
    gt = {}
    for g in ann['annotations']:
        gt.setdefault(g['image_id'],[]).append(g)
    rows = []
    for iid, gg in gt.items():
        im = ims[iid]; W,H = im['width'],im['height']
        sa = statuses(a,ai[iid],gg,W,H,mapping); sb = statuses(b,bi[iid],gg,W,H,mapping)
        for g in gg:
            before,after = sa[g['id']],sb[g['id']]
            kind = None
            if before['status']=='wrong' and after['status']=='correct': kind='correction'
            elif before['status']=='correct' and after['status']=='wrong': kind='adverse'
            elif before['status']=='correct' and after['status']=='correct' and before['IoU']<.75 and after['IoU']-before['IoU']>=.12 and after['IoU']>=.80:
                kind='localization'
            if kind is None:
                continue
            cid=g['category_id']; q0=before['query']
            ll=a['logits'][ai[iid],q0].copy(); ll[cid]=-np.inf
            competitor=int(ll.argmax())
            fraction=g['bbox'][2]*g['bbox'][3]/(W*H)
            rows.append({'group':group['name'],'cohort':cohort,'kind':kind,'image_id':iid,
                'image_record':im,'image_file':str(Path(DATA[cohort][1])/im['file_name']),
                'target_annotation':g,'target_FDI':mapping[cid],'competitor_category':competitor,
                'competitor_FDI':mapping[competitor],'before':before,'after':after,
                'area_fraction':fraction,'IoU_gain':after['IoU']-before['IoU']})
    chosen=[]; seen=set()
    for kind,limit in [('correction',5),('localization',1),('adverse',1)]:
        ordered=sorted([r for r in rows if r['kind']==kind],
            key=lambda r:(r['area_fraction']>=.005, r['after']['IoU'] if kind!='adverse' else r['before']['IoU'],r['area_fraction']),reverse=True)
        n=0
        for r in ordered:
            if r['image_id'] in seen:continue
            r['case_id']=f"{group['name']}_{kind}_{n+1:02d}_image{r['image_id']}_FDI{r['target_FDI']}"
            chosen.append(r); seen.add(r['image_id']); n+=1
            if n==limit:break
    assert len([c for c in chosen if c['kind']=='correction'])>=3
    for r in chosen:r['image_sha256']=sha(r['image_file'])
    return rows,chosen


def main():
    control=MAT/'cmig_controlled_2026-09-29'; reference=MAT/'cmig_best3_seeds_2026-10-08'
    specs=[('CMFA','DSLR',control,'dslr_base_adamw','dslr_cmfa_adamw'),
           ('HOOA','Mobile',reference,'internal_mobile_hooa_neither','mobile_1024_full'),
           ('Prototype','Mobile',reference,'internal_mobile_proto_off','internal_mobile_proto_k4_tau0p5')]
    groups=[]
    for name,cohort,root,before,after in specs:
        group={'name':name,'cohort':cohort,'before':run(root,before),'after':run(root,after)}
        x,y=[yaml.safe_load(Path(group[s]['job']['config']).read_text()) for s in ['before','after']]
        group['config_differences']=config_differences(x,y)
        group['controls']={s:group[s]['job']['controls'] for s in ['before','after']}
        allcases,cases=inventory(group)
        group['inventory_counts']={k:sum(r['kind']==k for r in allcases) for k in ['correction','localization','adverse']}
        group['cases']=cases
        (HERE/f'{name}_candidate_inventory.json').write_text(json.dumps(allcases,indent=2))
        groups.append(group)
        print(name,'AP',group['before']['metrics']['AP'],group['after']['metrics']['AP'],
              'inventory',group['inventory_counts'],'selected',[(r['image_id'],r['target_FDI'],r['kind']) for r in cases],flush=True)
    (HERE/'gallery_plan.json').write_text(json.dumps({'groups':groups,
        'selection':'Native ordinary-query top1 labels at score>=.30, class-agnostic one-to-one Hungarian IoU>=.50 matching. Selected illustrative corrections are not a prevalence sample. Adverse examples retained.',
        'scope':'Controlled internal-validation examples; no new training or manuscript edits.'},indent=2))


if __name__=='__main__':main()
