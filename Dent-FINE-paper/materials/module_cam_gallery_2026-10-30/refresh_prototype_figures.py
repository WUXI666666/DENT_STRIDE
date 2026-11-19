from pathlib import Path
import json
import render_gallery as r

HERE=Path(__file__).resolve().parent
oldcaptions=json.loads((HERE/'figure_captions.json').read_text())
oldmanifest=json.loads((HERE/'figure_manifest.json').read_text())
recommend=json.loads((HERE/'recommendations.json').read_text())
cases=r.DATA['Prototype']['cases']
for c in cases:r.grid([c],c['case_id']+'_Prototype_detail',prototype=True)
top=[next(c for c in cases if c['case_id']==key) for key in recommend['Prototype']['case_ids']]
r.grid(top,'Recommended_Prototype_mechanism',prototype=True,method='hirescam',prototype_scores=True)
oldcaptions.update(r.CAPTIONS);oldmanifest.update(r.MANIFEST)
(HERE/'figure_captions.json').write_text(json.dumps(oldcaptions,indent=2))
(HERE/'figure_manifest.json').write_text(json.dumps(oldmanifest,indent=2))
