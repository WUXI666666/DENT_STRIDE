from pathlib import Path
import os, json, shutil, subprocess, time
root=Path('/home/wuxi/Dent-FINE-paper')
work=root/'cmig_revision_visuals_20261117'
src=root/'CMIG_Overleaf_Visual_20261117'
build=work/'build'
build.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file(): shutil.copy2(f,build/f.name)
tc=root/'cmig_latex_work/tex_toolchain'
env=os.environ.copy(); env.update(json.loads((tc/'tex_env.json').read_text()))
binary=tc/'root/usr/bin'
env['PATH']=str(binary)+':'+env['PATH']
commands=[
 [str(binary/'pdftex'),'-fmt=pdflatex','-interaction=nonstopmode','-halt-on-error','cas-dc-template.tex'],
 [str(binary/'bibtex.original'),'cas-dc-template'],
 [str(binary/'pdftex'),'-fmt=pdflatex','-interaction=nonstopmode','-halt-on-error','cas-dc-template.tex'],
 [str(binary/'pdftex'),'-fmt=pdflatex','-interaction=nonstopmode','-halt-on-error','cas-dc-template.tex'],
]
stats=[]
for i,cmd in enumerate(commands):
    t=time.monotonic()
    r=subprocess.run(cmd,cwd=build,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (build/f'run{i+1}.stdout').write_text(r.stdout)
    stats.append({'step':i+1,'exit':r.returncode,'seconds':time.monotonic()-t})
    if r.returncode: print(r.stdout[-5000:]); break
result={'total_seconds':sum(x['seconds'] for x in stats),'steps':stats}
(build/'timing.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
