from pathlib import Path
from collections import Counter
import os, json, re, shutil, subprocess, time, zipfile, hashlib
import fitz

root=Path('/home/wuxi/Dent-FINE-paper')
work=root/'cmig_revision_visuals_20261117'
src=root/'CMIG_Overleaf_Visual_20261117'
zip_path=root/'DENT_STRIDE_CMIG_Overleaf_Visual_20261117.zip'
pdf_path=root/'DENT_STRIDE_CMIG_Visual_20261117.pdf'
allowed={'.tex','.bib','.cls','.sty','.bst','.pdf'}
files=sorted(f for f in src.iterdir() if f.is_file())
assert all(f.suffix in allowed for f in files)
assert len(list(src.glob('*.tex')))==1
assert all(f.name.startswith('Figure_') for f in src.glob('*.pdf'))
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for f in files:z.write(f,f.name)

build=work/'clean_overleaf_test'
if build.exists():shutil.rmtree(build)
build.mkdir()
with zipfile.ZipFile(zip_path) as z:z.extractall(build)
tc=root/'cmig_latex_work/tex_toolchain'; binary=tc/'root/usr/bin'
env=os.environ.copy();env.update(json.loads((tc/'tex_env.json').read_text()));env['PATH']=str(binary)+':'+env['PATH']
commands=[
 [str(binary/'pdftex'),'-fmt=pdflatex','-interaction=nonstopmode','-halt-on-error','cas-dc-template.tex'],
 [str(binary/'bibtex.original'),'cas-dc-template'],
 [str(binary/'pdftex'),'-fmt=pdflatex','-interaction=nonstopmode','-halt-on-error','cas-dc-template.tex'],
 [str(binary/'pdftex'),'-fmt=pdflatex','-interaction=nonstopmode','-halt-on-error','cas-dc-template.tex'],
]
times=[]
for i,cmd in enumerate(commands):
    t=time.monotonic();r=subprocess.run(cmd,cwd=build,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    times.append({'step':i+1,'seconds':time.monotonic()-t,'exit_code':r.returncode})
    (build/f'run{i+1}.stdout').write_text(r.stdout)
    if r.returncode:raise RuntimeError(r.stdout[-6000:])

s=(src/'cas-dc-template.tex').read_text()
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',s,re.S).group(1).strip()
intro=s[s.index('\\section{Introduction}'):s.index('% figure01')].strip()
original_abstract=(work/'original_abstract.txt').read_text().strip()
expected_abstract=original_abstract.replace('72.91','73.22').replace('92.49','92.84')
assert abstract==expected_abstract
assert intro==(work/'original_introduction.txt').read_text().strip()
figs=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s)
assert all((src/f).exists() for f in figs)
assert set(figs)=={f.name for f in src.glob('*.pdf')}
assert not re.search(r'\\(?:input|include)\{',s)
labels=re.findall(r'\\label\{([^}]+)\}',s)
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',s)
assert not set(refs)-set(labels)
assert all(n==1 for n in Counter(labels).values())
keys=set(re.findall(r'@\w+\s*\{\s*([^,]+)',(src/'cas-refs.bib').read_text()))
cites={k.strip() for g in re.findall(r'\\cite\w*\s*(?:\[[^]]*\])*\{([^}]+)\}',s) for k in g.split(',')}
assert not cites-keys
log=(build/'cas-dc-template.log').read_text(errors='replace')
assert not re.search(r'undefined|Missing character|multiply defined|Fatal error|^!',log,re.M)
shutil.copy2(build/'cas-dc-template.pdf',pdf_path)
d=fitz.open(pdf_path)
outside=[]
for i,p in enumerate(d):
    for b in p.get_text('dict')['blocks']:
        if b['type']!=0:continue
        for line in b['lines']:
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                if min(x0,y0)<-0.5 or x1>p.rect.width+.5 or y1>p.rect.height+.5:
                    outside.append({'page':i+1,'text':span['text'],'bbox':span['bbox']})
assert not outside
summary={
 'archive':str(zip_path),'archive_bytes':zip_path.stat().st_size,'flat_files':len(files),
 'source_bytes':sum(f.stat().st_size for f in files),'pdf':str(pdf_path),'pdf_bytes':pdf_path.stat().st_size,
 'pages':len(d),'clean_compile_seconds':sum(t['seconds'] for t in times),'compile_steps':times,
 'abstract_original_exact':False,'abstract_unchanged_except_author_approved_values':True,
 'abstract_author_approved_changes':{'DSLR_AP':['72.91','73.22'],'DSLR_F1':['92.49','92.84']},
 'introduction_original_exact':True,
 'all_manuscript_in_single_tex':True,'figure_pdfs':len(figs),'numbered_figures':18,
 'editable_tables':s.count('\\begin{table*}'),'numbered_equations':s.count('\\begin{equation}'),
 'cited_bib_entries':len(cites),'undefined_references':0,'undefined_citations':0,'missing_glyphs':0,
 'outside_page_text':outside,
 'overfull_box_log_entries':re.findall(r'Overfull[^\n]+',log),
 'overfull_note':'The remaining 123.62721pt box comes from the CAS keyword label in its zero-width box; rendered content remains within the page.',
 'files':[{'name':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files],
}
(work/'final_validation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['files','overfull_box_log_entries']},indent=2))
