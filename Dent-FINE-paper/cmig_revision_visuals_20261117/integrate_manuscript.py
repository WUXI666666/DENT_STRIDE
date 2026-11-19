from pathlib import Path
import re, shutil

root = Path('/home/wuxi/Dent-FINE-paper')
work = root / 'cmig_revision_visuals_20261117'
out = root / 'CMIG_Overleaf_Visual_20261117'
s = (root / 'CMIG_Overleaf_Lite_20261117/cas-dc-template.tex').read_text()

# Preserve template prose; update only the two author-approved DSLR abstract values.
s = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', lambda _: '\\begin{abstract}\n' + (work/'original_abstract.txt').read_text().strip().replace('72.91','73.22').replace('92.49','92.84') + '\n\\end{abstract}', s, flags=re.S)
fig1 = re.search(r'% figure01\n.*?\\end\{figure\*\}', s, re.S).group(0)
start = s.index('\\section{Introduction}')
end = s.index('% related work', start)
s = s[:start] + (work/'original_introduction.txt').read_text().rstrip() + '\n\n' + fig1 + '\n\n' + s[end:]

methods = r'''
\subsection{Query-specific attribution and judgment sensitivity}
\label{sec:visual-methods}

We complemented retrained controls with spatial attribution of selected tooth predictions. For a feature tensor $A$ and scalar query score $s$, the positive elementwise gradient--activation map is
\begin{equation}
 M_{uv}=\left[\sum_c A_{cuv}\frac{\partial s}{\partial A_{cuv}}\right]_+,
 \label{eq:cam}
\end{equation}
following the HiResCAM formulation \citep{draelos2021hirescam}. The score is the reference-FDI detection logit, or the auxiliary prototype class score where specified. Controlled CMFA and HOOA comparisons use the finest encoder output and a matched native query in each independently trained model. Final-model comparisons retain the same image, checkpoint, and query across module panels. Prototype maps explain the training branch; prototype scores remain excluded from inference fusion.

Maps were computed in FP32 on complete inputs, divided by their own positive full-image maxima, and then displayed with identical crops within each comparison. Ground-truth boxes identify and display the target tooth but do not mask the attribution. The native query-content initialization is detached, so these gradients describe the remaining differentiable paths. The maps indicate relative spatial attribution; their color intensity is not a common contribution scale across modules or models.

For the factor heads, an input-occlusion analysis \citep{zeiler2014visualizing} measures changes in the correct-factor judgment. Let $m_h(x)$ be the correct-factor logit minus the largest alternative logit in the unperturbed input for head $h$. The correct and alternative indices are held fixed. Replacing region $R$ by the image's mean RGB color gives
\begin{equation}
 \Delta_h(R)=m_h(x)-m_h(x_{\setminus R}).
 \label{eq:occlusion}
\end{equation}
Each $1024\times1024$ image is evaluated at $29\times29=841$ locations using a $128\times128$ patch and stride 32. Each perturbed forward pass yields all three heads. Encoder top-$k$ anchor indices, their order, and the target query are fixed to the baseline; feature content and downstream predictions are recomputed. The dense map averages signed margin changes over all patches covering each pixel. A symmetric raw-margin scale is shared across cases within each head. Red denotes a reduced correct-factor margin after occlusion (supporting evidence); blue denotes an increased margin (potential interference). These conditional judgment sensitivity maps are neither native attention maps nor evaluations of detection accuracy after unrestricted query reselection.

'''
s = s.replace('\\FloatBarrier\n\n% experiments', methods + '\\FloatBarrier\n\n% experiments', 1)

design = r'''
\subsubsection{Additional controlled and visual analyses}
\label{sec:additional-design}
The additional CMFA controls compare separately trained DSLR models without and with CMFA using paired seeds 0, 1, and 2 under each of two optimizer settings. One setting uses AdamW; the other uses a reference DualOpt configuration with initialization rollback weight $0.005$ and extra decay $0.01$. This second control differs from the final DSLR configuration, which disables rollback. We report within-seed AP differences and their sample SD, keeping these reference controls separate from the main framework comparison.

The Mobile prototype comparison uses seed 0 without prototype supervision versus four prototypes per class at temperature $0.50$, with prototype inference fusion disabled in both models. The HOOA visualization uses a separate seed-0 reference at temperature $0.20$, comparing full HOOA against removal of the high-pass and multiplicative second-order mechanisms while retaining its other operations. Selected examples illustrate the associated identity changes; they do not estimate the frequency of corrections. Predictions were retained at score $\geq0.30$ and associated with reference teeth by class-agnostic one-to-one matching at IoU $\geq0.50$. The final Mobile attribution and occlusion analyses use the same seed-2 EMA checkpoint with four prototypes per class and temperature $0.50$.

'''
s = s.replace('\\subsection{Detection performance and comparisons}', design + '\\subsection{Detection performance and comparisons}', 1)

new_results = r'''
\subsubsection{Paired CMFA controls and prototype supervision}
\label{sec:controlled-additional}
Figure~\ref{fig:figure12} summarizes the additional controlled comparisons. Enabling CMFA increased DSLR AP in every paired seed: the changes were $+0.492$, $+0.357$, and $+0.647$ points with AdamW, averaging $+0.498\pm0.145$. Under the reference DualOpt setting with rollback, the corresponding gains were $+0.545$, $+0.411$, and $+0.584$ points, averaging $+0.513\pm0.091$. The consistent direction across these pairs supports cross-scale aggregation in the tested settings. The Mobile seed-0 prototype control increased AP from $54.918$ to $55.683$ ($+0.765$ points), supporting auxiliary identity supervision without adding prototype scores at inference.

\begin{figure*}[tp]
\centering
\includegraphics[width=\textwidth]{Figure_12.pdf}
\caption{Controlled effects of CMFA and prototype supervision. (a,b) Within-seed DSLR AP gains from enabling CMFA for seeds 0, 1, and 2 under AdamW and reference DualOpt with rollback. Points are paired gains; diamonds and error bars show their mean and sample SD. The reference DualOpt setting includes rollback and differs from the final DSLR optimizer. (c) Mobile AP without and with prototype supervision at seed 0; no between-seed error bar is available for this contrast. All differences are AP percentage points.}
\label{fig:figure12}
\end{figure*}

\subsubsection{Identity corrections and module attribution}
\label{sec:module-attribution}
In the prototype example (Fig.~\ref{fig:figure13}), the reference primary tooth FDI 51 is predicted as permanent tooth FDI 11 without prototype supervision and correctly as 51 after it is enabled. The auxiliary prototype CAM and supported-class scores connect this correction with the supervised identity representation. The displayed LogSumExp scores are unnormalized class scores, not probabilities. Inspection of the final Mobile seed-2 bank found nearly collinear same-class vectors across the 48 supported classes (mean within-class pairwise cosine similarity $0.999999965$). Thus, the positive controlled result supports prototype-assisted identity supervision, without implying that the four slots encode four distinct tooth morphologies.

\begin{figure*}[tp]
\centering
\includegraphics[width=\textwidth]{Figure_13.pdf}
\caption{Prototype-assisted identity supervision for reference FDI 51 (Mobile image 44). Independently trained seed-0 models change the matched prediction from FDI 11 to 51 when prototype supervision is enabled. The bank uses four prototypes per class at temperature $0.50$, with inference fusion disabled. The prototype CAM targets the auxiliary class score; the final panel gives the five highest supported-class LogSumExp scores. Teal boxes denote ground truth and orange boxes predictions. CAMs follow Eq.~\eqref{eq:cam}, with independent full-image normalization before the same crop; dashed reference boxes do not mask the maps.}
\label{fig:figure13}
\end{figure*}

The separate HOOA reference comparison at temperature $0.20$ increased AP from $53.561$ to $54.910$ when the high-pass and second-order mechanisms were retained ($+1.349$ points). Figure~\ref{fig:figure14} shows the accompanying FDI 53-to-13 correction for a reference permanent canine. In this example, the positive reference-class attribution is more concentrated near the target tooth with full HOOA. This contrast tests two mechanisms within HOOA, rather than removing the entire module.

\begin{figure*}[tp]
\centering
\includegraphics[width=\textwidth]{Figure_14.pdf}
\caption{HOOA mechanism comparison for reference FDI 13 (Mobile image 77). The reduced model removes the high-pass response and substitutes the direct context response for the multiplicative second-order response; other HOOA operations remain active. Full HOOA corrects FDI 53 to 13. Both models use seed 0 and prototype temperature $0.20$. CAMs target the reference-FDI detection logit at the finest encoder output, using an independently matched query in each model. Boxes and per-map normalization follow Fig.~\ref{fig:figure13}.}
\label{fig:figure14}
\end{figure*}

The DSLR CMFA examples in Fig.~\ref{fig:figure15} show corrections from FDI 55 to 16 and from 81 to 41 under the AdamW seed-0 control. These cases illustrate the identity changes underlying the paired comparison without requiring every corrected prediction to have a more spatially concentrated CAM. Their attribution patterns include both the target tooth and adjacent image context.

\begin{figure*}[tp]
\centering
\includegraphics[width=\textwidth]{Figure_15.pdf}
\caption{CMFA controls on two DSLR images with AdamW and seed 0. Reference teeth are (a) FDI 16 in image 125 and (b) FDI 41 in image 124. Enabling CMFA corrects the matched labels from FDI 55 to 16 and from 81 to 41. CAMs target the reference-FDI detection logit at the finest encoder output using independently matched native queries. Boxes and normalization follow Fig.~\ref{fig:figure13}; intensity cannot be compared as a contribution magnitude between models. The paired validation gains are shown in Fig.~\ref{fig:figure12}.}
\label{fig:figure15}
\end{figure*}

Figure~\ref{fig:figure16} then examines the final Mobile model on a partly visible tooth and a blurred image. Within each row, detection, CMFA, and HOOA high-pass maps explain the same reference-FDI detection logit, while the prototype map explains the auxiliary identity score. All panels use the same checkpoint, image, and native query. The different spatial patterns provide complementary views of the executed feature and identity pathways; they are mechanism illustrations rather than additional ablations.

\begin{figure*}[tp]
\centering
\includegraphics[width=\textwidth]{Figure_16.pdf}
\caption{Module attribution in the final Mobile model (EMA seed 2; four prototypes; temperature $0.50$). Rows show (a) partly visible FDI 51 in image 44 and (b) FDI 64 in blurred image 115. Each row uses one image, checkpoint, and native query. Detection, CMFA, and HOOA edge maps target the reference-FDI detection logit at the finest encoder output, finer top-down fusion output, and high-pass branch tensor, respectively. The prototype map targets the auxiliary class score, excluded from inference fusion. Each map is normalized by its own full-image maximum before cropping. Orange boxes are predictions and dashed teal boxes are reference locations.}
\label{fig:figure16}
\end{figure*}

\subsubsection{Factor-head judgment sensitivity}
Figure~\ref{fig:figure17} shows the input-occlusion results for reference teeth FDI 64, 11, and 46. Quadrant and dentition judgments show supportive regions near the target teeth; position judgments also respond to surrounding teeth and context. Across these selected cases, the patch producing the largest correct-margin decrease intersects the target box for every head. Signed changes reveal both supportive regions and regions whose occlusion increases the correct margin. The maps therefore identify where the fixed-query judgments depend on input evidence, complementing the factor-supervision controls. Ground-truth boxes were used for correspondence and display, not to generate hotspots.

\begin{figure*}[tp]
\centering
\includegraphics[width=\textwidth]{Figure_17.pdf}
\caption{Judgment sensitivity maps for the quadrant, position, and dentition heads in the final Mobile model. Rows correspond to reference FDI 64, 11, and 46. Each image undergoes 841 mean-RGB occlusions ($128\times128$ pixels; stride 32 on a $1024\times1024$ input), with all three heads evaluated in each pass. Red indicates that occlusion weakens the correct-factor logit margin; blue indicates that it strengthens the margin. Dense maps average signed changes over overlapping patches. Each head shares one symmetric raw-margin scale across cases. Baseline anchor indices, their order, and the target query are fixed while input-dependent content and downstream predictions are recomputed. Dashed boxes mark reference teeth and do not mask the calculation. These are conditional sensitivity maps, not native attention or probability maps.}
\label{fig:figure17}
\end{figure*}

\subsubsection{Computational cost}
'''
a = s.index('\\subsubsection{Prototype-bank inspection}')
b = s.index('Table~\\ref{tab:table8} summarizes', a)
s = s[:a] + new_results + '\n' + s[b:]

s = s.replace("Three completed runs characterize each configuration's observed seed variability.", "The reported repeated runs characterize observed training variability on the fixed validation partitions.")
s = s.replace('These findings support particular mechanisms and configuration choices, not a universal benefit from every additional module.', 'The additional CMFA controls show positive paired AP changes in all three seeds under both reference optimizers, while the seed-0 prototype comparison supports auxiliary identity supervision. Together, these experiments support the tested feature, identity, and anatomical-scoring mechanisms within their respective configurations.')
s = s.replace('Prototype supervision and optimization choices are defined configuration components, without an independent accuracy benefit beyond what the corresponding controls support.', 'The prototype control demonstrates that useful auxiliary supervision can coexist with convergence of same-class prototype slots. The benefit does not require interpreting those slots as distinct morphological modes. The selected CAM examples connect aggregate effects to actual identity corrections, and the input-occlusion maps independently show spatial sensitivity of the factor judgments.')
s = s.replace('Component-learning ablations use one seed at their reference configurations, and prior comparisons are conditional on weights obtained under combined scoring.', 'The HOOA, factor-supervision, and prototype controls use one seed, whereas the additional CMFA comparisons use three paired seeds per reference optimizer. Prior comparisons remain conditional on weights obtained under combined scoring. Selected attribution examples do not quantify correction prevalence; the gradient maps follow the available differentiable paths, and mean-color occlusion introduces an artificial perturbation.')
s = s.replace('Reference-configuration HOOA and factor-supervision controls, together with same-weight center-prior comparisons, provide mechanism-specific evidence.', 'Reference-configuration HOOA and factor-supervision controls, paired CMFA gains, prototype supervision, and same-weight center-prior comparisons provide mechanism-specific evidence. Query-specific attribution and factor-head judgment sensitivity connect these controlled results to the spatial evidence used in selected tooth decisions.')
s = s.replace('detection overlays and internal-response plots were generated from actual model predictions and tensors.', 'detection overlays, attribution maps, and judgment sensitivity plots were computed from actual model predictions, tensors, and input perturbations.')
s = s.replace('% Source scientific content and the author-confirmed numbers are unchanged.', '% Introduction is verbatim; abstract changes only the two author-approved DSLR values.\n% New controlled analyses are identified by their own configurations.')
(out/'cas-dc-template.tex').write_text(s)

figs = {'Figure_Module_controlled_AP':12,'New_Prototype':13,'New_HOOA':14,'New_CMFA':15,'New_FinalMobile':16,'New_FactorSensitivity':17}
for stem, number in figs.items():
    src = work/('quantitative' if number==12 else 'figures')/(stem+'.pdf')
    shutil.copy2(src, out/f'Figure_{number}.pdf')

bib = (root/'CMIG_Overleaf_Lite_20261117/cas-refs.bib').read_text()
bib += r'''

@misc{draelos2021hirescam,
  author = {Draelos, Rachel Lea and Carin, Lawrence},
  title = {Use {HiResCAM} instead of {Grad-CAM} for faithful explanations of convolutional neural networks},
  year = {2021},
  eprint = {2011.08891},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2011.08891},
  url = {https://arxiv.org/abs/2011.08891},
  note = {Version 4}
}

@inproceedings{zeiler2014visualizing,
  author = {Zeiler, Matthew D. and Fergus, Rob},
  title = {Visualizing and Understanding Convolutional Networks},
  booktitle = {Computer Vision---ECCV 2014},
  series = {Lecture Notes in Computer Science},
  volume = {8689},
  pages = {818--833},
  year = {2014},
  publisher = {Springer},
  doi = {10.1007/978-3-319-10590-1_53}
}
'''
(out/'cas-refs.bib').write_text(bib)
print('Wrote', out/'cas-dc-template.tex', len(s), 'characters')
