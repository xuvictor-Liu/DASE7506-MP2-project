"""Consolidated six-page report using observed results, including failed audits."""
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from pypdf import PdfReader
root=Path(__file__).resolve().parent
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyA',fontName='Helvetica',fontSize=10.2,leading=14.2,spaceAfter=9,textColor=colors.HexColor('#243446')))
styles.add(ParagraphStyle(name='SmallA',fontName='Helvetica',fontSize=8.5,leading=11,spaceAfter=5))
styles.add(ParagraphStyle(name='CellA',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=0))
styles.add(ParagraphStyle(name='SubA',fontName='Helvetica-Bold',fontSize=12,leading=16,spaceBefore=9,spaceAfter=7,textColor=colors.HexColor('#155E75')))
story=[]
def p(t): story.append(Paragraph(t,styles['BodyA']))
def h(t): story.append(Paragraph(t,styles['SubA']))
def small(t): story.append(Paragraph(t,styles['SmallA']))
def page(title):
    if story: story.append(PageBreak())
    story.append(Paragraph(title,styles['Heading1'])); story.append(Spacer(1,8))
def table(rows,widths):
    t=Table([[Paragraph(str(c),styles['CellA']) for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DDEEF2')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F8FA')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#155E75'))])); story.append(t); story.append(Spacer(1,10))
page('Selecting Reliable LLM Reasoning')
p('<b>DASE7506 Mini-project 2 | Consolidated experimental report | 26 October 2026</b>')
h('Abstract')
p('We study selection among four supplied mathematical solutions using frozen embeddings, structural evidence and lightweight learned verifiers. A neural/PCA ensemble with narrow arithmetic correction (V2) scores 394/467 on the official test. A later text-semantic blend (V3), chosen for its higher hard-validation accuracy, scores 391/467. Additional architectures and an audit-driven calibration correction do not improve development accuracy. The requested 87% test target remains unmet. The main finding is that small gains on a repeatedly reused validation subset do not reliably transfer to test.')
table([['Frozen version','Hard validation','Official full test'],['V1: initial neural verifier','302/355 (85.0704%)','382/467 (81.7987%)'],['V2: projected ensemble + correction','307/355 (86.4789%)','394/467 (84.3683%)'],['V3: role-text + semantic blend','309/355 (87.0423%)','391/467 (83.7259%)']],[190,155,166])
h('Task and metric')
p('Each question has four candidate texts and fixed 1,024-dimensional representations. The method returns one index; the unchanged official parser and scorer assess the selected final numeric answer. Reasoning correctness is not directly scored. Every question has at least one correct candidate. Equivalent numeric fractions/decimals are handled by the supplied parser.')
p('The bank contains 4,755 training, 798 validation and 467 test questions. Only 355 validation groups have at least two distinct valid parsed answers; every test group meets that condition. V3\'s full-validation 752/798 = 94.2356% includes easier agreement groups and is not a leaderboard score.')
h('Status of the target')
p('At least 407/467 correct selections are needed to reach 87% (87.1520%); 406/467 is only 86.9379%. V2 and V3 are historical measurements for different checkpoints, not interchangeable score labels. This report does not claim a successful 87% method or select a replacement using test scores.')
small('Original benchmark: github.com/xudongwu-0/DASE7506-MP2. All data, supplied embeddings, parser, scorer and attribution are retained unchanged.')

page('1. Methods and frozen artifacts')
h('V2: compact neural ranking with arithmetic evidence')
p('The score combines 75% of a 64-unit neural verifier with 25% of a projected pairwise ensemble. The latter fits separate 96-component PCA transformations to raw and within-question centered training embeddings, then concatenates their projections with structural features. Three 64-unit ReLU networks, seeds 41/42/43 at epoch 60, are averaged. Learned normalization and PCA parameters are fit on training inputs only.')
p('Structural features encode answer validity and agreement, text length and formatting, lexical cues, numeric overlap and explicit arithmetic. Pairwise training penalizes a correct candidate receiving a lower score than an incorrect candidate in the same question. Invalid final answers are masked at selection.')
p('A weight-1 correction term uses a narrow pattern: an explicit numeric equation ends at a candidate\'s claimed answer, its left side evaluates differently, and that computed value equals another supplied candidate\'s final answer. It adds support to the matching candidate. It never generates a new candidate. The baseline arithmetic checker has incomplete-expression false positives; later stricter checks were investigated without silently changing frozen V2.')
h('V3: role-normalized text and semantic fusion')
p('Numeric tokens are represented by roles such as question quantity, proposed final value or derived value. A word n-gram TF-IDF logistic classifier uses these texts and a neural prior score. A second branch fits a 12,000-feature TF-IDF representation and a 64-dimensional SVD, then combines text coordinates and their within-question differences with supplied embeddings and structure in a 96-unit verifier. Three training seeds are averaged.')
p('The frozen blend uses 75% role-text scores and 25% semantic scores. Each component is centered within its valid candidate group and divided by the larger of its within-group standard deviation and 0.5. All components are stored in the checkpoint. The selected blend improved hard validation by two questions over V2, but the final test result declined by three questions.')
table([['Artifact','V2','V3'],['Entrypoint','selector_refined.py','selector_semantic_blend.py'],['Checkpoint directory','runs_v2/final_bundle','runs_v3/final_bundle'],['Assets, uncompressed','1,803,170 bytes','9,724,630 bytes'],['Dependencies for inference','NumPy','NumPy, SciPy, scikit-learn']],[120,191,200])
small('Architecture, preprocessing, checkpoints and original score JSON files are version-specific. Historical reports remain provenance records; this consolidated report supersedes their claims about which version is current.')

page('2. Development and evaluation protocol')
h('Training, development and independent scoring')
p('Training references provide final-answer labels; validation is used for method development. In later rounds, 423 of the 2,114 hard training questions form a fixed internal holdout using seed 731. For each learned family, configuration selection occurs there where implemented, followed by full-training refitting. Earlier rounds selected directly on official validation. These are development procedures, not independent estimates of final generalization.')
p('A separate prediction process receives only bank inputs, embeddings and checkpoint assets. The official scoring process reads references after choices are fixed. Source/checkpoint hashes were recorded before V2 and V3 scoring. IDs are used for alignment only. No per-question test errors were inspected for feature design or fitting.')
h('Test exposure is disclosed')
p('Three frozen versions have official test measurements: V1, V2 and V3. Earlier aggregate scores were already known during later research, so the test has not remained a pristine once-only holdout throughout the project. Nevertheless, later training/search used training and validation references, and no repeated test sweep was used to choose parameters. The small V3 development gain was frozen before its test score was observed.')
p('An earlier self-imposed rule to wait for 90% hard validation before another test was replaced, before observing the V3 score, by one final evaluation of the already validation-selected model. That decision and its rationale are recorded in runs_v3/frozen_manifest.json. Subsequent unsuccessful research did not trigger another test run.')
h('Why the uncertainty matters')
p('Many model families and configurations were compared on the same 355 hard validation questions. This creates selection optimism. V3 fixes three V2 validation errors and regresses on one; the exploratory paired-bootstrap gain interval is [-0.5634, 1.6901] percentage points and includes zero. It does not correct for the repeated search. The observed test decline is consistent with the lack of convincing development evidence, but does not establish a single causal explanation.')
p('Several calibration searches used precomputed out-of-fold teacher scores with a subsequent non-nested meta holdout. Each teacher prediction excludes its own question, but teacher models generating meta-training features may have seen other questions in that meta holdout. Such calibration holdout numbers are descriptive, not clean generalization evidence. Final teacher ensembles also differ in training size from individual OOF teachers.')
small('Resource limits: offline CPU inference, four total numerical-library threads, 30 seconds for fresh prediction plus scoring, and 16 MiB of uncompressed inference assets. Reference-machine certification is separate from local timing.')

page('3. Comparisons, ablations and failure modes')
table([['Selected family / experiment','Hard validation'],['Frozen V2','86.4789%'],['Answer-group pooling / nonlinear candidate pairs','85.9155% / 85.0704%'],['RBF kernel verifier / numerical-feature ranker','83.0986% / 80.8451%'],['Role-normalized text with prior','86.7606%'],['Semantic text + multitask verifier','85.6338%'],['Small learned attention / set transformer','84.2254% / 83.9437%'],['Text CNN, initialized from scratch','85.6338%'],['V3 role-text + semantic blend','87.0423%'],['Learned expert gate / anchored gate','85.3521% / 87.0423%'],['Pairwise training / semantic hard-only ranking','86.1972% / 85.9155%'],['Quantity-use calibration / eight-model ensemble','85.6338% / 84.7887%'],['Learned context-candidate interaction','85.9155%'],['Audit-aligned text prior / semantic blend','86.1972% / 86.1972%']],[365,146])
h('Controlled evidence rather than complexity alone')
p('Adding semantic scores to the role-text model at the frozen 25% weight changes hard validation from 308 to 309 correct; the 50% blend falls back to 307. Parameter-perturbation training compared radii 0, 0.05 and 0.15. The unperturbed and radius-0.15 variants tied on training holdout; the deterministic rule retained the unperturbed control. There is no demonstrated benefit from perturbation. An eight-seed ensemble and learned contextual interactions likewise failed to improve the retained model.')
small('Final bounded experiment: focal weighting of marginal-positive ranking loss compared exponents 0/1/2 on training holdout. The no-focusing control at epoch 50 was selected (81.56% holdout). Three-seed refitting scored 299/355 (84.23%) on hard validation. Its best fixed blend tied 309/355, fixing eight previous errors but introducing eight. Research stopped without another test evaluation.')
h('Coverage and reliability of arithmetic evidence')
p('Stricter numeric-equation parsing reduced false positive flags on correct training candidates from 9.75% to 1.58%, but also reduced coverage of wrong candidates from 14.09% to 5.30%. Its calibrated selector did not change V2 validation choices. Training-only error inspection found omitted factors, wrong percentage direction and wrong aggregation scope. Quantity-use checks corrected some errors but introduced more regressions. Some traces contain incorrect reasoning yet arrive at the correct numeric answer; a hard exclusion rule would conflict with the task metric.')
small('Full per-configuration results and runtimes are retained in the runs_* directories and ROUND3_RESEARCH.md through ROUND8_RESEARCH.md. These comparisons reuse development data and are not independent hypothesis tests.')

page('4. Audit findings and a focused correction')
h('Passed checks')
p('For train and validation, independently scoring each fixed candidate column reproduces the constructed label counts exactly. Every question has a positive candidate; no positive label is assigned to an invalid parsed answer. The audit finds 871 invalid training candidates and 157 invalid validation candidates, all correctly excluded from positive labels. Course data hashes pass on these splits.')
p('On 48 validation questions, candidate reordering changes V2 scores by at most 1.67e-6 and V3 by at most 1.41e-6 after restoring correspondence. Splitting the same questions into seven-question batches changes scores by zero in this check. Scores are finite. All frozen V2 and V3 source/checkpoint hashes match. These sampled checks do not prove absence of every possible defect, but give no evidence of a label, ordering or batching bug suppressing the results.')
h('Train/inference prior mismatch')
p('The V3 role-text classifier was trained on raw scores from single held-out members of a five-fold neural ensemble. At inference it received raw V2 scores, from a different model family including projected ranking and arithmetic correction. This changes the meaning and scale of a learned input. It is a methodological mismatch, not evidence of test leakage, and its causal effect on test performance is unknown.')
h('Audit-driven experiment')
p('A focused correction uses the same five-fold teacher family at training and inference and normalizes valid-candidate scores within each question. The role-text vocabulary remains training-fitted, and logistic regularization is fixed at C=1; no new hyperparameter sweep is performed. The final prior averages the five teacher models. A fixed 25% semantic blend is also checked. No additional test evaluation is performed.')
table([['Version','Hard validation','Changes vs V3'],['Frozen V3','309/355 = 87.0423%','Reference'],['Aligned teacher + normalized prior','306/355 = 86.1972%','3 fixed, 6 regressed'],['Aligned prior + 25% semantic','306/355 = 86.1972%','2 fixed, 5 regressed']],[220,155,136])
p('The correction improves conceptual consistency but not observed validation accuracy. It is therefore retained as an experimental result, not substituted into a scored checkpoint. Single-model OOF features and averaged deployment features still differ in variance. Neither this correction nor the audit justifies claiming that the score gap has been solved.')
h('Additional supervision from training reference solutions')
p('A separate experiment predicts reference-solution operation counts, step count and answer-scale properties as auxiliary training targets. References are absent at inference. The training holdout selects weight 1 and 20 epochs (82.51%, versus the control\'s best 81.32%). Three-seed refitting scores 305/355 on hard validation. A 25% blend ties V3 at 309/355, fixing one error and regressing on one. It is not retained as an improvement.')
small('Evidence: code/runs_audit/audit.json, runs_aligned/result.json and runs_reference_aux/result.json. These experiments read no test references. The auxiliary head is removed from inference assets.')

page('5. Resources, reproduction and disclosure')
table([['Measurement','V2','V3'],['Full-test correct / total','394 / 467','391 / 467'],['Official accuracy','84.3683083512%','83.7259100642%'],['Fresh prediction + scoring','1.0539 s','4.5102 s'],['Uncompressed assets','1.7196 MiB','9.2741 MiB'],['Local hardware / threads','Intel Core Ultra 7 251HX / 4','Same / 4']],[205,153,153])
p('Local measurements include fresh process startup, imports, input and asset loading, preprocessing and independent official scoring. Both tested versions meet local time and size limits. The Xeon Platinum 8457C reference CPU determines official eligibility; local timings are not certification. Training used CPU. The CNN and set-attention runs used torch 2.6.0+cpu, with no pretrained model weights; their measured research times were about 380 and 179 seconds. Some independent jobs overlapped; per-job timers should not be summed as total wall-clock time.')
h('Reproduction map')
p('From code/, install requirements-method.txt for V3 inference and requirements-round5.txt for later neural research. README_STATUS.md provides exact prediction/scoring commands and checkpoint-to-score mappings. Original V2 search commands remain in README_V2.md as historical reproduction instructions. Preserved checkpoints are required to reproduce reported scores without retraining; training commands may overwrite development outputs and should run in a copy.')
small('<b>V2:</b> selector_refined.py with runs_v2/final_bundle; result runs_v2/test-score.json.<br/><b>V3:</b> selector_semantic_blend.py with runs_v3/final_bundle; result runs_v3/test-score.json.<br/><b>Audit:</b> python audit_pipeline.py<br/><b>Focused correction:</b> python train_aligned.py<br/><b>Frozen evidence:</b> each runs_v2/runs_v3 frozen_manifest.json and resources.json.')
h('Conclusion and scope')
p('The compact V2 method has the highest historical test measurement among the three evaluated checkpoints. V3 was selected on development evidence but did not improve test performance. The project supplies reproducible methods, failed comparisons, resource measurements and an explicit audit; it does not supply a method demonstrated to achieve 87% or 90%. Further development would require stronger evidence than another small gain on the repeatedly reused validation subset.')
h('AI assistance and attribution')
p('OpenAI Codex substantially assisted with interpreting requirements, designing and implementing methods, running experiments and the audit, and drafting this report. The student should understand and review the implementation and findings before submission. No additional pretrained weights, external training examples or LLM inference API calls were used. The original course repository, licenses and attribution are preserved.')
small('<b>Sources:</b> Course guide: https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md<br/>Running instructions: https://github.com/xudongwu-0/DASE7506-MP2/blob/main/code/README.md<br/>Results and checks: local runs_* JSON logs and frozen manifests. Online submission/publication has not been performed.')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#D3E0E5')); canvas.line(42,35,553,35); canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#526575')); canvas.drawString(42,23,'DASE7506 MP2 | Consolidated results and audit | 26 October 2026'); canvas.drawRightString(553,23,str(doc.page))
path=root/'report_consolidated.pdf'
SimpleDocTemplate(str(path),pagesize=(595.28,841.89),leftMargin=42,rightMargin=42,topMargin=38,bottomMargin=46).build(story,onFirstPage=footer,onLaterPages=footer)
n=len(PdfReader(path).pages); assert n==6,n
print('Created',path,'pages',n)
