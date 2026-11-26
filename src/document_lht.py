from pathlib import Path
import json,collections,re
ROOT=Path(__file__).resolve().parents[1]
def main():
    d=json.loads((ROOT/'exp/lht/reproducibility.json').read_text());failed=d['failures'];ok=d['results']
    nonfinite=sum('nonfinite' in r['error'] for r in failed);timeout=sum('TimeoutExpired' in r['error'] for r in failed)
    assert ok+nonfinite+timeout==175
    s=(ROOT/'main.tex').read_text(encoding='utf-8')
    marker='\\section{Task composition and independent mechanism replication}'
    addition=rf'''
\section{{A recent implementation comparison and its limits}}\label{{app:lht}}
We examined the official Learning Hyperplane Tree implementation associated with \citet{{lht2025}}. A C++ input adapter supplies exactly the same preprocessed development/test partitions as the other methods, uses one one-versus-rest tree per class, fixes the feature factor parameter to one, and selects the minimum split size from $\{{8,16,32,64\}}$ by validation balanced accuracy. Predictions use the implementation's weighted class scores. A 30-second subprocess limit is imposed for each candidate. This is an implementation diagnostic, separate from the primary comparison; it does not introduce an independently verified mathematical reproduction of the method.

Of 175 task/split attempts, {ok} produced finite outputs for every selected comparison stage; {nonfinite} encountered nonfinite scores and {timeout} exceeded the per-candidate time limit. No missing result is treated as zero accuracy, and no surviving subset is used to claim a full-benchmark advantage. Source inspection identified a floating-point predictor assigned to an integer when accumulating leaf moments, moment accumulators reused across coordinates, and centroid denominators inconsistent with the stated sample-mean interpretation. These concerns limit the fidelity of a direct implementation comparison even when a particular output is finite. They do not establish that the method described in the paper is ineffective. A faithful comparison requires resolution against the intended equations and validation of any corrections before the method can be included in an accuracy ranking.
'''
    assert marker in s;s=s.replace(marker,addition+'\n'+marker)
    s=s.replace('they do not reproduce all recent methods such as LHT, CART-ELC, optimal oblique satisfiability trees or feature-concatenation approaches.',
      'they do not provide a fully validated mathematical reproduction of recent methods such as LHT, CART-ELC, optimal oblique satisfiability trees or feature-concatenation approaches. The official LHT implementation diagnostic in Appendix~\\ref{app:lht} identifies numerical and timeout obstacles without treating them as accuracy losses.')
    s=s.replace('\\begin{table}[p]','\\begin{table}[htbp]')
    (ROOT/'main.tex').write_text(s,encoding='utf-8')
    inventory=json.loads((ROOT/'research/lht_code_inventory.json').read_text())
    header=next(r for r in inventory if r['name']=='LHT.h')
    (ROOT/'research/lht_reproduction.md').write_text(f'''# Official LHT implementation diagnostic\n\nSource: {header['html_url']}\nGit blob SHA: {header['sha']}\nRetrieved: 2026-10-03. Original header retained unchanged under research/lht_source.\n\n133 finite task/split outputs, 36 nonfinite-score cases and 6 timeouts (30 seconds per candidate). These are implementation outcomes, not evidence that mathematical LHT loses to P2. They are excluded from the primary ranking and inferential family.\n\nThe adapter follows weighted one-vs-rest class scoring, uses one tree per class, beta=1, forest_rate=1, and validation-selects minimum split size 8,16,32,64. The original header has no unified depth limit comparable to the main tree grid. Scores are normalized only for exploratory probability summaries, not claimed calibrated probabilities.\n\nNumerical concerns:\n1. growTree declares `long value` and assigns a floating predictor to it.\n2. Ex2/Exy are not reset per active coordinate, and Ey is divided repeatedly within that loop.\n3. initCentroidSampleNumbers uses `accuC / forest_rate * feature_train_size` instead of a sample-mean denominator.\n4. splitSB increments Sn/Dn inside the coordinate loop, so a child centroid denominator includes the feature count.\n\nNo numerical correction was silently applied. A corrected implementation would be an explicitly documented adaptation requiring validation against intended equations. No baseline superiority claim is made from the diagnostic. Upstream redistribution/license rights have not been established; retain locally for provenance, do not assert a license or publish third-party code automatically.\n''',encoding='utf-8')
if __name__=='__main__':main()
