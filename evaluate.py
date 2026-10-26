import json
import numpy as np
from modeling import train_all, explain, SPECS, ROOT

d,bundles=train_all()
results={}
for name,b in bundles.items():
 assert not set(d.iloc[b['train']].household_id)&set(d.iloc[b['test']].household_id)
 row=d.loc[[b['test'][0]],SPECS[name]['features']]
 ref,local=explain(b,row)
 pred=b['pipe'].predict_proba(row)[0,1]
 assert np.isclose(1/(1+np.exp(-(ref+local.log_odds_contribution.sum()))),pred)
 assert np.isclose(pred,b['pipe'].predict_proba(row.copy())[0,1])
 b['importance'].to_csv(ROOT/'docs'/f"importance_{SPECS[name]['target']}.csv",index=False)
 results[name]=b['metrics']
(ROOT/'docs/model_metrics.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
print('PASS: household isolation, local explanation reconstruction, unchanged-input consistency for both models')
