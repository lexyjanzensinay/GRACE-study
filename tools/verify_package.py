from pathlib import Path
import ast,csv,hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[1]
def main():
 manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
 expected={x['path'] for x in manifest['files']}
 actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts and '.git' not in p.parts}
 assert actual==expected, ('Unexpected or missing files',sorted(actual^expected))
 for x in manifest['files']:
  p=ROOT/x['path'];b=p.read_bytes();assert len(b)==x['bytes'];assert hashlib.sha256(b).hexdigest()==x['sha256'],x['path']
  if p.suffix=='.py':ast.parse(b.decode('utf-8-sig'))
 with (ROOT/'results/policy_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 assert len(rows)==2112
 gated=[x for x in rows if x['policy']=='gated_Bayes'];assert len(gated)==576
 assert len({(x['cell'],x['feature'],x['target'],x['predictor']) for x in gated})==192
 assert {int(x['n']) for x in rows}=={252}
 residual=max(abs(float(x['total_cost'])-float(x['classification_cost'])-float(x['delay_cost'])) for x in rows);assert residual<1e-12
 for target,cost in [('Binary',.5410),('Ternary',1.5345)]:
  vals=[float(x['total_cost']) for x in gated if x['target']==target];assert len(vals)==288;assert abs(statistics.mean(vals)-cost)<.00005
 with (ROOT/'results/training_reference_metrics.csv').open(encoding='utf-8-sig',newline='') as f:prior=list(csv.DictReader(f))
 assert len(prior)==1152
 assert len(json.loads((ROOT/'docs/configuration_records.json').read_text())['records'])==192
 print(json.dumps({'status':'PASS','files':len(expected),'policy_rows':len(rows),'gated_rows':len(gated),'training_reference_rows':len(prior),'max_cost_identity_residual':residual,'model_fitting_run':False,'raw_data_verified':False},indent=2))
if __name__=='__main__':main()
