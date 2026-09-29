"""Run and audit the four final suites, then collect their thesis artifacts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
JOBS = [
    ('all_insects_comparison', ['comparison', 'friedman', 'transition']),
    ('ablation', ['summary', 'ablation', 'transition']),
    ('sensitivity', ['summary', 'tables', 'figures', 'transition']),
    ('additional_datasets', ['comparison', 'friedman', 'transition']),
]
status = {'state': 'running', 'suites': {}}
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
def save():
    (PACKAGE / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
save()
try:
    for name, sections in JOBS:
        config = ROOT / 'experiments/configs' / (name+'.json')
        specification = json.loads(config.read_text())
        suite = ROOT / 'model/data/thesis_experiments' / specification['name']
        status['suites'][name] = {'state':'training', 'suite':str(suite)}
        save()
        with (PACKAGE / (name+'.log')).open('a') as log:
            subprocess.run([sys.executable, 'model/experiments/run_experiment_suite.py',
                            str(config), '--resume', '--jobs', '3'], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            status['suites'][name]['state']='auditing_and_reporting';save()
            subprocess.run([sys.executable, 'model/analysis/report.py', str(suite),
                            '--config', str(config), '--sections', 'audit', *sections],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        target=PACKAGE/name
        shutil.copytree(suite/'report', target, dirs_exist_ok=True)
        status['suites'][name]['state']='complete';save()
    status['state']='complete';save()
except Exception as error:
    status['state']='failed';status['error']=str(error);save()
    raise
