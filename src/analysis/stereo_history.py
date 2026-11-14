"""用不可变的运行源码快照核验既有coverage结果，不恢复旧执行入口。"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

from src.stereo.protocol import sha256

ROOT = Path(__file__).resolve().parents[2]


def verify_archived_coverage(output):
    output = Path(output).resolve()
    record = json.loads((output / 'run_metadata.json').read_text())
    if record['config'].get('protocol_status') != 'middlebury_course_v1':
        raise ValueError('Only the recorded historical coverage protocol can use snapshots')
    with tempfile.TemporaryDirectory(prefix='middlebury-verify-') as folder:
        sandbox = Path(folder)
        for name, digest in record['source_sha256'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid historical source path')
            original = output / 'provenance/source' / relative
            if sha256(original) != digest:
                raise ValueError('Historical source snapshot changed: ' + name)
            destination = sandbox / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
        # 此依赖未被原运行归档；使用项目公共数组格式验证器，不调用模型。
        destination = sandbox / 'src/data/output_schema.py'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'src/data/output_schema.py', destination)
        for package in ('src', 'src/data', 'src/analysis', 'src/visualization', 'pipelines', 'pipelines/part1'):
            p = sandbox / package / '__init__.py'
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True); p.write_text('')
        (sandbox / 'data').symlink_to(ROOT / 'data', target_is_directory=True)
        code = (
            'import json,sys; from pathlib import Path; '
            'from pipelines.part1.study_middlebury_coverage import verify; '
            'p=Path(sys.argv[1]); r=json.loads((p/"run_metadata.json").read_text()); '
            'report=verify(p,r["settings"],r["config"]); '
            'Path(sys.argv[2]).write_text(json.dumps(report))'
        )
        result = subprocess.run([sys.executable, '-c', code, str(output), str(sandbox / 'report.json')],
                                cwd=sandbox, env={**os.environ, 'PYTHONPATH': '', 'PYTHONDONTWRITEBYTECODE': '1'},
                                text=True, capture_output=True)
        if result.returncode:
            raise ValueError('Archived coverage verification failed:\n' + result.stderr)
        report = json.loads((sandbox / 'report.json').read_text())
    return {**report, 'historical_source_replay': True,
            'archived_source_files_checked': len(record['source_sha256'])}
