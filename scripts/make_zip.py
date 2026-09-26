#!/usr/bin/env python3
"""Create a deterministic supplement ZIP from the checked public tree."""
from __future__ import annotations
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'RELEASE_MANIFEST.json'

def main():
    check=subprocess.run([sys.executable,str(ROOT/'scripts/release_manifest.py'),'--check'],cwd=ROOT)
    if check.returncode: return check.returncode
    files=[row['path'] for row in json.loads(MANIFEST.read_text())['files']]+['RELEASE_MANIFEST.json']
    out=ROOT.parent/'ca-worldmodels-release.zip'
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(files):
            info=zipfile.ZipInfo('ca-worldmodels-release/'+name,date_time=(2026,9,26,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            z.writestr(info,(ROOT/name).read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(out) as z:
        bad=z.testzip()
        if bad: raise RuntimeError('ZIP CRC failed: '+bad)
        assert len(z.namelist())==len(files)
    print(out, out.stat().st_size, 'bytes',len(files),'files')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
