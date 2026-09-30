"""Export complete replication evidence with per-file SHA256 verification."""
import sys,zipfile,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import cold_start_coverage as study
from roopf.experiment_io import sha256,write_json
study.verify();assert (study.OUT/'VERIFICATION.json').exists()
dest=ROOT/'artifacts/cold_start_coverage_v1';dest.mkdir(exist_ok=True)
files=sorted(set([p for d in (study.RUN,study.OUT) for p in d.rglob('*') if p.is_file() and p.suffix!='.lock']+[ROOT/p for p in study.SOURCES]+[Path(__file__),ROOT/'scripts/verify_cold_start_coverage.py']))
manifest={str(p.relative_to(ROOT)):sha256(p) for p in files};parts={}
for k in range(0,len(files),100):
    zp=dest/f'cold_start_coverage_v1.part{k//100+1:03d}.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files[k:k+100]:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None
        for name in z.namelist():assert hashlib.sha256(z.read(name)).hexdigest()==manifest[name]
    parts[zp.name]=sha256(zp)
write_json(dest/'MANIFEST.json',dict(files=manifest,archives=parts,parent='artifacts/cold_start_v1',new_calls=1036845))
(dest/'README.md').write_text('''# Cold-start coverage v1

先恢复 `artifacts/cold_start_v1` 及其列明的父依赖，再将本目录全部ZIP解压到仓库根目录。包括1728条完整轨迹、冻结协议/代码、结果与零调用重放核验；MANIFEST逐文件和分包记录SHA256。

```bash
.venv/bin/python scripts/verify_cold_start_coverage.py
.venv/bin/python scripts/cold_start_coverage.py report
```

以上复核不新增目标调用。本轮新运行1036845次评估、0轮神经训练。固定10个Sobol点与20个融合选点的早期分配，40次后退出，不能称未见函数族确认。原冷启动开发结果保持独立。
''')
print(f'Archived {len(files)} files in {len(parts)} parts',flush=True)
