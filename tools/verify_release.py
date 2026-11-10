from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'checksums.json').read_text())
bad=[]
for name,digest in manifest.items():
    p=root/name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:bad.append(name)
if bad:raise SystemExit('Checksum mismatch: '+', '.join(bad))
print(f'OK: {len(manifest)} release files verified')
