"""Lossless cached P32 extraction, no labels, scores, forward, or training."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hybrid_tsad.hybrid_v1.p32_readout import cache_bank


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--trusted-config-sha256', required=True)
    args = parser.parse_args()
    if sha(args.config) != args.trusted_config_sha256:
        raise PermissionError('Frozen origin-cache config required')
    config = json.loads(Path(args.config).read_text())
    for path, digest in config['source_sha256'].items():
        if sha(path) != digest:
            raise PermissionError('Frozen lossless extractor changed')
    if sha(Path(config['bank_directory']) / 'manifest.json') != config['bank_manifest_sha256']:
        raise PermissionError('Original completed P32 bank changed')
    result = cache_bank(config['bank_directory'], config['output'], expected_checkpoint=config['checkpoint_sha256'], role='QUERY')
    print(json.dumps({key: result[key] for key in ('status', 'seconds', 'records', 'channels', 'h1_native_formula_float64_check')}))


if __name__ == '__main__':
    main()
