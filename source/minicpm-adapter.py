#!/opt/venv/bin/python
"""MiniCPM Python serve adapter for a source-built native managed benchmark.

The native parent resolves public pinned snapshots, probes local hardware, measures
the unchanged suite, supervises this child and writes the receipt. No attached
receipt is edited or upgraded. This child execs the actual Python server.
"""
import json
import os
import pathlib
import platform
import sys

TARGET_REV = '0aee82655c3bdb7135099cd368af7e92235d066f'
MTP_REV = 'f84b054c5677b6e59bdb969ae61915b0983d68bb'

def facts():
    import tensorfold
    from importlib.metadata import version
    return {'tensorfold_version': tensorfold.__version__, 'python_version': platform.python_version(),
            'model_type': 'minicpmv4_7', 'torch': version('torch'), 'triton': version('triton'),
            'target_revision': TARGET_REV, 'mtp_revision': MTP_REV,
            'controller': 'TensorFold native community-v1, 1.0.12+frontier-b4'}

def prepare(argv):
    if len(argv) < 2 or argv[0] != 'serve':
        raise ValueError('This adapter accepts only managed serve or --facts')
    target = pathlib.Path(argv[1])
    # Resolve only the container's offline HF snapshot, never a mutable source dir.
    if target.name != TARGET_REV or target.parent.name != 'snapshots':
        raise ValueError('The managed MiniCPM target must be its pinned public snapshot')
    if json.loads((target / 'config.json').read_text()).get('model_type') != 'minicpmv4_7':
        raise ValueError('Not the supported MiniCPM model type')
    out = list(argv)
    for flag, value in [('--host', '127.0.0.1'), ('--port', '0'), ('--parallel', '1'),
                        ('--snapshot-dir', 'none'), ('--max-snapshots', '0'), ('--prompt-cache-gib', '0')]:
        if flag not in out or out[out.index(flag)+1] != value:
            raise ValueError('Managed benchmark requires ' + flag + ' ' + value)
    env = os.environ.copy()
    env['TF_BENCHMARK_NO_PREFIX_REUSE'] = '1'
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    if '--no-drafts' in out:
        if '--drafter' in out:
            raise ValueError('Serial managed runs must not pass a drafter')
        env['TF_QWEN36_MTP'] = '0'
    else:
        if '--drafter' not in out:
            raise ValueError('MiniCPM requires its explicitly pinned MTP sidecar')
        at = out.index('--drafter')
        mtp = pathlib.Path(out[at+1])
        if mtp.name != MTP_REV or mtp.parent.name != 'snapshots' or not (mtp / 'mtp-4bit.safetensors').is_file():
            raise ValueError('The MTP head must resolve to its pinned public snapshot')
        env['TF_QWEN36_MTP'] = str(mtp / 'mtp-4bit.safetensors')
        out[at+1] = 'none'  # The fork loads this verified MTP head, not an external draft engine.
        depth = int(env.get('TF_MINICPM_MTP_DRAFTS', '4'))
        if not 1 <= depth <= 15:
            raise ValueError('MTP depth must be 1..15')
        out += ['--mtp-drafts', str(depth)]
        confidence = float(env.get('TF_MINICPM_CONFIDENCE', '0.3'))
        if not 0 <= confidence <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        out += ['--mtp-confidence', str(confidence)]
    # The fixed suite sets thinking=false for chat. The server default matches it.
    out += ['--no-thinking']
    return out, env

def main():
    if sys.argv[1:] == ['--facts']:
        print(json.dumps(facts()))
        return
    argv, env = prepare(sys.argv[1:])
    os.execve(sys.executable, [sys.executable, '-m', 'tensorfold', *argv], env)

if __name__ == '__main__':
    main()
