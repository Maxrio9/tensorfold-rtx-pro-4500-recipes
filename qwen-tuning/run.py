"""Explicitly launch one qualified Qwen service or standard benchmark; never restore a model."""
import argparse, hashlib, json, os, pathlib, signal, subprocess, time

HERE = pathlib.Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['serve', 'benchmark'])
    p.add_argument('--engine', required=True, type=pathlib.Path)
    p.add_argument('--helpers', required=True, type=pathlib.Path)
    p.add_argument('--target', type=pathlib.Path, help='Pinned target snapshot, required for serve')
    p.add_argument('--drafter', type=pathlib.Path, help='Pinned drafter snapshot, required for serve')
    p.add_argument('--hf-cache', type=pathlib.Path, help='Pinned Hugging Face hub cache, required for benchmark')
    p.add_argument('--output', type=pathlib.Path, default=pathlib.Path('new-benchmark.json'))
    p.add_argument('--gpu', type=int, default=0, help='Physical nvidia-smi GPU index')
    p.add_argument('--allow-desktop', action='store_true', help='Allow known desktop render contexts; other compute processes still block startup')
    a = p.parse_args()
    if a.mode == 'serve' and not (a.target and a.drafter):
        p.error('serve requires --target and --drafter pinned snapshots')
    if a.mode == 'benchmark' and not a.hf_cache:
        p.error('benchmark requires --hf-cache with refs/main pinned as documented')
    recipe = json.loads((HERE / 'recipe.json').read_text())
    def smi(*args):
        return subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), *args], text=True).strip()
    desktop = {'chromium', 'chrome', 'voxtype-osd-gtk4', 'nautilus'}
    apps = smi('--query-compute-apps=process_name', '--format=csv,noheader,nounits').splitlines()
    busy = [name for name in apps if name.strip() and (not a.allow_desktop or pathlib.Path(name.split()[0]).name not in desktop)]
    if busy:
        raise SystemExit('GPU has another compute process; wait for an exclusive test slot.')
    def memory():
        return float(smi('--query-gpu=memory.used', '--format=csv,noheader,nounits'))
    baseline = memory()
    env = os.environ.copy()
    env.update(recipe['environment'])
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_MODULE_LOADING='LAZY',
               CUDA_VISIBLE_DEVICES=str(a.gpu), TENSORFOLD_CUDA_KERNELS=str(a.helpers.resolve()))
    engine = a.engine.resolve()
    if a.mode == 'serve':
        replacements = {'${TENSORFOLD_BINARY}': str(engine), '/models/qwen-tf-native-4bit': str(a.target.resolve()),
                        '/models/qwen-draft-bf16': str(a.drafter.resolve())}
        argv = [replacements.get(x, x) for x in recipe['server_argv']]
    else:
        env['HF_HUB_CACHE'] = str(a.hf_cache.resolve())
        argv = [str(engine), 'benchmark', recipe['target']['repository'], '--backend', 'cuda',
                '--drafter', recipe['drafter']['repository'], '--context', '2048', '--output', str(a.output)]
        a.output.with_suffix('.provenance.json').write_text(json.dumps({
            'recipe_sha256': hashlib.sha256((HERE/'recipe.json').read_bytes()).hexdigest(),
            'engine_binary_sha256': hashlib.sha256(engine.read_bytes()).hexdigest(),
            'engine_base_commit': recipe['engine_base_commit'],
            'engine_patch_sha256': recipe['profile']['build_patch_sha256'],
            'modified_engine': True, 'publication': 'Local measurement only; no upload performed'
        }, indent=2)+'\n')
    child = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        child = subprocess.Popen(argv, env=env, start_new_session=True)
        while child.poll() is None:
            if memory() - baseline >= 26000:
                raise RuntimeError('26000 MiB sampled additional-memory guard reached; stopping owned process group')
            time.sleep(0.2)
        if child.returncode:
            raise SystemExit(child.returncode)
    finally:
        if child is not None:
            # Clean up only the group created above, including a managed benchmark server.
            try:
                os.killpg(child.pid, signal.SIGTERM)
                if child.poll() is None:
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

if __name__ == '__main__':
    main()
