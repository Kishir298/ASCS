"""Fail-closed Pi generation adapter. No agent tools or arbitrary code execution."""
from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

GIB = 1024 ** 3


class Blocked(RuntimeError):
    pass


class QueueBusy(Blocked):
    pass


@dataclass
class Resources:
    available: int
    cpu_percent: float
    memory_pressure: float
    temperature: float
    load: float
    cores: int


def effective_available(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')) -> int:
    values = dict(line.split(':', 1) for line in (proc / 'meminfo').read_text().splitlines())
    available = int(values['MemAvailable'].split()[0]) * 1024
    # Unified cgroup membership, including nested service/container limits.
    membership = (proc / 'self/cgroup').read_text().splitlines()
    unified = next((x.split(':', 2)[2] for x in membership if x.startswith('0::')), None)
    if unified is None:
        raise Blocked('cgroup v2 required for this experimental Pi adapter')
    current = cgroup / unified.lstrip('/')
    # A cgroup namespace may expose its own root rather than host membership.
    if not current.is_dir():
        current = cgroup
    if not (current / 'memory.max').exists():
        raise Blocked('Cannot establish cgroup memory limit')
    while True:
        limit_path = current / 'memory.max'
        if limit_path.exists():
            limit = limit_path.read_text().strip()
            if limit != 'max':
                used = int((current / 'memory.current').read_text())
                available = min(available, max(0, int(limit) - used))
        if current == cgroup:
            break
        current = current.parent
        if not current.is_relative_to(cgroup):
            raise Blocked('Invalid cgroup hierarchy')
    return available


def _cpu():
    counts = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    return sum(counts), counts[3] + counts[4]


def sample_resources() -> Resources:
    samples = []
    previous = _cpu()
    for _ in range(10):
        time.sleep(1)
        current = _cpu()
        total, idle = current[0] - previous[0], current[1] - previous[1]
        if total <= 0:
            raise Blocked('CPU sampling unavailable')
        samples.append(100 * (1 - idle / total))
        previous = current
    pressure = Path('/proc/pressure/memory').read_text().splitlines()[0]
    psi = float(re.search(r'avg10=([\d.]+)', pressure)[1])
    temperature = float(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000
    return Resources(effective_available(), sum(samples) / len(samples), psi,
                     temperature, os.getloadavg()[0], os.cpu_count() or 1)


def admission(r: Resources, parameters: int, peak_gib: float, resident: bool, resident_bytes: int, context: int):
    if not all(math.isfinite(x) for x in (r.available, r.cpu_percent, r.memory_pressure, r.temperature, r.load, peak_gib)):
        raise Blocked("Unknown/non-finite resource measurements")
    if context not in (2048, 4096):
        raise Blocked('Context must be 2048, or separately benchmarked 4096')
    if parameters <= 600_000_000:
        gate, minimum_peak = 2.5, 1.5
    elif parameters <= 1_600_000_000:
        gate, minimum_peak = 3.5, 2.5
    elif parameters <= 3_200_000_000:
        gate, minimum_peak = 5.0, 3.5
    else:
        raise Blocked('Model exceeds automatic 3B-class ceiling; no substitution performed')
    if not (minimum_peak <= peak_gib <= 32):
        raise Blocked('Conservative measured peak budget is missing or too small')
    # Resident bytes are already deducted from available RAM; credit them once.
    additional = max(0, int(peak_gib * GIB) - (resident_bytes if resident else 0))
    budget = r.available - int(1.5 * GIB)
    reasons = []
    if r.available + (resident_bytes if resident else 0) < gate * GIB or additional > budget:
        reasons.append('insufficient effective RAM/headroom')
    if r.cpu_percent > 70 or r.load > r.cores * 1.5:
        reasons.append('sustained CPU/load pressure')
    if r.memory_pressure > 10 or r.temperature >= 75:
        reasons.append('memory/thermal pressure')
    return {'state': 'queued' if reasons else 'allowed', 'reasons': reasons,
            'available_gib': round(r.available / GIB, 3), 'reserve_gib': 1.5,
            'additional_gib': round(additional / GIB, 3), 'context': context,
            'threads': 2, 'resources': asdict(r)}


@contextmanager
def inference_slot(path: Path, wait: float = 30):
    import fcntl
    # Directory must be provisioned by the administrator, never world-writable.
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    start = time.monotonic()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start >= wait:
                    raise QueueBusy('Inference queue busy; retry later')
                time.sleep(0.1)
        yield
    finally:
        os.close(fd)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise Blocked('Ollama redirect refused')


class LocalOllama:
    def __init__(self, url):
        if url not in ('http://127.0.0.1:11434', 'http://localhost:11434', 'http://[::1]:11434'):
            raise Blocked('Pi mode requires an explicit loopback Ollama endpoint')
        self.url = url
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, path, body=None, timeout=10):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={'Content-Type': 'application/json'})
        with self.opener.open(req, timeout=timeout) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise Blocked('Ollama response exceeds bounded size')
        return json.loads(raw)


def deployment_supported():
    return sys.platform == "linux"


def enabled():
    return os.environ.get("ASCS_PI_MODE") == "1" or (deployment_supported() and platform.machine().lower() in ("aarch64", "arm64"))


def run(args):
    """Owner-only orchestration is CORE's responsibility; this endpoint has no tools."""
    try:
        if not deployment_supported():
            raise Blocked("Pi deployment requires Linux with cgroup v2")
        if getattr(os, "geteuid", lambda: 1)() == 0:
            raise Blocked("Pi inference must run as a dedicated unprivileged account")
        if args.task and len(args.task) > 32000:
            raise Blocked("Task exceeds 32000-character bound")
        if args.ui or args.tui or args.tasks:
            raise Blocked('Pi mode is headless generation-only; agent tools/TUI are not enabled')
        if not args.model:
            raise Blocked('Select an exact approved installed --model; no automatic model choice')
        if not (args.pi_status or args.task and args.task.strip()):
            raise Blocked('A nonempty task is required; interactive prompts are disabled')
        workspace = Path(args.workspace or '.').expanduser().resolve(strict=True)
        if not workspace.is_dir():
            raise Blocked('Workspace must be an existing directory')
        context = args.num_ctx or 2048
        manifest = json.loads(Path(os.environ.get('ASCS_PI_MANIFEST', '/etc/risarms/ascs-models.json')).read_text())
        approved = manifest[args.model]
        if approved.get('generation_verified') is not True or context not in approved.get('contexts', []):
            raise Blocked('Exact model/context has not been verified by the operator')
        with inference_slot(Path(os.environ.get('RISARMS_INFERENCE_LOCK', '/run/risarms-inference/queue.lock'))):
            client = LocalOllama(args.base_url or 'http://127.0.0.1:11434')
            tags = client.request('/api/tags')['models']
            tag = next((x for x in tags if x['name'] == args.model), None)
            if not tag or not re.fullmatch(r'(sha256:)?[0-9a-f]{64}', str(tag.get('digest', ''))) or tag['digest'] != approved['digest']:
                raise Blocked('Installed exact tag/digest does not match approval; no download attempted')
            if tag.get('remote_host') or tag.get('remote_model'):
                raise Blocked('Remote/cloud-backed model tags are disabled in Pi local mode')
            show = client.request('/api/show', {'model': args.model})
            if not show.get('template') or 'completion' not in show.get('capabilities', []):
                raise Blocked('Model chat protocol metadata is unsupported')
            quantization = show['details']['quantization_level']
            if not re.fullmatch(r'Q[45]_[A-Z0-9_]+|Q[45]_0', quantization):
                raise Blocked('Only explicitly approved Q4/Q5 quantization is admitted')
            if quantization != approved['quantization']:
                raise Blocked('Quantization differs from the approved benchmark')
            parameters = show['model_info']['general.parameter_count']
            running = client.request('/api/ps')['models']
            previous = [x for x in running if x.get('digest') != tag['digest']]
            if previous:
                if not getattr(args, 'pi_unload_previous', False):
                    raise Blocked('Different model resident; request --pi-unload-previous explicitly on the managed runtime')
                if any(manifest.get(x.get('name'), {}).get('digest') != x.get('digest') for x in previous):
                    raise Blocked('Unapproved resident model; no unrelated model will be unloaded')
                for model in previous:
                    client.request('/api/generate', {'model': model['name'], 'keep_alive': 0})
                running = client.request('/api/ps')['models']
                if any(x.get('digest') != tag['digest'] for x in running):
                    raise Blocked('Previous model did not unload; no inference started')
            resident = next((x for x in running if x.get('digest') == tag['digest']), None)
            status = admission(sample_resources(), parameters, approved['peak_gib'], bool(resident),
                               int(resident.get('size', 0)) if resident else 0, context)
            status.update(model=args.model, digest=tag['digest'], execution='generation-only')
            print(json.dumps(status), flush=True)
            if args.pi_status:
                return 0 if status['state'] == 'allowed' else 75
            if status['state'] != 'allowed':
                return 75
            # Memory recheck immediately before allocation; never retry/switch mid-job.
            status = admission(sample_resources(), parameters, approved['peak_gib'], bool(resident),
                               int(resident.get('size', 0)) if resident else 0, context)
            if status['state'] != 'allowed':
                print(json.dumps(status), flush=True)
                return 75
            result = client.request('/api/chat', {'model': args.model, 'stream': False,
                'keep_alive': 0, 'options': {'num_ctx': context, 'num_thread': 2, 'num_predict': 1024},
                'messages': [{'role': 'system', 'content': 'Generate an explanation or code as text only. You have no tools, file access, or command execution. Do not claim edits or tests were performed.'},
                             {'role': 'user', 'content': args.task}]}, timeout=120)
            if result.get('done') is not True or result.get('message', {}).get('tool_calls'):
                raise Blocked('Incomplete response or unsupported model tool call')
            content = result.get('message', {}).get('content')
            if not isinstance(content, str) or not content.strip():
                raise Blocked('Model returned no code/text')
            print('CODE GENERATION ONLY — no project files changed and no commands executed.', flush=True)
            print(content, flush=True)
            return 0
    except QueueBusy as exc:
        print(json.dumps({'state': 'queued', 'reason': str(exc)}), flush=True)
        return 75
    except KeyboardInterrupt:
        print('Cancelled; no automatic retry or model substitution.', file=sys.stderr, flush=True)
        return 130
    except (Blocked, OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        print(json.dumps({'state': 'blocked', 'reason': str(exc)}), flush=True)
        print(f'Pi job blocked/failed: {exc}', file=sys.stderr, flush=True)
        return 1


def cancelled(*_):
    raise KeyboardInterrupt
