"""One CUDA owner for many CPU solvers; no changes to solver RNG or objectives.

Only the ridge proxy is moved. Its formula is unchanged, but batched CUDA
arithmetic is NOT bitwise equivalent to the historical CPU implementation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
import traceback


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def cuda_predict(torch, requests, cache, cache_lock=None):
    """Batch a single request's trajectories and members into GPU matrix solves.

    Never merge separate requests: changing GEMM batch dimensions with queue
    timing can alter float32 outputs and break paired offline prefixes.
    """
    import numpy as np
    assert len(requests) == 1, 'Cross-request batching violates scheduling invariance'
    first = requests[0]
    key = first['features_id']
    import contextlib
    with cache_lock if cache_lock is not None else contextlib.nullcontext():
        if key not in cache:
            cache[key] = (torch.as_tensor(first['freq'], device='cuda'),
                          torch.as_tensor(first['phase'], device='cuda'))
            # Publish fixed features only after their creating CUDA stream is ready.
            torch.cuda.current_stream().synchronize()
    freq, phase = cache[key]
    ax, ay, cx, lower, upper = [torch.as_tensor(np.concatenate([r[k] for r in requests]), device='cuda')
                                for k in ('x', 'y', 'candidates', 'lower', 'upper')]
    width = (upper - lower).clamp_min(1e-8)
    ax = 2 * (ax - lower) / width - 1
    cx = 2 * (cx - lower) / width - 1

    def features(x):
        # [batch, members, points, features], preserving the original feature order.
        proj = x[:, None] @ freq[None].transpose(-1, -2) + phase[None, :, None]
        rff = (2.0 / freq.shape[1]) ** 0.5 * torch.cos(proj)
        plain = torch.cat((torch.ones_like(x[..., :1]), x, x * x), -1)
        return torch.cat((plain[:, None].expand(-1, freq.shape[0], -1, -1), rff), -1)

    phi, phi_c = features(ax), features(cx)
    mean = ay.mean(-1, keepdim=True)
    std = ay.std(-1, keepdim=True, unbiased=False).clamp_min(1e-8)
    yz = ((ay - mean) / std)[:, None, :, None]
    lhs = phi.transpose(-1, -2) @ phi
    lhs = lhs + first['ridge'] * torch.eye(phi.shape[-1], device='cuda', dtype=phi.dtype)
    rhs = phi.transpose(-1, -2) @ yz
    beta, info = torch.linalg.solve_ex(lhs, rhs)
    failed = info.cpu().numpy() != 0
    pinv_count = int(failed.sum())
    if pinv_count:
        failures = torch.as_tensor(failed, device='cuda')
        # Match the original per-system pseudoinverse fallback, not the whole batch.
        beta[failures] = torch.linalg.pinv(lhs[failures]) @ rhs[failures]
    predictions = (phi_c @ beta).squeeze(-1) * std[:, None] + mean[:, None]
    mu = predictions.mean(1)
    sigma = predictions.std(1, unbiased=False)
    if first['distance_scale'] > 0:
        sigma = sigma + first['distance_scale'] * std * torch.cdist(cx, ax).amin(-1)
    sigma = sigma.clamp_min(1e-8)
    # A single device-to-host transfer avoids repeated CUDA synchronizations.
    output = torch.stack((mu, sigma), 0).cpu().numpy()
    if not np.isfinite(output).all():
        raise RuntimeError('Nonfinite CUDA proxy prediction; refusing silent CPU fallback')
    return output[0], output[1], pinv_count


def serve(address, authkey, visible_devices, ready, stats_path, concurrency=2):
    """Spawned before importing any legacy CPU-only experiment modules."""
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from multiprocessing.connection import Listener
    listener = None
    pending = queue.Queue()
    stats = dict(requests=0, gpu_batches=0, trajectories=0, pinv_systems=0,
                 failures=0, gpu_compute_seconds=0.0, max_merged_requests=0)
    try:
        import torch
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in selected Python environment')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        # Force driver/kernel initialization before claiming ready.
        probe = torch.eye(4, device='cuda')
        torch.linalg.solve(probe, probe)
        torch.cuda.synchronize()
        metadata = dict(torch=torch.__version__, cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(0),
                        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                        tf32=False, dtype='float32', backend='batched_cuda_ridge_v2',
                        cuda_stream_workers=concurrency)
        listener = Listener(address, family='AF_UNIX', authkey=authkey)
        ready.send(dict(status='ready', **metadata))
        ready.close()

        def connection_loop(connection):
            try:
                while True:
                    request = connection.recv()
                    if request.get('op') == 'shutdown':
                        pending.put(None)
                        return
                    result = queue.Queue(maxsize=1)
                    pending.put((request, result))
                    connection.send(result.get())
            except (EOFError, BrokenPipeError, OSError):
                return
            finally:
                connection.close()

        def accept_loop():
            while True:
                try:
                    connection = listener.accept()
                except (OSError, EOFError):
                    return
                threading.Thread(target=connection_loop, args=(connection,), daemon=True).start()

        threading.Thread(target=accept_loop, daemon=True).start()
        from concurrent.futures import ThreadPoolExecutor
        cache, cache_lock, stats_lock = {}, threading.Lock(), threading.Lock()
        stream_local = threading.local()

        def execute(item):
            request, result = item
            if not hasattr(stream_local, 'stream'):
                stream_local.stream = torch.cuda.Stream()
            start = time.monotonic()
            try:
                with torch.cuda.stream(stream_local.stream), torch.inference_mode():
                    mu, sigma, pinv = cuda_predict(torch, [request], cache, cache_lock)
                result.put(dict(mu=mu, sigma=sigma))
                with stats_lock:
                    stats['requests'] += 1
                    stats['gpu_batches'] += 1
                    stats['trajectories'] += len(request['x'])
                    stats['pinv_systems'] += pinv
                    stats['max_merged_requests'] = 1
            except Exception:
                with stats_lock:
                    stats['failures'] += 1
                result.put(dict(error=traceback.format_exc()))
            with stats_lock:
                stats['gpu_compute_seconds'] += time.monotonic() - start

        executor = ThreadPoolExecutor(max_workers=concurrency)
        last_stats = 0.0
        while True:
            try:
                item = pending.get(timeout=1)
            except queue.Empty:
                write_json(stats_path, dict(**stats, **metadata, status='running'))
                continue
            if item is None:
                break
            executor.submit(execute, item)
            if time.monotonic() - last_stats >= 1:
                write_json(stats_path, dict(**stats, **metadata, status='running'))
                last_stats = time.monotonic()
        executor.shutdown(wait=True)
        write_json(stats_path, dict(**stats, **metadata, status='stopped'))
    except Exception:
        failure = dict(status='failed', error=traceback.format_exc(), **stats)
        write_json(stats_path, failure)
        try:
            ready.send(failure)
        except (OSError, EOFError):
            pass
        raise
    finally:
        if listener is not None:
            listener.close()


class ProxyClient:
    def __init__(self, address, authkey):
        from multiprocessing.connection import Client
        self.connection = Client(address, family='AF_UNIX', authkey=authkey)

    def predict(self, proxy, archive_x, archive_y, candidates, problem, distance_scale=0.15):
        import torch
        from roopf.model import _get_bounds, _ensure_fitness_2d
        lower, upper = _get_bounds(problem, archive_x)
        lower = lower.view(1, 1, -1).expand(archive_x.shape[0], -1, -1)
        upper = upper.view(1, 1, -1).expand(archive_x.shape[0], -1, -1)
        freq, phase = proxy.freq.detach().cpu().numpy(), proxy.phase.detach().cpu().numpy()
        key = hashlib.sha256(freq.tobytes() + phase.tobytes()).hexdigest()
        request = dict(features_id=key, freq=freq, phase=phase, ridge=proxy.ridge,
                       distance_scale=distance_scale)
        for name, value in dict(x=archive_x, y=_ensure_fitness_2d(archive_y), candidates=candidates,
                                lower=lower, upper=upper).items():
            request[name] = value.detach().cpu().contiguous().numpy()
        self.connection.send(request)
        if not self.connection.poll(120):
            raise TimeoutError('CUDA proxy service did not respond within 120 seconds')
        response = self.connection.recv()
        if 'error' in response:
            raise RuntimeError(response['error'])
        return tuple(torch.from_numpy(response[key]).to(archive_x.device) for key in ('mu', 'sigma'))

    def close(self):
        self.connection.close()
