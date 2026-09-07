from __future__ import annotations

import argparse
import multiprocessing
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

from mcpmft.args import (
    LaunchArguments,
    ProjectConfig,
    load_project_document,
)


def _launch_config(config: dict) -> LaunchArguments:
    return ProjectConfig.from_dict(config).launch


def _hosts(config: LaunchArguments) -> list[str]:
    if config.hosts and config.hostfile:
        raise ValueError("Set launch.hosts or launch.hostfile, not both")
    hosts = list(config.hosts)
    if config.hostfile:
        hostfile = Path(config.hostfile).expanduser()
        with hostfile.open("r", encoding="utf-8") as handle:
            hosts = [
                line.split()[0]
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
    if not hosts:
        hosts = ["localhost"]
    if len(hosts) != len(set(hosts)):
        raise ValueError("launch hosts must be unique")
    return hosts


def _host_address(host: str) -> str:
    return host.rsplit("@", 1)[-1]


def _endpoint(host: str, user: str) -> str:
    return host if "@" in host or not user else f"{user}@{host}"


def _is_local(host: str) -> bool:
    address = _host_address(host)
    return address in {
        "localhost",
        "127.0.0.1",
        "::1",
        socket.gethostname(),
        socket.getfqdn(),
    }


def _project_dir(config: LaunchArguments) -> Path:
    if config.project_dir:
        return Path(config.project_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _training_argv(config_paths: list[Path], overrides: list[str]) -> list[str]:
    argv: list[str] = []
    for path in config_paths:
        argv.extend(("--config", str(path)))
    argv.extend(overrides)
    return argv


def _remote_command(
    config: LaunchArguments,
    *,
    node_rank: int,
    config_paths: list[Path],
    overrides: list[str],
) -> str:
    project_dir = _project_dir(config)
    argv = [
        config.python,
        "-m",
        "mcpmft.launch",
        "--node-rank",
        str(node_rank),
        *_training_argv(config_paths, overrides),
    ]
    return f"cd {shlex.quote(str(project_dir / 'minicpm_ft'))} && {shlex.join(argv)}"


def _run_cluster(
    config: LaunchArguments,
    *,
    config_paths: list[Path],
    overrides: list[str],
) -> int:
    hosts = _hosts(config)
    master_addr = config.master_addr or _host_address(hosts[0])
    log_dir = Path(config.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = _project_dir(config) / "minicpm_ft" / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[str, subprocess.Popen[bytes], Any]] = []
    for node_rank, host in enumerate(hosts):
        # The resolved rendezvous address must reach every node even when it was
        # inferred from the first host rather than written in YAML.
        node_overrides = [
            *overrides,
            "--launch.master_addr",
            master_addr,
        ]
        remote = _remote_command(
            config,
            node_rank=node_rank,
            config_paths=config_paths,
            overrides=node_overrides,
        )
        if len(hosts) == 1 and _is_local(host):
            command = [
                config.python,
                "-m",
                "mcpmft.launch",
                "--node-rank",
                str(node_rank),
                *_training_argv(config_paths, node_overrides),
            ]
            cwd = _project_dir(config) / "minicpm_ft"
        else:
            command = [
                "ssh",
                *config.ssh_options,
                _endpoint(host, config.ssh_user),
                remote,
            ]
            cwd = None
        log_handle = (log_dir / f"node_{node_rank}.log").open("wb")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((host, process, log_handle))
        print(f"[launch] node_rank={node_rank} host={host} pid={process.pid}")

    try:
        while processes:
            for host, process, _ in processes:
                code = process.poll()
                if code is not None and code != 0:
                    print(f"[launch] {host} exited with code {code}", file=sys.stderr)
                    return code
            if all(process.poll() == 0 for _, process, _ in processes):
                return 0
            time.sleep(1)
    finally:
        for _, process, _ in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _, process, handle in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            handle.close()
    return 0


def _activate_local_source(config: LaunchArguments) -> Path:
    project_dir = _project_dir(config)
    source = project_dir / "minicpm_ft" / "mcpmft"
    if not source.is_dir():
        raise FileNotFoundError(f"mcpmft source directory not found: {source}")

    cache_root = Path(config.local_cache_dir).expanduser().resolve()
    local_root = cache_root / "source"
    local_package = local_root / "mcpmft"
    if local_package.exists():
        shutil.rmtree(local_package)
    local_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        local_package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_MODULES_CACHE"] = str(cache_root / "huggingface" / "modules")
    os.environ["PYTHONPYCACHEPREFIX"] = str(cache_root / "pycache")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache_root / "torch_extensions")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    sys.path.insert(0, str(local_root))

    package = sys.modules["mcpmft"]
    package.__path__[:] = [str(local_package)]
    return local_root


def _rank_main(
    local_rank: int,
    node_rank: int,
    world_size: int,
    gpus_per_node: int,
    master_addr: str,
    master_port: int,
    train_argv: list[str],
    log_dir: str,
) -> None:
    global_rank = node_rank * gpus_per_node + local_rank
    os.environ.update(
        {
            "RANK": str(global_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(gpus_per_node),
            "GROUP_RANK": str(node_rank),
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
        }
    )
    if local_rank:
        path = Path(log_dir) / f"rank_{global_rank}.log"
        handle = path.open("a", encoding="utf-8", buffering=1)
        os.dup2(handle.fileno(), sys.stdout.fileno())
        os.dup2(handle.fileno(), sys.stderr.fileno())
    try:
        from mcpmft.train.main import main as train_main

        train_main(train_argv)
    except BaseException:
        traceback.print_exc()
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        os._exit(1)


def _run_node(
    config: LaunchArguments,
    *,
    node_rank: int,
    config_paths: list[Path],
    overrides: list[str],
) -> int:
    hosts = _hosts(config)
    if not 0 <= node_rank < len(hosts):
        raise ValueError(f"node rank {node_rank} is outside a {len(hosts)}-node launch")
    if config.gpus_per_node < 1:
        raise ValueError("launch.gpus_per_node must be positive")
    master_addr = config.master_addr or _host_address(hosts[0])
    _activate_local_source(config)

    if config.nccl_socket_ifname:
        os.environ["NCCL_SOCKET_IFNAME"] = config.nccl_socket_ifname
    os.environ["NCCL_IB_DISABLE"] = "1" if config.nccl_ib_disable else "0"
    os.environ["OMP_NUM_THREADS"] = str(config.omp_num_threads)

    # Import the CUDA-neutral training stack once on this node. Importing
    # ``mcpmft.train.trainer`` also imports DeepSpeed, which initializes CUDA and
    # cannot happen before local workers fork.
    import mcpmft.train.main  # noqa: F401

    project_dir = _project_dir(config)
    os.chdir(project_dir / "minicpm_ft")
    log_dir = Path(config.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = project_dir / "minicpm_ft" / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    world_size = len(hosts) * config.gpus_per_node
    train_argv = _training_argv(config_paths, overrides)
    context = multiprocessing.get_context("fork")
    workers = [
        context.Process(
            target=_rank_main,
            args=(
                local_rank,
                node_rank,
                world_size,
                config.gpus_per_node,
                master_addr,
                config.master_port,
                train_argv,
                str(log_dir),
            ),
        )
        for local_rank in range(config.gpus_per_node)
    ]
    for worker in workers:
        worker.start()
    try:
        while True:
            failed = next(
                (worker for worker in workers if worker.exitcode not in {None, 0}),
                None,
            )
            if failed is not None:
                return int(failed.exitcode or 1)
            if all(worker.exitcode == 0 for worker in workers):
                return 0
            time.sleep(1)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(15)
            if worker.is_alive():
                worker.kill()
                worker.join()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Launch MiniCPM training from one YAML configuration"
    )
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="print the resolved training mode and exit",
    )
    parser.add_argument("--node-rank", type=int, help=argparse.SUPPRESS)
    args, overrides = parser.parse_known_args(argv)
    config_paths = [Path(path).expanduser().resolve() for path in args.config]
    document = load_project_document(config_paths, overrides)
    project = ProjectConfig.from_dict(document)
    if args.show_config:
        print(yaml.safe_dump(project.to_dict(), sort_keys=False, allow_unicode=True))
        return
    config = project.launch
    result = (
        _run_cluster(config, config_paths=config_paths, overrides=overrides)
        if args.node_rank is None
        else _run_node(
            config,
            node_rank=args.node_rank,
            config_paths=config_paths,
            overrides=overrides,
        )
    )
    raise SystemExit(result)


if __name__ == "__main__":
    main()
