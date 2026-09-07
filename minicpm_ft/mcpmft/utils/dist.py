from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    node_rank: int

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


def get_dist_info() -> DistInfo:
    return DistInfo(
        rank=int(os.environ.get("RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        local_rank=int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))),
        node_rank=int(os.environ.get("NODE_RANK", os.environ.get("SLURM_NODEID", "0"))),
    )


def shard_index(index: int, world_size: int | None = None, rank: int | None = None) -> bool:
    info = get_dist_info()
    world_size = info.world_size if world_size is None else world_size
    rank = info.rank if rank is None else rank
    return index % max(world_size, 1) == rank

