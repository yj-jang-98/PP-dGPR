"""Vectorized additive masking and secure average consensus (Protocol 1)."""

import torch
import numpy as np
import time


def prepare_graph_cache(G):
    """Cache graph neighborhoods and intersections reused at every iteration."""
    cache = G.graph.get("_pp_consensus_cache")
    graph_signature = (
        tuple(G.nodes()),
        tuple(sorted(tuple(sorted(edge)) for edge in G.edges())),
    )
    if cache is not None and cache["signature"] == graph_signature:
        return cache

    nodes = list(G.nodes())
    neighbors = {i: list(G.neighbors(i)) for i in nodes}
    neighbors_plus = {i: set(neighbors[i]) | {i} for i in nodes}
    common_neighbors = {
        (i, j): tuple(sorted(neighbors_plus[i] & neighbors_plus[j]))
        for i in nodes
        for j in neighbors_plus[i]
    }
    node_to_index = {node: index for index, node in enumerate(nodes)}
    edge_sources, edge_destinations = [], []
    mask_random_groups, mask_random_targets = [], []
    mask_last_targets = []
    group = 0
    for i in nodes:
        i_index = node_to_index[i]
        for j in neighbors[i]:
            edge_sources.append(i_index)
            edge_destinations.append(node_to_index[j])
        for j in sorted(neighbors_plus[i]):
            common = common_neighbors[(i, j)]
            targets = [i_index * len(nodes) + node_to_index[l] for l in common]
            mask_random_groups.extend([group] * (len(targets) - 1))
            mask_random_targets.extend(targets[:-1])
            mask_last_targets.append(targets[-1])
            group += 1
    cache = {
        "signature": graph_signature,
        "nodes": nodes,
        "neighbors": neighbors,
        "neighbors_plus": neighbors_plus,
        "common_neighbors": common_neighbors,
        # Device-independent index lists for the vectorized Protocol-1 kernel.
        "edge_sources": edge_sources,
        "edge_destinations": edge_destinations,
        "mask_random_groups": mask_random_groups,
        "mask_random_targets": mask_random_targets,
        "mask_last_targets": mask_last_targets,
        "device_tensors": {},
    }
    G.graph["_pp_consensus_cache"] = cache
    return cache


def _cached_graph_tensors(graph_cache, Wbar, device):
    """Materialize graph indices/weights once per device and weight matrix."""
    weights_np = np.asarray(Wbar)
    weight_signature = (weights_np.shape, weights_np.tobytes())
    key = (str(torch.device(device)), weight_signature)
    cached = graph_cache["device_tensors"].get(key)
    if cached is not None:
        return cached

    sources = torch.as_tensor(
        graph_cache["edge_sources"], dtype=torch.long, device=device
    )
    destinations = torch.as_tensor(
        graph_cache["edge_destinations"], dtype=torch.long, device=device
    )
    weights = torch.as_tensor(weights_np, dtype=torch.int64, device=device)[
        sources, destinations
    ]
    cached = {
        "edge_sources": sources,
        "edge_destinations": destinations,
        "edge_weights": weights,
        "mask_random_groups": torch.as_tensor(
            graph_cache["mask_random_groups"], dtype=torch.long, device=device
        ),
        "mask_random_targets": torch.as_tensor(
            graph_cache["mask_random_targets"], dtype=torch.long, device=device
        ),
        "mask_last_targets": torch.as_tensor(
            graph_cache["mask_last_targets"], dtype=torch.long, device=device
        ),
    }
    graph_cache["device_tensors"][key] = cached
    return cached


def vectorized_masks(graph_cache, graph_tensors, q, p, device):
    """Generate every Protocol-1 mask with two indexed reductions."""
    random_groups = graph_tensors["mask_random_groups"]
    random_targets = graph_tensors["mask_random_targets"]
    last_targets = graph_tensors["mask_last_targets"]
    group_count = last_targets.numel()

    mask_flat = torch.zeros(
        (len(graph_cache["nodes"]) ** 2, p), dtype=torch.int64, device=device
    )
    # A complete graph can have thousands of share rows.  Keep those rows
    # vectorized while processing state columns in bounded-memory chunks.
    max_random_elements = 8_000_000
    column_chunk = max(1, min(p, max_random_elements // random_groups.numel()))
    for begin in range(0, p, column_chunk):
        end = min(begin + column_chunk, p)
        width = end - begin
        partials = torch.randint(
            low=(-q + 1) // 2,
            high=q // 2,
            size=(random_groups.numel(), width),
            dtype=torch.int64,
            device=device,
        )
        group_sums = torch.zeros((group_count, width), dtype=torch.int64, device=device)
        group_sums.index_add_(0, random_groups, partials)
        target = mask_flat[:, begin:end]
        target.index_add_(0, random_targets, partials)
        target.index_add_(0, last_targets, torch.remainder(-group_sums, q))
    return torch.remainder(mask_flat, q).reshape(
        len(graph_cache["nodes"]), len(graph_cache["nodes"]), p
    )


def privacy_preserving_avg_consensus(
    zini, G, config, T, device, comm_delay: float = 0.0, return_history: bool = True
):
    """Run Protocol 1 on states shaped (M, p).

    ``config`` supplies the modulus q, scales Lz and Lw, and integer weights
    Wbar = W / Lw. Nodes are indexed from 0 to M - 1. Each iteration includes
    two modeled communication delays when ``comm_delay`` is positive.

    Return T + 1 torch tensors, including the initial state, or only the final
    tensor when ``return_history=False``."""
    q = config["q"]
    Lz = config["Lz"]
    Lw = config["Lw"]
    LwLz = Lw * Lz
    graph_cache = prepare_graph_cache(G)
    graph_tensors = _cached_graph_tensors(graph_cache, config["Wbar"], device)
    edge_sources = graph_tensors["edge_sources"]
    edge_destinations = graph_tensors["edge_destinations"]
    edge_weights = graph_tensors["edge_weights"]

    if isinstance(zini, np.ndarray):
        z = torch.tensor(zini, dtype=torch.float32, device=device)
    else:
        z = zini.to(device=device, dtype=torch.float32)

    M, p = z.shape
    z_history_pp = [z.clone()] if return_history else None

    for _ in range(T):
        # Model the two communication phases of Protocol 1.
        if comm_delay > 0.0:
            time.sleep(2 * comm_delay)

        z_q = torch.round(z / Lz).to(torch.int64)
        masks = vectorized_masks(graph_cache, graph_tensors, q, p, device)

        edge_terms = (
            edge_weights[:, None] * (z_q[edge_destinations] - z_q[edge_sources])
            + masks[edge_sources, edge_destinations]
        )
        summed = torch.zeros((M, p), dtype=torch.int64, device=device)
        summed.index_add_(0, edge_sources, edge_terms)
        diagonal = masks[torch.arange(M, device=device), torch.arange(M, device=device)]
        total = torch.remainder(diagonal + summed, q)
        total = torch.where(total >= q // 2, total - q, total)
        z = z + (LwLz * total).to(z.dtype)
        if return_history:
            z_history_pp.append(z.clone())

    return z_history_pp if return_history else z
