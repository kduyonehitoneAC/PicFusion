# jigsaw_edge_graph_solver_improved.py
# Requirements: numpy, PIL, matplotlib, torch
# Run: python jigsaw_edge_graph_solver_improved.py

import os
import glob
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from train import EdgeCNN, EMBED_DIM, EDGE_STRIP   # your edge CNN
import heapq
import math
import random
import torch
import itertools
from collections import defaultdict

# ========== CONFIG ==========
ROWS = 3
COLS = 5
NUM_PIECES = ROWS * COLS
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 600
PATCH_HEIGHT = IMAGE_HEIGHT // ROWS
PATCH_WIDTH = IMAGE_WIDTH // COLS
IMAGE_DIR = './data_train'

# SEARCH / PERFORMANCE knobs
MAX_CANDIDATES = 697
CENTER_BEAM = 200           # beam width for center 3x3 search (lower -> faster)
TOP_SPLITS = 50             # top splits to try for left/right
BRANCH_CUT_THRESHOLD = 0.7256
LOCAL_MAX_ITERS = 2000      # local refinement iterations (swap attempts)
EDGE_STRIP = EDGE_STRIP     # from train import
# ==============================

# ---------- helper: edge descriptor wrapper -----------
def edge_descriptor_for_side(patch, side='left'):
    H, W, _ = patch.shape
    if side == 'left':
        strip = patch[:, :EDGE_STRIP, :]
    elif side == 'right':
        strip = patch[:, -EDGE_STRIP:, :]
    elif side == 'top':
        strip = patch[:EDGE_STRIP, :, :]
    elif side == 'bottom':
        strip = patch[-EDGE_STRIP:, :, :]
    else:
        raise ValueError(side)
    t = torch.tensor(strip.transpose(2,0,1).copy(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        desc = model(t).cpu().numpy().flatten()
    desc /= np.linalg.norm(desc) + 1e-8
    return desc

# ---------- compute descriptors for all pieces ----------
def compute_all_edge_features(pieces):
    feats = []
    for p in pieces:
        fs = {}
        for s in ('left','right','top','bottom'):
            fs[s] = edge_descriptor_for_side(p, side=s)
        feats.append(fs)
    return feats

# ---------- compatibility helpers (fast lookup arrays) ----------
# build comp as dict + numpy array for speed
def build_pairwise_compatibility(feats):
    N = len(feats)
    # order channels: 0: R-L, 1: L-R, 2: B-T, 3: T-B
    comp_arr = np.zeros((N, N, 4), dtype=np.float32)
    comp_dict = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r_l = compatibility_score(feats[i]['right'], feats[j]['left'])
            l_r = compatibility_score(feats[i]['left'], feats[j]['right'])
            b_t = compatibility_score(feats[i]['bottom'], feats[j]['top'])
            t_b = compatibility_score(feats[i]['top'], feats[j]['bottom'])
            comp_arr[i, j, 0] = r_l
            comp_arr[i, j, 1] = l_r
            comp_arr[i, j, 2] = b_t
            comp_arr[i, j, 3] = t_b
            comp_dict[(i, j, 'R-L')] = r_l
            comp_dict[(i, j, 'L-R')] = l_r
            comp_dict[(i, j, 'B-T')] = b_t
            comp_dict[(i, j, 'T-B')] = t_b
    return comp_arr, comp_dict

def compatibility_score(a, b):
    if a is None or b is None: return 0.0
    denom = (np.linalg.norm(a)*np.linalg.norm(b))
    if denom == 0: return 0.0
    sim = np.dot(a, b)/denom
    return float(max(0.0, min(1.0, sim)))

# ---------- neighbors positions ----------
def neighbors_of_position(pos_idx, rows=3, cols=3):
    r, c = divmod(pos_idx, cols)
    neigh = []
    if c - 1 >= 0: neigh.append(((r * cols + (c-1)), 'L'))
    if c + 1 < cols: neigh.append(((r * cols + (c+1)), 'R'))
    if r - 1 >= 0: neigh.append((((r-1) * cols + c), 'T'))
    if r + 1 < rows: neigh.append((((r+1) * cols + c), 'B'))
    return neigh

# ---------- cost for a (partial/full) assignment ----------
# full_map: list length rows*cols mapping pos->global_piece_index
def compute_total_cost_from_map(full_map, comp_dict, rows=ROWS, cols=COLS):
    # sum over unique adjacent pairs (pos_a,pos_b) where pos_b is right or bottom neighbor
    total = 0.0
    Npos = rows*cols
    for pos in range(Npos):
        r, c = divmod(pos, cols)
        gp = full_map[pos]
        if gp == -1: continue
        # right neighbor
        if c+1 < cols:
            nbpos = pos+1
            gnb = full_map[nbpos]
            if gnb != -1:
                total += (1.0 - comp_dict.get((gp, gnb, 'R-L'), 0.0))
        # bottom neighbor
        if r+1 < rows:
            nbpos = pos + cols
            gnb = full_map[nbpos]
            if gnb != -1:
                total += (1.0 - comp_dict.get((gp, gnb, 'B-T'), 0.0))
    return total

# ---------- beam search for 3x3 center ----------
def solve_center_by_beam(candidate_piece_indices, comp_arr, comp_dict, rows=3, cols=3, beam_width=CENTER_BEAM):
    P = rows*cols
    global_pieces = candidate_piece_indices
    # pos neighbors for local grid
    pos_neighbors = {pos: neighbors_of_position(pos, rows, cols) for pos in range(P)}
    # Beam: list of tuples (cost_so_far, assign_dict(pos->piece), used_set)
    beam = [(0.0, {}, frozenset())]
    # iterate until all P positions are filled
    for step in range(P):
        new_candidates = []
        for cost, assign, used in beam:
            # choose next pos to fill: the unfilled position with most assigned neighbors (MRV heuristic)
            free_positions = [p for p in range(P) if p not in assign]
            pos_to_fill = max(free_positions, key=lambda pos: sum(1 for (npos,_) in pos_neighbors[pos] if npos in assign))
            # try all unused pieces
            for gidx in global_pieces:
                if gidx in used: continue
                inc = 0.0
                nbcount = 0
                for (npos, rel) in pos_neighbors[pos_to_fill]:
                    if npos in assign:
                        nbcount += 1
                        neighbor_piece = assign[npos]
                        if rel == 'L':
                            sc = comp_dict.get((neighbor_piece, gidx, 'R-L'), 0.0)
                        elif rel == 'R':
                            sc = comp_dict.get((neighbor_piece, gidx, 'L-R'), 0.0)
                        elif rel == 'T':
                            sc = comp_dict.get((neighbor_piece, gidx, 'B-T'), 0.0)
                        elif rel == 'B':
                            sc = comp_dict.get((neighbor_piece, gidx, 'T-B'), 0.0)
                        else:
                            sc = 0.0
                        inc += (1.0 - sc)
                new_cost = cost + inc
                new_assign = assign.copy()
                new_assign[pos_to_fill] = gidx
                new_used = frozenset(set(used) | {gidx})
                new_candidates.append((new_cost, new_assign, new_used))
        # keep top beam_width candidates by cost
        new_candidates.sort(key=lambda x: x[0])
        beam = [(c,a,u) for (c,a,u) in new_candidates[:beam_width]]
        # early exit if beam empty
        if not beam:
            break
    # pick best full one
    best = None
    for cost, assign, used in beam:
        if len(assign) == P:
            if best is None or cost < best[0]:
                best = (cost, assign)
    if best is None:
        return None, float('inf')
    # convert to ordered list of length P (pos 0..P-1)
    assign_list = [best[1][p] for p in range(P)]
    return assign_list, best[0]

# ---------- local refinement: swap any two positions if reduces cost ----------
def local_refine_full_map(full_map, comp_dict, max_iters=LOCAL_MAX_ITERS, rows=ROWS, cols=COLS):
    Npos = rows*cols
    best_map = full_map.copy()
    best_cost = compute_total_cost_from_map(best_map, comp_dict, rows, cols)
    improved = True
    it = 0
    # also try swapping entire left/right columns (3x1 columns) as a special move
    while improved and it < max_iters:
        improved = False
        it += 1
        # try pairwise single-tile swaps (randomized order)
        indices = list(range(Npos))
        random.shuffle(indices)
        for i in indices:
            for j in indices:
                if i >= j: continue
                # swap
                cand_map = best_map.copy()
                cand_map[i], cand_map[j] = cand_map[j], cand_map[i]
                cand_cost = compute_total_cost_from_map(cand_map, comp_dict, rows, cols)
                if cand_cost + 1e-9 < best_cost:
                    best_map = cand_map
                    best_cost = cand_cost
                    improved = True
                    # break to restart scanning from top (greedy)
                    break
            if improved:
                break
        if improved:
            continue
        # try swapping whole columns (col blocks)
        for c1 in range(cols):
            for c2 in range(c1+1, cols):
                cand_map = best_map.copy()
                # swap column c1 and c2
                for r in range(rows):
                    p1 = r*cols + c1
                    p2 = r*cols + c2
                    cand_map[p1], cand_map[p2] = cand_map[p2], cand_map[p1]
                cand_cost = compute_total_cost_from_map(cand_map, comp_dict, rows, cols)
                if cand_cost + 1e-9 < best_cost:
                    best_map = cand_map
                    best_cost = cand_cost
                    improved = True
                    break
            if improved:
                break
    return best_map, best_cost

# ---------- utility: greedy select center candidates simplified ----------
def greedy_select_center_candidates(all_indices, comp_dict, center_size=9, max_candidates=150):
    # simpler greedy: start from pieces with highest total degree (sum compat with others)
    deg = {}
    for i in all_indices:
        d = 0.0
        for j in all_indices:
            if i == j: continue
            d += comp_dict.get((i,j,'R-L'),0) + comp_dict.get((i,j,'L-R'),0) + comp_dict.get((i,j,'T-B'),0) + comp_dict.get((i,j,'B-T'),0)
        deg[i] = d
    # pick seeds by top degree
    sorted_by_deg = sorted(all_indices, key=lambda x: -deg[x])
    seeds = sorted_by_deg[:min(40, len(sorted_by_deg))]
    candidates = []
    for seed in seeds:
        s = {seed}
        while len(s) < center_size:
            # choose piece maximizing sum compat to current s
            best_cand, best_gain = None, -1.0
            for cand in all_indices:
                if cand in s: continue
                gain = 0.0
                for u in s:
                    scs = [comp_dict.get((u, cand, k),0) for k in ('R-L','L-R','B-T','T-B')]
                    gain += max(scs)
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand
            if best_cand is None:
                break
            s.add(best_cand)
        if len(s) == center_size:
            key = tuple(sorted(s))
            candidates.append(list(key))
        if len(candidates) >= max_candidates:
            break
    # if not enough, sample random sets
    while len(candidates) < max_candidates:
        sample = sorted(random.sample(all_indices, center_size))
        if sample not in candidates:
            candidates.append(sample)
    return candidates

# ---------- solve full 3x5 with center-beam + splits + local refinement ----------
def solve_full_puzzle(image_path, rows=ROWS, cols=COLS, verbose=True):
    t0 = time.time()
    img = Image.open(image_path).convert('RGB').resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = np.array(img)/255.0
    pieces = [img_array[r*PATCH_HEIGHT:(r+1)*PATCH_HEIGHT, c*PATCH_WIDTH:(c+1)*PATCH_WIDTH, :]
              for r in range(rows) for c in range(cols)]
    N = len(pieces)
    all_indices = list(range(N))
    if verbose:
        print(f"Loaded image {image_path}, {N} pieces")

    # compute features + pairwise comp
    feats = compute_all_edge_features(pieces)
    comp_arr, comp_dict = build_pairwise_compatibility(feats)
    if verbose:
        print("Edge features & pairwise compatibility computed.")

    # choose center candidates
    center_size = 9
    candidates = greedy_select_center_candidates(all_indices, comp_dict, center_size=center_size, max_candidates=MAX_CANDIDATES)
    if verbose:
        print(f"Generated {len(candidates)} center candidate sets (heuristic). Center beam={CENTER_BEAM}")

    best_global = None
    best_global_cost = float('inf')
    best_solution_details = None

    # precompute remaining positions
    center_positions = []
    for r in range(rows):
        for c in (1,2,3):
            center_positions.append(r*cols + c)

    # iterate candidates with progress
    for idx_set in candidates:
        # Beam-solve center (3x3) using candidate set indices
        assign_center, cost_center = solve_center_by_beam(idx_set, comp_arr, comp_dict, rows=3, cols=3, beam_width=CENTER_BEAM)
        if assign_center is None:
            continue
        # map local 3x3 to full grid positions
        full_assign = {}
        for local_pos, gidx in enumerate(assign_center):
            rloc, cloc = divmod(local_pos, 3)
            full_col = 1 + cloc
            full_pos = rloc * cols + full_col
            full_assign[full_pos] = gidx

        remaining = [i for i in all_indices if i not in idx_set]
        if len(remaining) != (rows*cols - center_size):
            continue

        # Rank splits (left 3 vs right 3) using heuristics
        combs = list(combinations(remaining, 3))
        scored = []
        for left_choice in combs:
            right_choice = tuple(sorted(set(remaining) - set(left_choice)))
            hscore = 0.0
            # compatibility left <-> center
            for r in range(rows):
                left_piece = left_choice[r] if r < len(left_choice) else None
                center_piece = full_assign.get(r*cols + 1, None)
                if left_piece is not None and center_piece is not None:
                    hscore += comp_dict.get((left_piece, center_piece, 'R-L'), 0.0)
            # internal left bonus
            internal = 0.0
            for a,b in permutations(left_choice,2):
                internal += max(comp_dict.get((a,b,'R-L'),0.0), comp_dict.get((a,b,'B-T'),0.0))
            scored.append((hscore + internal, left_choice, right_choice))
        scored.sort(key=lambda x: -x[0])
        top_splits = scored[:TOP_SPLITS]

        # try each split and permutations
        for score_split, left_choice, right_choice in top_splits:
            best_local_cost = float('inf')
            best_local_assign = None
            for left_perm in permutations(left_choice):
                local_cost_left = 0.0
                for r in range(rows):
                    gp = left_perm[r]
                    cpos = r*cols + 1
                    if cpos in full_assign:
                        center_piece = full_assign[cpos]
                        local_cost_left += (1.0 - comp_dict.get((gp, center_piece, 'R-L'), 0.0))
                    if r > 0:
                        above_piece = left_perm[r-1]
                        local_cost_left += (1.0 - comp_dict.get((above_piece, gp, 'B-T'), 0.0))
                for right_perm in permutations(right_choice):
                    local_cost_right = 0.0
                    for r in range(rows):
                        gp = right_perm[r]
                        cpos = r*cols + 3
                        if cpos in full_assign:
                            center_piece = full_assign[cpos]
                            local_cost_right += (1.0 - comp_dict.get((center_piece, gp, 'R-L'), 0.0))
                        if r > 0:
                            above_piece = right_perm[r-1]
                            local_cost_right += (1.0 - comp_dict.get((above_piece, gp, 'B-T'), 0.0))
                    total_cost = cost_center + local_cost_left + local_cost_right
                    if total_cost < best_local_cost:
                        best_local_cost = total_cost
                        best_local_assign = (left_perm, right_perm, total_cost)
            if best_local_assign and best_local_assign[2] < best_global_cost:
                left_perm, right_perm, tot = best_local_assign
                # build full mapping
                full_map = [-1] * (rows*cols)
                # center
                for local_pos, gidx in enumerate(assign_center):
                    rloc, cloc = divmod(local_pos, 3)
                    full_col = 1 + cloc
                    pos = rloc * cols + full_col
                    full_map[pos] = gidx
                # left
                for r in range(rows):
                    pos = r*cols + 0
                    full_map[pos] = left_perm[r]
                # right
                for r in range(rows):
                    pos = r*cols + cols-1
                    full_map[pos] = right_perm[r]

                # Local refinement (swaps and column swaps)
                refined_map, refined_cost = local_refine_full_map(full_map, comp_dict, max_iters=LOCAL_MAX_ITERS, rows=rows, cols=cols)
                if refined_cost < best_global_cost:
                    best_global_cost = refined_cost
                    best_global = refined_map
                    best_solution_details = {
                        'center_set': idx_set,
                        'center_assign': assign_center,
                        'left_perm': left_perm,
                        'right_perm': right_perm,
                        'cost_before_refine': tot,
                        'cost_after_refine': refined_cost
                    }
        # quick stop if perfect (cost ~ 0)
        if best_global_cost < 1e-6:
            break

    if best_global is None:
        print("No good solution found. Try increasing CENTER_BEAM or MAX_CANDIDATES.")
        return None, None, None

    # rebuild full image
    full_img = np.zeros_like(np.array(Image.open(image_path).convert('RGB').resize((IMAGE_WIDTH, IMAGE_HEIGHT)))/255.0)
    for pos, gidx in enumerate(best_global):
        r, c = divmod(pos, cols)
        full_img[r*PATCH_HEIGHT:(r+1)*PATCH_HEIGHT, c*PATCH_WIDTH:(c+1)*PATCH_WIDTH, :] = pieces[gidx]

    if verbose:
        print(f"Solved. Best cost = {best_global_cost:.4f}. Time = {time.time()-t0:.1f}s")
    return full_img, best_global, best_solution_details

# ---------- visualize ----------
def visualize_solution(original_path, arranged_img, mapping):
    orig = np.array(Image.open(original_path).convert('RGB').resize((IMAGE_WIDTH, IMAGE_HEIGHT)))/255.0
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    axes[0].imshow(orig); axes[0].set_title("Original"); axes[0].axis('off')
    axes[1].imshow(arranged_img); axes[1].set_title("Reconstructed (best)"); axes[1].axis('off')
    plt.show()

# ========== MAIN ==========
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeCNN().to(DEVICE)
    model.load_state_dict(torch.load("edge_cnn_best.pth", map_location=DEVICE))
    model.eval()
    all_paths = glob.glob(os.path.join(IMAGE_DIR, '*.jpg')) + glob.glob(os.path.join(IMAGE_DIR, '*.png'))
    if not all_paths:
        raise ValueError(f"No images in {IMAGE_DIR}")
    test_image = random.choice(all_paths)
    print("Solving image:", test_image)
    arranged_img, mapping, details = solve_full_puzzle(test_image)
    if arranged_img is not None:
        print("Mapping (pos->piece index):", mapping)
        print("Details:", details)
        visualize_solution(test_image, arranged_img, mapping)
