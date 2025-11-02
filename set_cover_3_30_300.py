# -*- coding: utf-8 -*-
"""
Set Cover optimizer for the 3-30-300 rule (minimum number of trees)
-------------------------------------------------------------------

This script selects tree candidate locations along social/green corridors
so that each residential parcel sees at least a minimum number of trees
(e.g., 3 trees within a 30 m visibility radius), while minimizing the
number of trees selected. It implements a greedy, impact-maximizing
Set Cover heuristic.

Requirements
------------
- ArcGIS Pro / arcpy
- A file geodatabase (GDB) containing:
  - Residential parcels (polygons)
  - Social/green corridors (lines)

Typical workflow
----------------
1) Generate candidate points every N meters along the corridors.
2) Buffer each candidate by the visibility radius (e.g., 30 m).
3) Spatial join buffers to parcels to map candidate↔parcel coverage.
4) Run a greedy Set Cover that prioritizes candidates with highest impact.
5) Export selected tree points as a feature class in the GDB.

CLI usage (defaults shown)
--------------------------
python code/set_cover_3_30_300.py \
  --gdb "D:\data\city\project.gdb" \
  --parcels-search "residencial_3" \
  --corridors-search "corredores_interSOC" \
  --output-name "trees_optimized" \
  --candidate-spacing 5 \
  --visibility-radius 30 \
  --min-trees-per-parcel 3

Author: Jose Martinez
License: MIT
"""

import os
import sys
import argparse
from collections import defaultdict, Counter

import arcpy


# ----------------------------- Helpers --------------------------------- #
def find_feature_classes(gdb: str, search_term: str):
    """Return full paths of feature classes in `gdb` whose name contains `search_term`."""
    matches = []
    for dirpath, dirnames, filenames in arcpy.da.Walk(gdb, datatype="FeatureClass"):
        for fname in filenames:
            if search_term.lower() in fname.lower():
                matches.append(os.path.join(dirpath, fname))
    return matches


def calculate_maximum_impact(
    cand_id: int,
    candidate_to_parcels: dict[int, set[int]],
    parcel_coverage: dict[int, int],
    min_required: int,
):
    """
    Compute the MAXIMUM impact a candidate may have right now.

    Impact = number of parcels currently under-covered that would gain at most 1 unit
    of coverage if this candidate is selected (bounded by their remaining need).
    Returns (impact, total_parcels_covered_by_candidate).
    """
    covered_parcels = candidate_to_parcels.get(cand_id, set())
    impact = 0

    for parcel in covered_parcels:
        current = parcel_coverage.get(parcel, 0)
        if current < min_required:
            contribution = min(1, min_required - current)
            impact += contribution

    return impact, len(covered_parcels)


def ultra_efficient_set_cover(
    parcels_needing_trees: set[int],
    candidate_to_parcels: dict[int, set[int]],
    min_trees_needed: int,
):
    """
    Greedy Set Cover that maximizes impact each iteration.
    Prioritizes candidates that cover *more* parcels that are still under target.
    """
    selected: list[int] = []
    parcel_coverage: dict[int, int] = {p: 0 for p in parcels_needing_trees}
    available_candidates = set(candidate_to_parcels.keys())

    iteration = 0
    total_target = len(parcels_needing_trees) * min_trees_needed
    print(f"Target: {total_target} total coverages for {len(parcels_needing_trees)} parcels")

    while available_candidates:
        iteration += 1

        # Stop if all parcels meet the rule
        under_covered = {p for p, count in parcel_coverage.items() if count < min_trees_needed}
        if not under_covered:
            print(f"All parcels satisfied in {iteration-1} iterations.")
            break

        # Pick best candidate by "impact + small tie-break on total coverage"
        best_candidate = None
        best_score = 0.0
        best_details = None

        for cand_id in available_candidates:
            impact, total_parcels_covered = calculate_maximum_impact(
                cand_id, candidate_to_parcels, parcel_coverage, min_trees_needed
            )
            if impact > 0:
                score = impact + (total_parcels_covered * 0.1)
                if score > best_score:
                    best_score = score
                    best_candidate = cand_id
                    best_details = (impact, total_parcels_covered)

        if best_candidate is None or best_score == 0:
            print(f"No more useful candidates at iteration {iteration}.")
            break

        # Select and update coverage
        selected.append(best_candidate)
        available_candidates.remove(best_candidate)

        for parcel in candidate_to_parcels.get(best_candidate, set()):
            if parcel in parcels_needing_trees:
                parcel_coverage[parcel] += 1

        if iteration % 10 == 0 or iteration <= 20:
            satisfied = sum(1 for c in parcel_coverage.values() if c >= min_trees_needed)
            total_coverage = sum(parcel_coverage.values())
            impact, total_cov = best_details
            print(f"Iter {iteration}: picked candidate {best_candidate}")
            print(f"  Impact: {impact}, Parcels covered by candidate: {total_cov}")
            print(f"  Parcels satisfied: {satisfied}, Remaining: {len(under_covered)}")
            print(f"  Total coverage: {total_coverage}/{total_target}")

        # Early stop if completion rate is very high and progress slows
        if iteration > 100 and iteration % 50 == 0:
            satisfied_now = sum(1 for c in parcel_coverage.values() if c >= min_trees_needed)
            rate = satisfied_now / len(parcels_needing_trees)
            if rate > 0.95:
                print(f"Early stop: {rate*100:.1f}% parcels satisfied.")
                break

    return selected, parcel_coverage


# ----------------------------- Main ------------------------------------ #
def run(
    gdb_path: str,
    parcels_search: str,
    corridors_search: str,
    output_name: str,
    candidate_spacing: float,
    visibility_radius: float,
    min_trees_per_parcel: int,
):
    arcpy.env.overwriteOutput = True
    arcpy.env.workspace = gdb_path

    # 1) Check GDB
    print(f"GDB: {gdb_path} | Exists: {arcpy.Exists(gdb_path)}")
    if not arcpy.Exists(gdb_path):
        raise RuntimeError("Geodatabase not found.")

    # 2) Find feature classes
    parcels_matches = find_feature_classes(gdb_path, parcels_search)
    corridors_matches = find_feature_classes(gdb_path, corridors_search)
    if not parcels_matches or not corridors_matches:
        raise RuntimeError("Required feature classes not found. Check search terms.")

    parcels_fc = parcels_matches[0]
    corridors_fc = corridors_matches[0]
    print(f"Parcels FC:   {parcels_fc}")
    print(f"Corridors FC: {corridors_fc}")

    # 3) Temp FCs / outputs
    candidates_fc = os.path.join(gdb_path, "tmp_candidates_pts")
    cand_buf_fc = os.path.join(gdb_path, "tmp_candidates_buf")
    cand_parcel_join = os.path.join(gdb_path, "tmp_cand_parcel_join")
    output_trees = os.path.join(gdb_path, output_name)

    try:
        print("\n=== Ultra-optimized Set Cover for minimum trees ===")

        # Generate candidate points
        if arcpy.Exists(candidates_fc):
            arcpy.management.Delete(candidates_fc)
        print(f"Generating candidate points every {candidate_spacing} m...")

        arcpy.management.GeneratePointsAlongLines(
            corridors_fc,
            candidates_fc,
            "DISTANCE",
            Distance=f"{candidate_spacing} Meters",
            Include_End_Points="NO_END_POINTS",
        )
        n_pts = int(arcpy.management.GetCount(candidates_fc).getOutput(0))
        print(f"Candidates generated: {n_pts}")
        if n_pts == 0:
            raise RuntimeError("No candidate points were generated.")

        # Buffers (visibility areas)
        if arcpy.Exists(cand_buf_fc):
            arcpy.management.Delete(cand_buf_fc)
        print(f"Creating visibility buffers of {visibility_radius} m...")

        arcpy.analysis.Buffer(
            candidates_fc,
            cand_buf_fc,
            f"{visibility_radius} Meters",
            "FULL",
            "ROUND",
            "NONE",
            None,
            "PLANAR",
        )

        # Spatial join: candidate buffers ↔ parcels
        if arcpy.Exists(cand_parcel_join):
            arcpy.management.Delete(cand_parcel_join)
        print("Running spatial join (buffers ↔ parcels)...")

        arcpy.analysis.SpatialJoin(
            cand_buf_fc,
            parcels_fc,
            cand_parcel_join,
            "JOIN_ONE_TO_MANY",
            "KEEP_COMMON",
            match_option="INTERSECT",
        )

        # Build coverage maps
        print("Building coverage maps...")
        candidate_to_parcels: dict[int, set[int]] = defaultdict(set)
        parcel_to_candidates: dict[int, set[int]] = defaultdict(set)

        with arcpy.da.SearchCursor(cand_parcel_join, ["TARGET_FID", "JOIN_FID"]) as cur:
            for target_fid, join_fid in cur:
                candidate_to_parcels[int(target_fid)].add(int(join_fid))
                parcel_to_candidates[int(join_fid)].add(int(target_fid))

        # Candidate efficiency stats
        print("Analyzing candidate efficiency...")
        candidate_eff = {cid: len(ps) for cid, ps in candidate_to_parcels.items()}
        if candidate_eff:
            max_cov = max(candidate_eff.values())
            avg_cov = sum(candidate_eff.values()) / len(candidate_eff)
            high_eff = sum(1 for eff in candidate_eff.values() if eff >= avg_cov)
            n_max = sum(1 for eff in candidate_eff.values() if eff == max_cov)
            print(f"Candidates at max coverage ({max_cov} parcels): {n_max}")
            print(f"High-efficiency candidates (>= {avg_cov:.1f} parcels): {high_eff}")

        # All parcels and coverability
        all_parcels = set()
        oid_field = arcpy.Describe(parcels_fc).OIDFieldName
        with arcpy.da.SearchCursor(parcels_fc, [oid_field]) as cur:
            for (oid,) in cur:
                all_parcels.add(int(oid))

        coverable_parcels = set(parcel_to_candidates.keys())
        uncoverable_parcels = all_parcels - coverable_parcels
        print("\nParcels summary:")
        print(f"- Total parcels:     {len(all_parcels)}")
        print(f"- Coverable parcels: {len(coverable_parcels)}")
        print(f"- Out of reach:      {len(uncoverable_parcels)}")
        if not coverable_parcels:
            print("No coverable parcels. Exiting.")
            return

        # Run Set Cover
        print("\n=== Running ultra-efficient Set Cover ===")
        print(f"Goal: {min_trees_per_parcel} visible trees per parcel")

        selected_candidates, final_coverage = ultra_efficient_set_cover(
            coverable_parcels, candidate_to_parcels, min_trees_per_parcel
        )

        print("\n=== Final results ===")
        print(f"Selected trees: {len(selected_candidates)}")

        satisfied = sum(1 for c in final_coverage.values() if c >= min_trees_per_parcel)
        total_cov = sum(final_coverage.values())
        pct = 100 * satisfied / len(coverable_parcels) if coverable_parcels else 0
        print(f"Parcels satisfied: {satisfied}/{len(coverable_parcels)} ({pct:.1f}%)")

        if len(selected_candidates) > 0:
            eff = satisfied / len(selected_candidates) if selected_candidates else 0
            cov_per_tree = total_cov / len(selected_candidates)
            print(f"Efficiency: {eff:.2f} satisfied parcels per tree")
            print(f"Coverage:   {cov_per_tree:.2f} parcels covered per tree")

            coverage_dist = Counter(final_coverage.values())
            print("\nDistribution of trees per parcel:")
            for trees, count in sorted(coverage_dist.items()):
                status = " ✓ OK" if trees >= min_trees_per_parcel else " ✗ INSUFFICIENT"
                print(f"  {trees} trees: {count} parcels{status}")

            insufficient = [p for p, c in final_coverage.items() if c < min_trees_per_parcel]
            if insufficient:
                print(f"\nParcels with insufficient coverage: {len(insufficient)}")
                print("Likely located far from corridors or candidate visibility zones.")

        if not selected_candidates:
            print("ERROR: No trees were selected. Check inputs/parameters.")
            return

        # Export selected points
        print("\nExporting optimized solution...")

        buf_oid = arcpy.Describe(cand_buf_fc).OIDFieldName
        sel_layer = "tmp_sel_layer"
        if arcpy.Exists(sel_layer):
            arcpy.management.Delete(sel_layer)
        arcpy.MakeFeatureLayer_management(cand_buf_fc, sel_layer)

        batch_size = 500
        for i in range(0, len(selected_candidates), batch_size):
            batch = selected_candidates[i : i + batch_size]
            where = f"{arcpy.AddFieldDelimiters(cand_buf_fc, buf_oid)} IN ({','.join(map(str, batch))})"
            selection_type = "ADD_TO_SELECTION" if i > 0 else "NEW_SELECTION"
            arcpy.SelectLayerByAttribute_management(sel_layer, selection_type, where)

        cand_layer = "tmp_cand_layer"
        if arcpy.Exists(cand_layer):
            arcpy.management.Delete(cand_layer)
        arcpy.MakeFeatureLayer_management(candidates_fc, cand_layer)
        arcpy.SelectLayerByLocation_management(cand_layer, "INTERSECT", sel_layer, selection_type="NEW_SELECTION")

        if arcpy.Exists(output_trees):
            arcpy.management.Delete(output_trees)
        arcpy.management.CopyFeatures(cand_layer, output_trees)

        final_count = int(arcpy.management.GetCount(output_trees).getOutput(0))
        print(f"\n✓ Exported {final_count} optimized tree points")
        print(f"Output feature class: {output_trees}")

        print("\n=== Optimization summary ===")
        print("Strategy: maximize the marginal impact of each selected tree")
        print(f"Result:   {len(selected_candidates)} minimum trees")
        print(f"Coverage: {satisfied} parcels meet the 3-30-300 rule")

    except Exception as e:
        import traceback

        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ultra-efficient Set Cover optimizer for the 3-30-300 rule."
    )
    parser.add_argument("--gdb", dest="gdb_path", required=True, help="Path to the file geodatabase (GDB).")
    parser.add_argument("--parcels-search", default="residencial_3", help="Substring to locate the parcels FC.")
    parser.add_argument("--corridors-search", default="corredores_interSOC", help="Substring to locate the corridors FC.")
    parser.add_argument("--output-name", default="trees_optimized", help="Name of output feature class in the GDB.")
    parser.add_argument("--candidate-spacing", type=float, default=5.0, help="Meters between candidate points.")
    parser.add_argument("--visibility-radius", type=float, default=30.0, help="Visibility buffer radius in meters.")
    parser.add_argument("--min-trees-per-parcel", type=int, default=3, help="Minimum visible trees per parcel.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(
        gdb_path=args.gdb_path,
        parcels_search=args.parcels_search,
        corridors_search=args.corridors_search,
        output_name=args.output_name,
        candidate_spacing=args.candidate_spacing,
        visibility_radius=args.visibility_radius,
        min_trees_per_parcel=args.min_trees_per_parcel,
    )
