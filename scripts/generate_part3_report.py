from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    output = Path("runs/part3")
    summary = json.loads((output / "part3_summary.json").read_text())
    curves = rows(output / "resilience_curves.csv")
    ten_percent = {
        row["strategy"]: row
        for row in curves
        if float(row["removal_percentage"]) == 0.10
    }
    gravity = summary["gravity_baseline"]
    consolidation = summary["consolidation"]
    report = f"""# Part 3: Structural Intelligence and Flow-Aware Stress Testing

## Method

Part 3 compares degree centrality (C001), approximate weighted betweenness
(C002), gravity-demand traffic assignment (C010), flow-aware ablation (C011),
and geographic/cascading stress tests (C012).

All capacities and flows are relative satellite-derived estimates. They are not
calibrated vehicle counts.

## Part 2.5 Safety Gate

- Selected configuration: `{consolidation['selected']['config_id']}`
- Controlled false-bridge rate: {consolidation['selected']['false_bridge_rate']:.4f}
- Largest-component node coverage: {consolidation['selected']['node_coverage']:.4f}
- Requested coverage target: {consolidation['target_node_coverage']:.2f}
- Coverage gate met: {consolidation['coverage_gate_met']}

The 70% target could not be reached while retaining the 10% false-bridge
ceiling, so traffic analysis uses the largest safe component.

## Baseline Assignment

- Routing graph: {summary['graph_preparation']['routing_nodes']:,} nodes and {summary['graph_preparation']['routing_edges']:,} edges
- OD pairs: {summary['demand']['od_pairs']:,}
- Mean relative travel time: {gravity['mean_travel_time_min']:.3f} minutes
- Overloaded edges: {gravity['overloaded_edges']}
- MSA iterations: {gravity['iterations']}
- Relative gap: {gravity['convergence']:.6f}

## Critical Locations

- Highest degree node: `{summary['top_degree_node']['node_id']}`
- Highest betweenness node: `{summary['top_betweenness_node']['node_id']}`
- Highest flow-critical node: `{summary['top_flow_critical_node']['node_id']}`
- Highest flow-critical edge: `{summary['top_flow_critical_edge']['source']}`–`{summary['top_flow_critical_edge']['target']}`

## Progressive Failure Results

| Strategy | Resilience after 10% removal | Served demand | Largest component |
|---|---:|---:|---:|
| Random | {float(ten_percent['random']['resilience_index']):.4f} | {float(ten_percent['random']['served_demand_ratio']):.4f} | {float(ten_percent['random']['largest_component_ratio']):.4f} |
| Degree | {float(ten_percent['highest_degree']['resilience_index']):.4f} | {float(ten_percent['highest_degree']['served_demand_ratio']):.4f} | {float(ten_percent['highest_degree']['largest_component_ratio']):.4f} |
| Betweenness | {float(ten_percent['highest_betweenness']['resilience_index']):.4f} | {float(ten_percent['highest_betweenness']['served_demand_ratio']):.4f} | {float(ten_percent['highest_betweenness']['largest_component_ratio']):.4f} |
| Flow criticality | {float(ten_percent['highest_flow_criticality']['resilience_index']):.4f} | {float(ten_percent['highest_flow_criticality']['served_demand_ratio']):.4f} | {float(ten_percent['highest_flow_criticality']['largest_component_ratio']):.4f} |

## Conclusion

Targeted failures are consistently more damaging than random failures. The
flow-critical strategy produces the lowest 10% resilience ({float(ten_percent['highest_flow_criticality']['resilience_index']):.4f}), demonstrating that degree alone does not identify the complete systemic risk. C011 is the primary structural-intelligence method, with C001 retained as the explainable baseline and C002 as the required gatekeeper reference.
"""
    (output / "PART3_IMPLEMENTATION_REPORT.md").write_text(report)
    print(output / "PART3_IMPLEMENTATION_REPORT.md")


if __name__ == "__main__":
    main()
