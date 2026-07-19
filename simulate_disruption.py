from __future__ import annotations

import argparse
import json
from pathlib import Path

from mobility.scenario import ScenarioSpec
from mobility.simulation import SimulationEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a transport-network disruption."
    )
    parser.add_argument("--config", default="configs/phase4.yaml")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--mode", choices=("preview", "exact"), default="exact")
    parser.add_argument("--output")
    args = parser.parse_args()

    engine = SimulationEngine.from_config(args.config)
    scenario = ScenarioSpec.from_json(args.scenario)
    result = (
        engine.preview(scenario)
        if args.mode == "preview"
        else engine.simulate(scenario)
    )
    output = Path(args.output or engine.output / "scenarios" / scenario.scenario_id)
    result.to_directory(output)
    print(json.dumps(result.summary, indent=2))
    print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
