#!/usr/bin/env python3
"""JSON-lines bridge from Ctrl-World rollout code to the StarVLA policy server.

The Ctrl-World runtime environment does not need StarVLA dependencies.  This
helper runs with the StarVLA Python environment, keeps one websocket connection
to the policy server, and exchanges JSON lines with the Ctrl-World process.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starvla_root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument("--mode", default="vla")
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600)
    return parser.parse_args()


def decode_image(payload: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(payload["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(payload["dtype"]))
    return arr.reshape(tuple(payload["shape"])).copy()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.starvla_root))

    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    print(json.dumps({"type": "ready", "metadata": metadata}, ensure_ascii=False, default=str), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("type") == "close":
            break

        images = [decode_image(item) for item in request["images"]]
        example = {
            "lang": request["lang"],
            "image": images,
        }
        query = {
            "type": "infer",
            "request_id": request.get("request_id", "ctrlworld-policy-rollout"),
            "examples": [example],
            "do_sample": False,
            "use_ddim": True,
            "num_ddim_steps": int(request.get("num_ddim_steps", args.num_ddim_steps)),
            "mode": request.get("mode", args.mode),
            "num_steps": int(request.get("num_steps", request.get("num_ddim_steps", args.num_ddim_steps))),
            "seed": int(request.get("seed", 0)),
        }
        response = client.predict_action(query)
        if response.get("status") != "ok":
            error = response.get("error", {})
            print(
                json.dumps(
                    {
                        "ok": False,
                        "request_id": query["request_id"],
                        "error": error.get("message", str(error)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue

        data = response["data"]
        actions = np.asarray(data["normalized_actions"], dtype=np.float32)
        print(
            json.dumps(
                {
                    "ok": True,
                    "request_id": query["request_id"],
                    "response_keys": sorted(data.keys()),
                    "action_shape": list(actions.shape),
                    "normalized_actions": actions.tolist(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    client.close()


if __name__ == "__main__":
    main()
