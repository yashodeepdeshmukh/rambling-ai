"""Static checks on generated flow JSON: catches broken flows without running Node-RED."""

from __future__ import annotations

from .templates import MISSING

CORE_TYPES = {
    "inject", "debug", "complete", "catch", "status", "link in", "link out", "link call",
    "comment", "function", "switch", "change", "range", "template", "delay", "trigger",
    "exec", "rbe", "tls-config", "http proxy", "mqtt in", "mqtt out", "mqtt-broker",
    "http in", "http response", "http request", "websocket in", "websocket out",
    "websocket-listener", "websocket-client", "tcp in", "tcp out", "tcp request", "udp in",
    "udp out", "csv", "html", "json", "xml", "yaml", "split", "join", "sort", "batch",
    "file", "file in", "watch", "junction", "global-config",
}
CONFIG_TYPES = {"mqtt-broker", "tls-config", "http proxy", "websocket-listener",
                "websocket-client", "global-config"}

REQUIRED = {
    "mqtt in": lambda n: n["topic"] and n["broker"],
    "mqtt out": lambda n: n["topic"] and n["broker"],
    "http request": lambda n: str(n["url"]).startswith(("http://", "https://")),
    "http in": lambda n: str(n["url"]).startswith("/") and n["method"] in {"get", "post", "put", "delete", "patch"},
    "inject": lambda n: n.get("repeat", "") == "" or float(n["repeat"]) > 0,
    "switch": lambda n: n["rules"] and len(n["wires"]) == len(n["rules"]),
    "file": lambda n: str(n["filename"]).startswith("/"),
}


def validate(nodes: list[dict], known_types: set[str] | None = None) -> list[str]:
    """Returns a list of problems; empty means the flow passed every static check."""
    known = known_types or CORE_TYPES
    errors = []
    ids = [n["id"] for n in nodes]
    if len(ids) != len(set(ids)):
        errors.append("duplicate node ids")
    by_id = {n["id"]: n for n in nodes}
    tabs = {n["id"] for n in nodes if n["type"] == "tab"}
    incoming = {i: 0 for i in by_id}

    for n in nodes:
        t = n["type"]
        if t == "tab":
            continue
        if MISSING in map(str, n.values()):
            errors.append(f"{t} {n['id']}: a slot had no candidate in the request")
        if t not in known:
            errors.append(f"{n['id']}: unknown node type {t!r}")
            continue
        if t in CONFIG_TYPES:
            continue
        if n.get("z") not in tabs:
            errors.append(f"{t} {n['id']}: not on a tab")
        for port in n.get("wires", []):
            for dst in port:
                if dst not in by_id:
                    errors.append(f"{t} {n['id']}: wire to missing node {dst}")
                elif by_id[dst]["type"] in CONFIG_TYPES | {"tab"}:
                    errors.append(f"{t} {n['id']}: wire to non-flow node {dst}")
                else:
                    incoming[dst] += 1
        if "broker" in n and n["broker"] not in by_id:
            errors.append(f"{t} {n['id']}: missing broker config")
        check = REQUIRED.get(t)
        try:
            if check and not check(n):
                errors.append(f"{t} {n['id']}: invalid configuration")
        except (KeyError, ValueError) as e:
            errors.append(f"{t} {n['id']}: invalid configuration ({e})")

    flow_nodes = [n for n in nodes if n["type"] not in CONFIG_TYPES | {"tab"}]
    for n in flow_nodes:
        if len(flow_nodes) > 1 and not incoming[n["id"]] and not any(n.get("wires", [])):
            errors.append(f"{n['type']} {n['id']}: not connected")
    return errors
