import pytest

from rambling import datasets, extract
from rambling.deciders import OracleDecider
from rambling.nodered import build, live, validate

REQUESTS = datasets.nodered_requests()


@pytest.mark.parametrize("ex", REQUESTS, ids=[r["id"] for r in REQUESTS])
def test_gold_decisions_build_a_valid_flow(ex):
    template, slots, nodes = build.build(ex["text"], OracleDecider(build.gold_answers(ex)))
    assert template == ex["template"] and slots == ex["slots"]
    assert validate.validate(nodes) == []


def test_validator_catches_dangling_wires_and_unknown_types():
    nodes = [{"id": "t", "type": "tab"},
             {"id": "a", "type": "inject", "z": "t", "repeat": "5", "wires": [["ghost"]]},
             {"id": "b", "type": "telepathy", "z": "t", "wires": []}]
    errors = validate.validate(nodes)
    assert any("missing node ghost" in e for e in errors)
    assert any("unknown node type" in e for e in errors)


def test_extractors_separate_urls_paths_topics_files_and_hosts():
    t = "Poll https://api.example.com/v1/x every 5 minutes into home/x on mqtt.local, POST /hook, log /tmp/a.log"
    assert extract.urls(t) == ["https://api.example.com/v1/x"]
    assert extract.topics(t) == ["home/x"]
    assert extract.http_paths(t) == ["/hook"]
    assert extract.files(t) == ["/tmp/a.log"]
    assert extract.hosts(t) == ["mqtt.local"]
    assert extract.durations(t) == [300]


@pytest.mark.skipif(not live.available(), reason="run `npm install` in ./nodered for live tests")
def test_http_flows_behave_correctly_in_real_node_red():
    probed = [ex for ex in REQUESTS if ex.get("probes")]
    with live.NodeRedServer() as srv:
        for ex in probed:
            _, _, nodes = build.build(ex["text"], OracleDecider(build.gold_answers(ex)))
            assert srv.deploy(nodes)[0], ex["id"]
            for p in ex["probes"]:
                ok, msg = srv.probe(p)
                assert ok, f"{ex['id']}: {msg}"
