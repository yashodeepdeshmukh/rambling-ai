"""Node-RED flow templates: a pattern plus slots, each slot filled by one typed decision."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from .. import extract

MISSING = "<none found>"
METHODS = ("GET", "POST", "PUT", "DELETE")
OPERATORS = {"greater than": "gt", "less than": "lt", "at least": "gte",
             "at most": "lte", "equal to": "eq", "not equal to": "neq"}
INTERVAL_DEFAULTS = (1, 5, 10, 30, 60, 300, 3600)


def _or_missing(items) -> tuple[str, ...]:
    items = tuple(dict.fromkeys(items))[:20]
    return items or (MISSING,)


SOURCES: dict[str, Callable[[str], tuple[str, ...]]] = {
    "topic": lambda t: _or_missing(extract.topics(t)),
    "url": lambda t: _or_missing(extract.urls(t)),
    "path": lambda t: _or_missing(extract.http_paths(t)),
    "file": lambda t: _or_missing(extract.files(t)),
    "host": lambda t: _or_missing(extract.hosts(t) + ["localhost"]),
    "seconds": lambda t: _or_missing(f"{s} seconds" for s in extract.durations(t) + list(INTERVAL_DEFAULTS)),
    "number": lambda t: _or_missing(extract.fmt(n) for n in extract.numbers(t)),
    "method": lambda t: METHODS,
    "operator": lambda t: tuple(OPERATORS),
    "property": lambda t: _or_missing(["payload"] + [f"payload.{w}" for w in extract.entities(t, 19)]),
}


@dataclass(frozen=True)
class Slot:
    name: str
    source: str
    question: str


class FlowBuilder:
    """Tiny helper producing Node-RED flow JSON with deterministic ids and a left-to-right layout."""

    def __init__(self, label: str):
        self.label = label
        self.tab = {"id": self._id("tab"), "type": "tab", "label": label, "disabled": False, "info": ""}
        self.nodes = [self.tab]

    def _id(self, key: str) -> str:
        return hashlib.sha1(f"{self.label}/{key}/{len(getattr(self, 'nodes', []))}".encode()).hexdigest()[:16]

    def config(self, type_: str, **props) -> dict:
        node = {"id": self._id(type_), "type": type_, **props}
        self.nodes.append(node)
        return node

    def add(self, type_: str, col: int, row: int = 0, outputs: int = 1, **props) -> dict:
        node = {"id": self._id(type_), "type": type_, "z": self.tab["id"], "name": "",
                **props, "x": 160 + 200 * col, "y": 80 + 80 * row,
                "wires": [[] for _ in range(outputs)]}
        self.nodes.append(node)
        return node

    @staticmethod
    def wire(src: dict, dst: dict, port: int = 0):
        src["wires"][port].append(dst["id"])


# ------------------------------------------------------------- node factories

def broker(fb, host):
    h, _, port = host.partition(":")
    return fb.config("mqtt-broker", name="", broker=h, port=port or "1883", clientid="",
                     autoConnect=True, usetls=False, protocolVersion="4", keepalive="60",
                     cleansession=True)


def mqtt_in(fb, col, topic, brk, row=0):
    return fb.add("mqtt in", col, row, topic=topic, qos="0", datatype="json",
                  broker=brk["id"], nl=False, rap=True, rh=0, inputs=0)


def mqtt_out(fb, col, topic, brk, row=0):
    return fb.add("mqtt out", col, row, outputs=0, topic=topic, qos="", retain="", broker=brk["id"])


def debug(fb, col, row=0):
    return fb.add("debug", col, row, outputs=0, active=True, tosidebar=True, console=False,
                  tostatus=False, complete="payload", targetType="msg")


def inject(fb, col, seconds, row=0):
    return fb.add("inject", col, row, props=[{"p": "payload"}], repeat=str(seconds), crontab="",
                  once=True, onceDelay=0.1, topic="", payload="", payloadType="date")


def http_request(fb, col, method, url, ret="txt", row=0):
    return fb.add("http request", col, row, method=method, ret=ret, paytoqs="ignore", url=url,
                  tls="", persist=False, proxy="", authType="", senderr=False, headers=[])


def http_in(fb, col, method, path, row=0):
    return fb.add("http in", col, row, url=path, method=method.lower(), upload=False, swaggerDoc="")


def http_response(fb, col, row=0):
    return fb.add("http response", col, row, outputs=0, statusCode="", headers={})


def switch(fb, col, prop, op, value, otherwise=False, row=0):
    rules = [{"t": OPERATORS[op], "v": value, "vt": "num"}] + ([{"t": "else"}] if otherwise else [])
    return fb.add("switch", col, row, outputs=len(rules), property=prop, propertyType="msg",
                  rules=rules, checkall="false" if otherwise else "true", repair=False)


def text(fb, col, body, row=0):
    return fb.add("template", col, row, field="payload", fieldType="msg", format="text",
                  syntax="plain", template=body, output="str")


def seconds(s: str) -> int:
    return int(s.split()[0])


# ------------------------------------------------------------- templates

@dataclass(frozen=True)
class Template:
    name: str
    summary: str
    slots: tuple[Slot, ...]
    build: Callable[[FlowBuilder, dict], None]


def _mqtt_threshold_webhook(fb, s):
    b = broker(fb, s["broker"])
    src = mqtt_in(fb, 0, s["topic"], b)
    sw = switch(fb, 1, s["property"], s["operator"], s["threshold"])
    req = http_request(fb, 2, s["method"], s["url"])
    out = debug(fb, 3)
    fb.wire(src, sw); fb.wire(sw, req); fb.wire(req, out)


def _mqtt_monitor(fb, s):
    b = broker(fb, s["broker"])
    fb.wire(mqtt_in(fb, 0, s["topic"], b), debug(fb, 1))


def _mqtt_bridge(fb, s):
    b = broker(fb, s["broker"])
    fb.wire(mqtt_in(fb, 0, s["topic_in"], b), mqtt_out(fb, 1, s["topic_out"], b))


def _poll_http_debug(fb, s):
    t = inject(fb, 0, seconds(s["interval"]))
    req = http_request(fb, 1, "GET", s["url"], ret="obj")
    fb.wire(t, req); fb.wire(req, debug(fb, 2))


def _poll_http_to_mqtt(fb, s):
    b = broker(fb, s["broker"])
    t = inject(fb, 0, seconds(s["interval"]))
    req = http_request(fb, 1, "GET", s["url"])
    fb.wire(t, req); fb.wire(req, mqtt_out(fb, 2, s["topic"], b))


def _http_echo(fb, s):
    fb.wire(http_in(fb, 0, s["method"], s["path"]), http_response(fb, 1))


def _http_threshold(fb, s):
    src = http_in(fb, 0, "POST", s["path"])
    sw = switch(fb, 1, s["property"], s["operator"], s["threshold"], otherwise=True)
    alert, ok = text(fb, 2, "ALERT", row=0), text(fb, 2, "OK", row=1)
    res = http_response(fb, 3)
    fb.wire(src, sw); fb.wire(sw, alert, 0); fb.wire(sw, ok, 1)
    fb.wire(alert, res); fb.wire(ok, res)


def _webhook_to_file(fb, s):
    src = http_in(fb, 0, "POST", s["path"])
    f = fb.add("file", 1, filename=s["file"], filenameType="str", appendNewline=True,
               createDir=True, overwriteFile="false", encoding="none")
    done, res = text(fb, 2, "saved"), http_response(fb, 3)
    fb.wire(src, f); fb.wire(f, done); fb.wire(done, res)


def _rate_limit_forward(fb, s):
    b = broker(fb, s["broker"])
    src = mqtt_in(fb, 0, s["topic"], b)
    d = fb.add("delay", 1, pauseType="rate", timeout="5", timeoutUnits="seconds", rate="1",
               nbRateUnits=str(seconds(s["interval"])), rateUnits="second", randomFirst="1",
               randomLast="5", randomUnits="seconds", drop=True, allowrate=False, outputs=1)
    req = http_request(fb, 2, "POST", s["url"])
    fb.wire(src, d); fb.wire(d, req)


def _timer_publish(fb, s):
    b = broker(fb, s["broker"])
    fb.wire(inject(fb, 0, seconds(s["interval"])), mqtt_out(fb, 1, s["topic"], b))


S = Slot
BROKER = S("broker", "host", "Which MQTT broker host should be used?")
TEMPLATES = (
    Template("mqtt_threshold_webhook", "MQTT value crosses a threshold, then call a URL",
             (BROKER, S("topic", "topic", "Which MQTT topic carries the value?"),
              S("property", "property", "Which message field holds the value to compare?"),
              S("operator", "operator", "How is the value compared with the threshold?"),
              S("threshold", "number", "What is the threshold?"),
              S("method", "method", "Which HTTP method calls the URL?"),
              S("url", "url", "Which URL is called?")), _mqtt_threshold_webhook),
    Template("mqtt_monitor", "Show messages from an MQTT topic in debug",
             (BROKER, S("topic", "topic", "Which MQTT topic should be watched?")), _mqtt_monitor),
    Template("mqtt_bridge", "Forward MQTT messages from one topic to another topic",
             (BROKER, S("topic_in", "topic", "Which topic do messages come from?"),
              S("topic_out", "topic", "Which topic do messages go to?")), _mqtt_bridge),
    Template("poll_http_debug", "Fetch a URL on a timer and show the result in debug",
             (S("interval", "seconds", "How often is the URL fetched?"),
              S("url", "url", "Which URL is fetched?")), _poll_http_debug),
    Template("poll_http_to_mqtt", "Fetch a URL on a timer and publish the response to MQTT",
             (S("interval", "seconds", "How often is the URL fetched?"),
              S("url", "url", "Which URL is fetched?"), BROKER,
              S("topic", "topic", "Which MQTT topic receives the response?")), _poll_http_to_mqtt),
    Template("http_echo", "HTTP endpoint that replies with the request it receives",
             (S("method", "method", "Which HTTP method does the endpoint accept?"),
              S("path", "path", "What is the endpoint path?")), _http_echo),
    Template("http_threshold", "HTTP endpoint that compares a posted value and replies ALERT or OK",
             (S("path", "path", "What is the endpoint path?"),
              S("property", "property", "Which field of the posted body holds the value?"),
              S("operator", "operator", "Which comparison means ALERT?"),
              S("threshold", "number", "What is the threshold?")), _http_threshold),
    Template("webhook_to_file", "HTTP endpoint that appends each posted body to a file",
             (S("path", "path", "What is the endpoint path?"),
              S("file", "file", "Which file receives the bodies?")), _webhook_to_file),
    Template("rate_limit_forward", "Forward MQTT messages to a URL, at most one per interval",
             (BROKER, S("topic", "topic", "Which MQTT topic is forwarded?"),
              S("interval", "seconds", "At most one message per how long?"),
              S("url", "url", "Which URL receives the messages?")), _rate_limit_forward),
    Template("timer_publish", "Publish a timestamp to an MQTT topic on a timer",
             (S("interval", "seconds", "How often is a message published?"), BROKER,
              S("topic", "topic", "Which MQTT topic is published to?")), _timer_publish),
)
BY_NAME = {t.name: t for t in TEMPLATES}
BY_SUMMARY = {t.summary: t for t in TEMPLATES}
