# Baseline evaluation

`oracle` replays gold decisions (pipeline ceiling, and the source of training labels). `rule` is the lexical-overlap baseline a fine-tuned Laya must beat.

## Decider: `oracle`

### Solver (linear/ILP via Z3): 14/15 end-to-end

| example | ok | status | questions | decider calls |
|---|---|---|---|---|
| chickens_cows | ✅ | ok | 22 | 5 |
| tickets | ✅ | ok | 22 | 5 |
| ages | ✅ | ok | 22 | 5 |
| coins | ✅ | ok | 22 | 5 |
| sum_diff | ✅ | ok | 22 | 5 |
| rectangle | ✅ | ok | 22 | 5 |
| pens | ✅ | ok | 22 | 5 |
| marbles | ✅ | ok | 36 | 6 |
| bakery | ✅ | ok | 25 | 5 |
| feed_mix | ✅ | ok | 25 | 5 |
| plumber | ✅ | ok | 12 | 4 |
| apples | ✅ | ok | 22 | 5 |
| savings | ✅ | ok | 12 | 4 |
| van | ✅ | ok | 13 | 4 |
| discount | ❌ | not representable | 6 | 2 |

### Node-RED (live deploy + probes): 20/20 end-to-end

| example | ok | status | questions | decider calls |
|---|---|---|---|---|
| kitchen_alert | ✅ | deployed | 8 | 2 |
| pressure_drop | ✅ | deployed | 8 | 2 |
| watch_livingroom | ✅ | deployed | 3 | 2 |
| watch_soil | ✅ | deployed | 3 | 2 |
| bridge_legacy | ✅ | deployed | 4 | 2 |
| bridge_plant | ✅ | deployed | 4 | 2 |
| poll_weather | ✅ | deployed | 3 | 2 |
| poll_health | ✅ | deployed | 3 | 2 |
| prices_to_mqtt | ✅ | deployed | 5 | 2 |
| air_to_mqtt | ✅ | deployed | 5 | 2 |
| echo_post | ✅ | deployed, probes 1/1 | 3 | 2 |
| echo_ping | ✅ | deployed, probes 1/1 | 3 | 2 |
| check_temp | ✅ | deployed, probes 2/2 | 5 | 2 |
| battery | ✅ | deployed, probes 2/2 | 5 | 2 |
| log_requests | ✅ | deployed, probes 1/1 | 3 | 2 |
| save_events | ✅ | deployed, probes 1/1 | 3 | 2 |
| door_throttle | ✅ | deployed | 5 | 2 |
| motion_throttle | ✅ | deployed | 5 | 2 |
| clock_tick | ✅ | deployed | 4 | 2 |
| heartbeat | ✅ | deployed | 4 | 2 |

## Decider: `rule`

### Solver (linear/ILP via Z3): 0/15 end-to-end

| example | ok | status | questions | decider calls |
|---|---|---|---|---|
| chickens_cows | ❌ | solved, wrong answer 0 (expected 7) | 12 | 4 |
| tickets | ❌ | solved, wrong answer 0 (expected 70) | 12 | 4 |
| ages | ❌ | solved, wrong answer 0 (expected 40) | 27 | 7 |
| coins | ❌ | solved, wrong answer 0 (expected 18) | 12 | 4 |
| sum_diff | ❌ | solved, wrong answer 0 (expected 35) | 12 | 4 |
| rectangle | ❌ | invalid | 36 | 6 |
| pens | ❌ | invalid | 22 | 5 |
| marbles | ❌ | solved, wrong answer 0 (expected 30) | 12 | 4 |
| bakery | ❌ | solved, wrong answer 0 (expected 180) | 12 | 4 |
| feed_mix | ❌ | invalid | 22 | 5 |
| plumber | ❌ | solved, wrong answer 0 (expected 4) | 12 | 4 |
| apples | ❌ | invalid | 36 | 6 |
| savings | ❌ | solved, wrong answer 0 (expected 24) | 12 | 4 |
| van | ❌ | solved, wrong answer 0 (expected 24) | 13 | 4 |
| discount | ❌ | solved, wrong answer 0 (expected 100) | 12 | 4 |

Teacher-forced decision accuracy: **155/297 = 52.2%**

| decision kind | accuracy |
|---|---|
| `c.const` | 2/26 = 8% |
| `c.rel` | 21/26 = 81% |
| `c.rhs` | 6/26 = 23% |
| `c.v.mag` | 20/50 = 40% |
| `c.v.sign` | 43/52 = 83% |
| `domain` | 11/14 = 79% |
| `goal` | 12/14 = 86% |
| `n_cons` | 5/14 = 36% |
| `n_vars` | 5/14 = 36% |
| `nonneg` | 13/14 = 93% |
| `obj.v.mag` | 1/5 = 20% |
| `obj.v.sign` | 5/5 = 100% |
| `target` | 6/11 = 55% |
| `var` | 5/26 = 19% |

### Node-RED (live deploy + probes): 8/20 end-to-end

| example | ok | status | questions | decider calls |
|---|---|---|---|---|
| kitchen_alert | ❌ | deployed; wrong template | 5 | 2 |
| pressure_drop | ❌ | deployed; wrong template | 5 | 2 |
| watch_livingroom | ❌ | static: http request 4d80a1ba2219e1cf: a slot had no candidate in the request; wrong template | 3 | 2 |
| watch_soil | ✅ | deployed | 3 | 2 |
| bridge_legacy | ❌ | deployed; wrong topic_out | 4 | 2 |
| bridge_plant | ❌ | static: http request 09d2aab76cb6b3c5: a slot had no candidate in the request; wrong template | 5 | 2 |
| poll_weather | ❌ | deployed; wrong interval | 3 | 2 |
| poll_health | ✅ | deployed | 3 | 2 |
| prices_to_mqtt | ✅ | deployed | 5 | 2 |
| air_to_mqtt | ❌ | deployed; wrong interval | 5 | 2 |
| echo_post | ✅ | deployed, probes 1/1 | 3 | 2 |
| echo_ping | ✅ | deployed, probes 1/1 | 3 | 2 |
| check_temp | ❌ | deployed, probes 1/2; wrong property | 5 | 2 |
| battery | ❌ | deployed, probes 0/2; wrong template | 3 | 2 |
| log_requests | ❌ | static: http request fb4f661c2a31d51e: a slot had no candidate in the request; wrong template | 5 | 2 |
| save_events | ❌ | deployed, probes 0/1; wrong template | 3 | 2 |
| door_throttle | ✅ | deployed | 5 | 2 |
| motion_throttle | ✅ | deployed | 5 | 2 |
| clock_tick | ❌ | static: http request 98ceab1ab0ee6cf7: a slot had no candidate in the request; wrong template | 5 | 2 |
| heartbeat | ✅ | deployed | 4 | 2 |

Teacher-forced decision accuracy: **68/86 = 79.1%**

| decision kind | accuracy |
|---|---|
| `broker` | 12/12 = 100% |
| `file` | 2/2 = 100% |
| `interval` | 6/8 = 75% |
| `method` | 4/4 = 100% |
| `operator` | 1/4 = 25% |
| `path` | 6/6 = 100% |
| `property` | 1/4 = 25% |
| `template` | 12/20 = 60% |
| `threshold` | 4/4 = 100% |
| `topic` | 10/10 = 100% |
| `topic_in` | 2/2 = 100% |
| `topic_out` | 0/2 = 0% |
| `url` | 8/8 = 100% |
