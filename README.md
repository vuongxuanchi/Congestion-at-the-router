
# NS-3 Packet-Level Trace Dataset of Network Congestion at a Bottleneck Link

## 1. Overview

This dataset contains packet-level traces generated with the **ns-3 discrete-event network simulator** to study congestion at a single bottleneck link.

A **50-node dumbbell topology** carries **300 concurrent flows** belonging to three traffic classes:

- **Elephant:** Long-lived TCP flows.
- **Sheep:** Medium-rate UDP On/Off flows.
- **Mice:** Sparse, small-packet UDP On/Off flows.

Every flow is forced to cross the bottleneck link.

The same traffic scenario is run under seven queue management schemes at the bottleneck router:

1. DropTail
2. CoDel
3. FQ-CoDel
4. PIE
5. CAKE-equivalent FQ-COBALT
6. RED
7. Adaptive RED

Each configuration is repeated using independent replications with different RNG run numbers.

For every run, one unified CSV file records:

- IPv4 transmit/receive events at every node, including IP and TCP/UDP header fields, ports, TCP sequence numbers, and ACK flags.
- Enqueue, dequeue, and drop events of the bottleneck queue in both directions, including queue occupancy and per-packet queueing delay.
- TCP RTT estimator updates.

The dataset was produced for doctoral research on:

- Traffic classification (Mice/Sheep/Elephant).
- Federated Reinforcement Learning (FRL).
- Delay-based Active Queue Management (CoDel and dynamic variants).

The DropTail traces are intended as a neutral baseline on which other AQM algorithms can be re-implemented and evaluated offline (e.g., in Python).

### Possible Uses

- Benchmarking AQM algorithms through offline trace-driven evaluation.
- Flow classification and elephant-flow detection with ground-truth labels derived from destination port numbers (see Section 5).
- Training machine-learning and reinforcement-learning models for congestion control.
- Teaching ns-3 tracing and network congestion behaviour.

---

## 2. Simulation Environment

| Item | Value |
|:---|:---|
| Simulator | ns-3.42 |
| OS / Compiler | Ubuntu 22.04, GCC 11.4 (example configuration) |
| Simulation script | `simulation_la50_v6.cc` (internal revision v8, see change log in the file header) |
| Simulation time | 180 seconds per run |
| Application start time | 1.0–1.5 seconds |
| RNG seed | 10 (`RngSeedManager::SetSeed`) |
| RNG runs | 1–5 (`--run=1..5`), independent replications |
| Randomness sources | Start-time jitter U(0, 0.5) seconds per application; per-flow On rate; exponential On/Off durations |

---

## 3. Network Topology and Link Parameters

The network uses a 50-node dumbbell topology consisting of 10 client nodes, 2 routers, and 38 server nodes.

All traffic flows traverse the central bottleneck link between Router-1 (N11) and Router-2 (N12).

### Network Topology

```text
 N1 ─┐                                            ┌─ N13
 N2 ─┤                                            ├─ N14
 ... ├──────── N11 ══════════════════ N12 ────────┤ ...
 N10─┘        (Router-1)  bottleneck  (Router-2)  └─ N50
                          10 Mbps / 50 ms

          10 client nodes                38 server nodes

 Access links: 20 Mbps / 5 ms
 Bottleneck link: 10 Mbps / 50 ms
```

### Node and IP Address Mapping

| Node (1-based) | ns-3 index | Role | IPv4 address |
|:---|:---:|:---|:---|
| N1–N10 | 0–9 | Clients (senders) | Nk → 10.1.k.1 (k = 1…10) |
| N11 | 10 | Router-1 (bottleneck ingress) | 10.1.k.2 on access side; 10.1.100.1 on bottleneck |
| N12 | 11 | Router-2 (bottleneck egress) | 10.1.100.2 on bottleneck; 10.1.m.2 on access side |
| N13–N50 | 12–49 | Servers (receivers) | Node index j → 10.1.(j−1).1 (N13 = 10.1.11.1 … N50 = 10.1.48.1) |

### Link Configuration

| Link | Bandwidth | Propagation Delay |
|:---|:---:|:---:|
| Access links | 20 Mbps | 5 ms |
| Bottleneck link | 10 Mbps | 50 ms |

---

## 4. Queue Disciplines at the Bottleneck

Queue disciplines are selected using the command-line argument `--qdisc`.

All queue disciplines use a **300-packet limit**.

| `--qdisc` | ns-3 class | Notes |
|:---|:---|:---|
| `droptail` | `DropTailQueue<Packet>` (device queue) | Baseline; traffic-control layer disabled (`DisableFlowControl()`) so no default FQ-CoDel is installed. |
| `codel` | `CoDelQueueDisc` | Target = 5 ms, Interval = 100 ms (RFC 8289). |
| `fqcodel` | `FqCoDelQueueDisc` | ns-3 defaults. |
| `pie` | `PieQueueDisc` | ns-3 defaults. |
| `cake` | `FqCobaltQueueDisc` | CAKE-equivalent (COBALT AQM + flow queueing); ns-3 mainline has no full CAKE implementation. |
| `red` | `RedQueueDisc` | ns-3 defaults. |
| `ared` | `RedQueueDisc` with `ARED=true` | Adaptive RED. |

---

## 5. Traffic Model (300 Flows)

For each index `i = 0…99`, three flows (one per traffic class) are created on the same client/server pair but with different destination ports.

### Client and Server Mapping

- Client node = `i mod 10` (N1–N10).
- Server node = `12 + (i mod 38)` (N13–N50).

### Traffic Characteristics

| Class | Count | Transport / Application | Destination Port | Packet Size | Sending Rate | On / Off (Exponential Mean) | Expected Duty Cycle |
|:---|:---:|:---|:---:|:---|:---|:---:|:---:|
| Elephant | 100 | TCP BulkSend, unlimited bytes | `20000 + i` | Full segments: 1400 B data, 1432 B IP payload (TCP header with options); application write size 1200 B is re-segmented by TCP. | Elastic (TCP congestion control) | Always on | > 0.9 |
| Sheep | 100 | UDP OnOff | `21000 + i` | 600 B payload (608 B IP payload) | U(30, 80) kbps while On, fixed per flow | 0.50 s / 0.15 s | ≈ 0.77 |
| Mice | 100 | UDP OnOff | `22000 + i` | 150 B payload (158 B IP payload) | U(40, 120) kbps while On, fixed per flow | 0.03 s / 0.17 s | ≈ 0.15 |

### Traffic Class Summary

| Traffic Class | Number of Flows | Transport Protocol | Main Characteristics |
|:---|:---:|:---|:---|
| Elephant | 100 | TCP | Long-lived, congestion-responsive flows |
| Sheep | 100 | UDP | Medium-rate, intermittent flows |
| Mice | 100 | UDP | Sparse, short-duration, small-packet flows |
| **Total** | **300** | **TCP / UDP** | **Three traffic classes** |

### Ground-Truth Traffic Classification

Traffic classes can be identified using destination port numbers:

| Destination Port Range | Traffic Class |
|:---|:---|
| 20000–20099 | Elephant |
| 21000–21099 | Sheep |
| 22000–22099 | Mice |

These port ranges provide ground-truth labels for supervised traffic classification and machine-learning experiments.

---

## 6. How to Reproduce

The following instructions describe how to reproduce the simulation experiments using ns-3.

### Step 1. Copy the Simulation Script

Copy the simulation script into the ns-3 scratch directory:

```bash
cp simulation_la50_v6.cc ~/ns-3-dev/scratch/
```

### Step 2. Build and Run One Configuration

Navigate to the ns-3 directory and build the project:

```bash
cd ~/ns-3-dev
./ns3 build
```

Run one simulation using the DropTail queue discipline:

```bash
./ns3 run "scratch/simulation_la50_v6 --qdisc=droptail --seed=10 --run=1"
```

### Step 3. Run All Queue Disciplines and Replications

To reproduce the complete experimental dataset, execute seven queue disciplines with five independent replications each.

```bash
for q in droptail codel fqcodel pie cake red ared; do
  for r in 1 2 3 4 5; do
    ./ns3 run "scratch/simulation_la50_v6 --qdisc=$q --seed=10 --run=$r"
  done
done
```

### Experimental Configuration Summary

| Parameter | Value |
|:---|:---:|
| Number of queue disciplines | 7 |
| Replications per queue discipline | 5 |
| Total simulation runs | 35 |
| Simulation duration per run | 180 seconds |
| Total simulated time | 6,300 seconds |

---

## 7. Known Limitations

The dataset has the following limitations:

1. **Constant packet size within each On/Off flow:** Due to an `OnOffHelper` limitation, packet size is constant within each flow. Therefore, the within-flow packet-size coefficient of variation is close to zero for Sheep and Mice.

2. **Non-responsive UDP traffic:** Sheep and Mice are both non-responsive UDP flows. Only Elephant flows react to network congestion through TCP congestion control.

3. **CAKE approximation:** The `cake` configuration is approximated using `FqCobaltQueueDisc`, rather than being a one-to-one port of Linux CAKE.

4. **Topology availability:** Only the 50-node scenario is included in this version. The 6-node and 20-node scenarios may be added in future versions.

---

## 8. How to Cite

If you use this dataset in your research, please cite it as follows:

**Vuong Xuan Chi (2026).** *NS-3 Packet-Level Trace Dataset of Network Congestion at a Bottleneck Link* (Version v1.0) [Data set]. Zenodo.

**DOI:** [https://doi.org/10.5281/zenodo.23831638](https://doi.org/10.5281/zenodo.23831638)

---

## Dataset Information

| Attribute | Description |
|:---|:---|
| Dataset title | NS-3 Packet-Level Trace Dataset of Network Congestion at a Bottleneck Link |
| Version | v1.0 |
| Simulator | ns-3.42 |
| Network topology | 50-node dumbbell topology |
| Number of traffic flows | 300 |
| Traffic classes | Elephant, Sheep, Mice |
| Queue disciplines | DropTail, CoDel, FQ-CoDel, PIE, FQ-COBALT, RED, Adaptive RED |
| Repository | Zenodo |
| DOI | [10.5281/zenodo.23831638](https://doi.org/10.5281/zenodo.23831638) |

---

**Keywords:** ns-3, Network Congestion, Active Queue Management, AQM, PIE, CoDel, FQ-CoDel, RED, Adaptive RED, FQ-COBALT, Traffic Classification, Elephant Flow, Mice Flow, Sheep Flow, Reinforcement Learning, Federated Reinforcement Learning, Packet-Level Traces, Dumbbell Topology.
