
<div class="dataset-document">

    <h1>NS-3 PACKET-LEVEL TRACE DATASET OF NETWORK CONGESTION</h1>

    <h2>1. Overview</h2>

    <p>
        This dataset contains packet-level traces generated with the ns-3
        discrete-event network simulator to study congestion at a single
        bottleneck link. A 50-node dumbbell topology carries 300 concurrent
        flows belonging to three traffic classes: Elephant (long-lived TCP),
        Sheep (medium-rate UDP On/Off), and Mice (sparse, small-packet UDP
        On/Off). Every flow is forced to cross the bottleneck.
    </p>

    <p>
        The same traffic scenario is run under seven queue management
        schemes at the bottleneck router (DropTail, CoDel, FQ-CoDel, PIE,
        CAKE-equivalent FQ-COBALT, RED, and Adaptive RED), each repeated
        as independent replications with different RNG run numbers.
    </p>

    <p>For every run, one unified CSV file records:</p>

    <ul>
        <li>
            IPv4 transmit/receive events at every node (with IP, TCP/UDP
            header fields, ports, TCP sequence number, and ACK flag).
        </li>
        <li>
            Enqueue/dequeue/drop events of the bottleneck queue in both
            directions (with queue occupancy and per-packet queueing delay).
        </li>
        <li>TCP RTT estimator updates.</li>
    </ul>

    <p>
        The dataset was produced for doctoral research on traffic
        classification (Mice/Sheep/Elephant), federated reinforcement
        learning (FRL), and delay-based active queue management (CoDel
        and dynamic variants). The DropTail traces are intended as a
        neutral baseline on which other AQM algorithms can be
        re-implemented and evaluated offline (e.g., in Python).
    </p>

    <p><strong>Possible uses:</strong></p>

    <ul>
        <li>Benchmarking AQM algorithms (offline trace-driven evaluation).</li>
        <li>
            Flow classification / elephant-flow detection with ground-truth
            labels (derived from port numbers, see Section 5).
        </li>
        <li>
            Training machine-learning / reinforcement-learning models for
            congestion control.
        </li>
        <li>Teaching ns-3 tracing and congestion behaviour.</li>
    </ul>

    <h2>2. Simulation Environment</h2>

    <table border="1" cellpadding="6" cellspacing="0" width="100%">
        <thead>
            <tr>
                <th>Item</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Simulator</td><td>ns-3.42</td></tr>
            <tr><td>OS / compiler</td><td>[e.g. Ubuntu 22.04, GCC 11.4]</td></tr>
            <tr>
                <td>Simulation script</td>
                <td>simulation_la50_v6.cc (internal revision v8, see change log in the file header)</td>
            </tr>
            <tr><td>Simulation time</td><td>180 s per run (applications start at 1.0–1.5 s)</td></tr>
            <tr><td>RNG seed</td><td>10 (RngSeedManager::SetSeed)</td></tr>
            <tr><td>RNG runs</td><td>1–5 (--run=1..5), independent replications</td></tr>
            <tr>
                <td>Randomness sources</td>
                <td>
                    Start-time jitter U(0, 0.5) s per application;
                    per-flow On rate; exponential On/Off durations.
                </td>
            </tr>
        </tbody>
    </table>

    <h2>3. Network Topology and Link Parameters</h2>

    <pre>
 N1 ─┐                                            ┌─ N13
 N2 ─┤   access 20 Mbps / 5 ms                    ├─ N14
 ... ├──────── N11 ══════════════════ N12 ────────┤  ...
 N10─┘        (Router-1)  bottleneck  (Router-2)  └─ N50
                          10 Mbps / 50 ms
       10 client nodes                         38 server nodes
    </pre>

    <h3>Node and IP Address Mapping</h3>

    <table border="1" cellpadding="6" cellspacing="0" width="100%">
        <thead>
            <tr>
                <th>Node (1-based)</th>
                <th>ns-3 index</th>
                <th>Role</th>
                <th>IPv4 address</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>N1–N10</td>
                <td>0–9</td>
                <td>Clients (senders)</td>
                <td>Nk → 10.1.k.1 (k = 1…10)</td>
            </tr>
            <tr>
                <td>N11</td>
                <td>10</td>
                <td>Router-1 (bottleneck ingress)</td>
                <td>10.1.k.2 on access side, 10.1.100.1 on bottleneck</td>
            </tr>
            <tr>
                <td>N12</td>
                <td>11</td>
                <td>Router-2 (bottleneck egress)</td>
                <td>10.1.100.2 on bottleneck, 10.1.m.2 on access side</td>
            </tr>
            <tr>
                <td>N13–N50</td>
                <td>12–49</td>
                <td>Servers (receivers)</td>
                <td>Node index j → 10.1.(j−1).1 (N13 = 10.1.11.1 … N50 = 10.1.48.1)</td>
            </tr>
        </tbody>
    </table>

    <h2>4. Queue Disciplines at the Bottleneck</h2>

    <p>
        Queue disciplines are selected with the command-line argument
        <code>--qdisc</code>. All use a 300-packet limit.
    </p>

    <table border="1" cellpadding="6" cellspacing="0" width="100%">
        <thead>
            <tr>
                <th>--qdisc</th>
                <th>ns-3 class</th>
                <th>Notes</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>droptail</td>
                <td>DropTailQueue&lt;Packet&gt; (device queue)</td>
                <td>
                    Baseline; traffic-control layer disabled
                    (DisableFlowControl()) so no default FQ-CoDel is installed.
                </td>
            </tr>
            <tr>
                <td>codel</td>
                <td>CoDelQueueDisc</td>
                <td>Target 5 ms, Interval 100 ms (RFC 8289).</td>
            </tr>
            <tr>
                <td>fqcodel</td>
                <td>FqCoDelQueueDisc</td>
                <td>ns-3 defaults.</td>
            </tr>
            <tr>
                <td>pie</td>
                <td>PieQueueDisc</td>
                <td>ns-3 defaults.</td>
            </tr>
            <tr>
                <td>cake</td>
                <td>FqCobaltQueueDisc</td>
                <td>
                    CAKE-equivalent (COBALT AQM + flow queueing);
                    ns-3 mainline has no full CAKE implementation.
                </td>
            </tr>
            <tr>
                <td>red</td>
                <td>RedQueueDisc</td>
                <td>ns-3 defaults.</td>
            </tr>
            <tr>
                <td>ared</td>
                <td>RedQueueDisc with ARED=true</td>
                <td>Adaptive RED.</td>
            </tr>
        </tbody>
    </table>

    <h2>5. Traffic Model (300 Flows)</h2>

    <p>
        For each index i = 0…99, three flows (one per class) are created
        on the same client/server pair but with different destination ports:
    </p>

    <ul>
        <li>Client node = i mod 10 (N1–N10).</li>
        <li>Server node = 12 + (i mod 38) (N13–N50).</li>
    </ul>

    <table border="1" cellpadding="6" cellspacing="0" width="100%">
        <thead>
            <tr>
                <th>Class</th>
                <th>Count</th>
                <th>Transport / App</th>
                <th>Destination port</th>
                <th>Packet size</th>
                <th>Sending rate</th>
                <th>On / Off (exponential mean)</th>
                <th>Expected duty cycle</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Elephant</td>
                <td>100</td>
                <td>TCP BulkSend, unlimited bytes</td>
                <td>20000 + i</td>
                <td>
                    Full segments: 1400 B data, 1432 B IP payload
                    (TCP header with options); app write size 1200 B
                    is re-segmented by TCP.
                </td>
                <td>Elastic (TCP congestion control)</td>
                <td>Always on</td>
                <td>&gt; 0.9</td>
            </tr>
            <tr>
                <td>Sheep</td>
                <td>100</td>
                <td>UDP OnOff</td>
                <td>21000 + i</td>
                <td>600 B payload (608 B IP payload)</td>
                <td>U(30, 80) kbps while On, fixed per flow</td>
                <td>0.50 s / 0.15 s</td>
                <td>≈ 0.77</td>
            </tr>
            <tr>
                <td>Mice</td>
                <td>100</td>
                <td>UDP OnOff</td>
                <td>22000 + i</td>
                <td>150 B payload (158 B IP payload)</td>
                <td>U(40, 120) kbps while On, fixed per flow</td>
                <td>0.03 s / 0.17 s</td>
                <td>≈ 0.15</td>
            </tr>
        </tbody>
    </table>

    <h2>6. How to Reproduce</h2>

    <p><strong>Step 1. Copy the script into the ns-3 scratch folder.</strong></p>

    <pre><code>cp simulation_la50_v6.cc ~/ns-3-dev/scratch/</code></pre>

    <p><strong>Step 2. Build and run one configuration.</strong></p>

    <pre><code>cd ~/ns-3-dev
./ns3 build
./ns3 run "scratch/simulation_la50_v6 --qdisc=droptail --seed=10 --run=1"</code></pre>

    <p><strong>Step 3. Run the full set: 7 queue disciplines × 5 replications.</strong></p>

    <pre><code>for q in droptail codel fqcodel pie cake red ared; do
  for r in 1 2 3 4 5; do
    ./ns3 run "scratch/simulation_la50_v6 --qdisc=$q --seed=10 --run=$r"
  done
done</code></pre>

    <h2>7. Known Limitations</h2>

    <ul>
        <li>
            Packet size is constant within each On/Off flow (an OnOffHelper
            limitation), so the within-flow packet-size coefficient of
            variation is close to 0 for Sheep and Mice.
        </li>
        <li>
            Sheep and Mice are both non-responsive UDP; only Elephant
            flows react to congestion.
        </li>
        <li>
            cake is approximated by FqCobaltQueueDisc, not a 1:1 port
            of Linux CAKE.
        </li>
        <li>
            Only the 50-node scenario is included in this version.
            [Edit if the 6- and 20-node scenarios are added.]
        </li>
    </ul>

    <h2>8. How to Cite</h2>

    <p>
        Vuong Xuan Chi, (2026). NS-3 Packet-Level Trace Dataset of Network
        Congestion at a Bottleneck Link (Version v1.0) [Data set]. Zenodo.
    </p>

    <p>
        <a href="https://doi.org/10.5281/zenodo.23831638"
           target="_blank" rel="noopener noreferrer">
            https://doi.org/10.5281/zenodo.23831638
        </a>
    </p>

</div>
