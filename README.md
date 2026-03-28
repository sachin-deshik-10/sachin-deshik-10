<h1 align="center">Hi 👋, I'm Sachin Deshik</h1>
<h3 align="center">AI • Computer Vision • Sensor Systems • IoT / Edge • Cybersecurity</h3>

<img src="https://readme-typing-svg.herokuapp.com/?font=JetBrains+Mono&weight=700&size=26&duration=3200&pause=1200&color=00D9FF&center=true&vCenter=true&width=860&height=55&lines=Sachin+Deshik+%E2%80%94+Systems+%26+AI+Engineer;Sensor+Fusion+%7C+Edge+Intelligence+%7C+O-RAN;Real+constraints.+Deliberate+trade-offs.+Deployable+systems." alt="Typing SVG" />
 
<br/>
 
<p align="center">
  <img src="https://img.shields.io/badge/Focus-Applied_AI_%26_Systems_Engineering-00D9FF?style=flat-square&labelColor=0d1117" />
  <img src="https://img.shields.io/badge/Domain-Sensor_Fusion_%7C_CV_%7C_RF_%7C_IoT-7C3AED?style=flat-square&labelColor=0d1117" />
  <img src="https://img.shields.io/badge/Status-Open_to_Research_%26_Internships-22C55E?style=flat-square&labelColor=0d1117" />
</p>
 
<p align="center">
“The best way to predict the future is to invent it.” — Alan Kay
</p>

---

## Profile
 
I design and deploy **AI-driven sensing and signal processing systems** under hard real-world constraints — latency budgets, hardware limits, noisy data, and adversarial environments. My work occupies the boundary where **applied ML meets signal engineering and systems architecture**.
 
An ECE background (RV College of Engineering, Bangalore) means I engage with problems at the physics layer — signal acquisition, sensor noise, RF propagation — before reaching for an algorithm. I think in **end-to-end pipelines**: from waveform or pixel to decision output, with explicit attention to where components fail and why.
 
> *"The gap between a working prototype and a deployable system is where most of the real engineering happens."*
 
---
 
## 🧬 Engineering Philosophy
 
```
Reliability > novelty.
Every architecture decision is a trade-off — make it explicit.
A model that fails gracefully beats one that performs well only in isolation.
The bottleneck is almost never where you expect it.
```
 
| Principle | Practice |
|---|---|
| **Pipeline-first thinking** | Define data contracts at every interface before writing any model |
| **Failure isolation** | Components should fail independently; cascades are architecture failures |
| **Observability by design** | If you can't measure it, you can't debug it — metrics are first-class deliverables |
| **Hardware-aware ML** | Quantization, ONNX export, and inference budget are design inputs, not post-hoc optimizations |
| **Trade-off explicitness** | Every "chose X over Y" decision is documented — undocumented trade-offs become future bugs |
 
---
 
## 🧠 How I Design Systems
 
Before writing code, four questions are answered:
 
**1 — What is the data contract at each interface?**
Input schema, noise characteristics, latency SLA, and failure semantics — defined upfront, not discovered during debugging.
 
**2 — Where will this break first?**
Identify the highest-risk component. Build failure handling there before building features anywhere else.
 
**3 — What is the irreducible latency floor?**
Hardware I/O, network round-trips, and inference compute set a physical lower bound. Everything above that is addressable. Knowing the difference matters.
 
**4 — How will I know it's working in production?**
If the answer is "I'll check manually," the system isn't finished. Observable state, metrics, and alerting are part of the deliverable.
 
---
 
## 🔬 Current Focus
 
| Area | Engineering Direction |
|---|---|
| **O-RAN + AI** | ML inference within near-RT RIC control loops; latency-constrained decision systems |
| **Edge Intelligence** | Sub-10ms inference on constrained hardware; model compression and runtime optimization |
| **Sensor Fusion** | Multi-modal target state estimation (radar + IR + optical); Kalman variants and track management |
| **Real-Time Telemetry** | Event-driven IoT architectures; QoS-aware message delivery under intermittent connectivity |
| **Secure Embedded AI** | Threat modeling for AI-enabled sensing systems; adversarial input characterization |
 
---
 
## 🚀 Engineering Projects
 
> Each project is documented as a system — explicit architecture, deliberate trade-offs, failure analysis, and contextual metrics.
 
---
 
### 🛰 Airborne Object Detection & Tracking System
 
**Problem Statement**
Small, fast-moving airborne targets in cluttered visual backgrounds constitute a failure case for generic object detectors — targets occupy fewer than 0.1% of frame pixels, exhibit unpredictable inter-frame motion, and are visually indistinguishable from background clutter (birds, atmospheric artifacts, sun glint). Pretrained YOLO weights optimized for large-object COCO classes perform below 40% recall on this target class without targeted domain adaptation.
 
**Approach**
Re-anchored YOLO architecture for small-target aspect ratios using K-means clustering on domain-specific bounding box distributions. Built a tracking layer maintaining target identity across frames via IoU-based assignment with Kalman filter state prediction during occlusion intervals.
 
**Architecture**
```
┌──────────────────────────────────────────────────────────────┐
│  Input: Video stream / sensor feed                          │
└──────────────────────────┬───────────────────────────────────┘
                           ↓
         Frame Preprocessing
         (resize 640×640, normalize, letterbox pad)
                           ↓
         YOLO Inference Engine
         (custom anchor config, low-IoU NMS at 0.35 threshold)
                           ↓
         Detection → Track Association
         (Hungarian algorithm, IoU cost matrix)
                           ↓
         Kalman Filter State Propagation
         (predict during occlusion; update on detection match)
                           ↓
┌──────────────────────────────────────────────────────────────┐
│  Output: Track ID, bbox, class conf, velocity estimate      │
└──────────────────────────────────────────────────────────────┘
```
> **System bottleneck:** YOLO inference (~28ms/frame on T4). Pre/post-processing accounts for <2ms. Tracker runs in <0.5ms. Any further latency optimization must target the inference stage.
 
**Tech Stack**
`Python` · `PyTorch` · `OpenCV` · `YOLOv5/v8` · `NumPy` · `ONNX`
 
**Key Contributions**
- Re-engineered anchor configurations via K-means on domain bounding box statistics — target aspect ratios clustered tightly around 1:1 at 8–16px, far outside COCO anchor distribution; re-anchoring improved small-target recall by ~18%
- Implemented Kalman-filter track prediction sustaining identity continuity across occlusion gaps of up to 8 consecutive frames at 30 FPS
- Added minimum track lifetime filter (≥3 frame persistence before track promotion) — eliminated ~85% of single-frame false alarms with negligible true-positive cost
- Exported inference graph to ONNX, enabling deployment to edge accelerators (Jetson-class hardware) without framework dependency
 
**Contextual Metrics**
- ~30 FPS throughput on NVIDIA T4 GPU at 640×640 input resolution, batch size 1
- ~40% reduction in missed detections versus per-frame detection baseline on occlusion-heavy synthetic sequences (≥4 frame occlusion events)
- NMS IoU threshold set to 0.35 vs. COCO default 0.45 — reduces false merges on spatially proximate small targets at cost of ~8% increase in duplicate detections, resolved by tracker-level deduplication
 
**⚖️ Engineering Trade-offs**
 
| Decision | Chosen | Rejected | Reasoning |
|---|---|---|---|
| Detector | YOLOv5/v8 | Faster R-CNN | Single-stage inference is 3–5× faster; two-stage detectors offer ~5% better small-object recall but cannot meet 30 FPS budget on available hardware |
| Tracker | IoU + Kalman | DeepSORT | DeepSORT's ReID embedding adds ~12ms per frame on CPU — unacceptable for real-time operation; classical IoU association is sufficient when frame-rate detection coverage is high |
| Input resolution | 640×640 | 1280×1280 | Doubling resolution quadruples FLOPs; 640 provides sufficient small-target resolution with 2× speed advantage |
| Track promotion threshold | 3 frames | 1 frame (immediate) | Immediate promotion passes single-frame clutter artifacts as tracks; 3-frame threshold eliminates transient false positives with <2% true-target miss cost |
 
**⚠️ Failure Modes & Learnings**
 
*Failure 1 — Clutter-induced false positive burst:*
Dense background textures (foliage, cloud edges) generated false detections that overwhelmed the tracker, creating hundreds of spurious track IDs per sequence. Root cause: detection threshold too permissive for high-texture backgrounds.
 
*Resolution:* Track lifetime filter requiring ≥3-frame persistence before confirmation. Atmospheric and texture artifacts are spatially incoherent across frames; legitimate targets persist. Eliminated ~85% of spurious tracks.
 
*Failure 2 — Tracker drift during high-acceleration maneuvers:*
Kalman constant-velocity model diverged when targets executed rapid direction changes, causing predicted state to miss the IoU association gate and breaking track identity.
 
*Resolution:* Increased process noise covariance Q to allow faster state correction; widened IoU matching gate during detected high-velocity frames (velocity > 2σ from track history). Reduced track loss events ~30% on high-maneuverability sequences.
 
*Failure 3 — Anchor mismatch causing sub-40% baseline recall:*
Default pretrained anchors clustered around human/vehicle scale. First deployment produced recall under 40% — detection failure was not an algorithm problem, it was a configuration mismatch.
 
*Resolution:* K-means re-anchoring on domain data. This is documented because the diagnostic process — distinguishing model failure from configuration failure — matters as much as the fix.
 
**🔍 Research Extensions**
- Transformer-based trackers (TrackFormer, MOTR) model long-range temporal dependencies and handle erratic motion profiles where Kalman constant-velocity assumptions break down
- Multi-sensor fusion (optical + IRST thermal channel) would enable detection under low-visibility and thermally camouflaged scenarios — a natural architectural extension
- Adaptive NMS with scene-density-aware thresholding is an open research direction for dense multi-target scenarios
 
**→** [Repository](https://github.com/sachin-deshik-10/IRST-Airborne-Object-Detection-Tracking-YOLO)
 
---
 
### 📡 RADAR Signal Processing Library
 
**Problem Statement**
Radar DSP pipelines are predominantly implemented in proprietary MATLAB environments. Research prototyping requires either expensive toolchain licenses or per-project reimplementation from first principles. A Python-native, modular, composable library covering the full radar processing chain — waveform generation through track output — does not exist at research-accessible quality with clean interface contracts.
 
**Approach**
Designed a radar processing toolkit with strict module boundaries — each stage exposes a consistent `process(data) → output` signature. Components are independently testable and substitutable. Simulation harness injects calibrated noise profiles for algorithm benchmarking without hardware.
 
**Architecture**
```
┌────────────────────────────────────────────────────────────────┐
│  Waveform Generator (LFM chirp, CW, stepped frequency)       │
└──────────────────────────┬─────────────────────────────────────┘
                           ↓
           Raw ADC Sampling + I/Q Separation
                           ↓
           Pulse Compression
           (matched filter, FFT-based, processing gain validation)
                           ↓
           Range-Doppler Map Generation
           (2D FFT, Hamming/Blackman window functions)
                           ↓
           CFAR Detection
           (CA-CFAR + OS-CFAR, configurable guard cells + Pfa)
                           ↓
           Target State Extraction
           (range, radial velocity, angular estimate)
                           ↓
┌────────────────────────────────────────────────────────────────┐
│  Track Output + Visualization (range-Doppler map, plots)     │
└────────────────────────────────────────────────────────────────┘
```
> **System bottleneck:** 2D FFT for range-Doppler map generation at large CPI sizes. Pulse compression is FFT-based and fast; the bottleneck shifts to Doppler dimension at high pulse count. Addressed via NumPy's FFT implementation with pre-computed window vectors.
 
**Tech Stack**
`Python` · `NumPy` · `SciPy` · `Matplotlib`
 
**Key Contributions**
- Implemented CA-CFAR and OS-CFAR detectors with configurable Pfa and guard cell parameters — validated to achieve target Pfa = 10⁻⁴ under simulated Gaussian clutter across SNR range −5 dB to +20 dB
- Built matched-filter pulse compression achieving theoretical processing gain — validated against SNR improvement formula (10·log₁₀(BT) for LFM waveform)
- Enforced continuous phase across pulse repetition intervals in waveform generator — a non-obvious numerical detail that eliminates inter-chirp spectral artifacts mimicking false Doppler returns
- Designed simulation harness covering 12 noise-and-clutter parameter combinations, including Swerling 1 and Swerling 3 target fluctuation models
 
**Contextual Metrics**
- 3× reduction in per-algorithm iteration time versus monolithic scripts — attributable to module-level unit tests and clean interface isolation enabling targeted component changes
- CFAR detection validated across SNR range −5 dB to +20 dB on simulated AWGN + clutter scenarios
- OS-CFAR masking resistance validated: maintains detection of secondary target when primary target occupies CFAR reference window (CA-CFAR fails this case)
 
**⚖️ Engineering Trade-offs**
 
| Decision | Chosen | Rejected | Reasoning |
|---|---|---|---|
| CFAR variants | CA-CFAR + OS-CFAR | GO-CFAR, VI-CFAR | CA-CFAR is optimal under homogeneous clutter; OS-CFAR handles clutter edges and multi-target reference windows where CA fails; these two cover the dominant operational regimes |
| FFT window | Hamming / Blackman | Rectangular | Rectangular window produces spectral leakage that smears Doppler returns across adjacent bins; Hamming reduces sidelobe level ~40 dB at cost of 1.3× main lobe broadening — necessary trade-off |
| Language | Python + NumPy | MATLAB | Reproducibility, CI/CD compatibility, zero cost, open-source accessibility; NumPy FFT performance is within 1.5× of MATLAB for batch simulation workloads |
 
**⚠️ Failure Modes & Learnings**
 
*Failure 1 — CFAR masking under dense target conditions:*
CA-CFAR averages reference cells to estimate clutter level. When multiple targets occupied the reference window simultaneously, the clutter estimate inflated, raising the detection threshold and masking weaker nearby targets.
 
*Resolution:* OS-CFAR selection when detected target density exceeds threshold — k-th ordered statistic estimation is robust to target contamination in the reference window. Pipeline selects CFAR variant dynamically.
 
*Failure 2 — Inter-chirp phase discontinuities generating false Doppler returns:*
Initial waveform generator produced phase jumps at pulse repetition boundaries, creating spectral artifacts in the Doppler dimension indistinguishable from real slow-moving targets.
 
*Resolution:* Tracked cumulative phase state across pulses and enforced phase continuity at boundaries. This was not documented in any referenced DSP textbook — discovered through systematic artifact analysis.
 
**🔍 Research Extensions**
- MUSIC and ESPRIT algorithms for super-resolution angle-of-arrival estimation beyond FFT angular resolution — relevant for applications requiring sub-beamwidth target separation
- CNN-based CFAR replacement for non-stationary clutter environments where classical CFAR statistical assumptions fail (e.g., sea clutter, urban multipath)
- Synthetic aperture processing (SAR mode) as a planned extension for ground imaging geometry
 
**→** [Repository](https://github.com/sachin-deshik-10/RADAR_LIBRARY)
 
---
 
### 🌌 IRST Sensor Processing Library
 
**Problem Statement**
Infrared Search & Track systems operate where frame-level detection is physically impossible: sub-pixel targets, signal-to-clutter ratio below 3 dB, and spatially correlated background clutter (sky gradient, atmospheric turbulence, cloud edges). Detection requires accumulating weak signal energy over multiple frames along predicted target trajectories — a fundamentally different paradigm from detect-then-track.
 
**Approach**
Constructed a thermal processing framework implementing spatiotemporal background subtraction, multi-frame non-coherent integration, and morphological clutter rejection. Threat trajectory simulator generates synthetic engagement scenarios for closed-loop evaluation without live hardware.
 
**Architecture**
```
┌─────────────────────────────────────────────────────────────┐
│  Thermal Frame Sequence (simulated / sensor)               │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
          Background Estimation
          (temporal median filter, adaptive update rate)
                            ↓
          Residual Frame Extraction
          (background subtraction + morphological noise rejection)
                            ↓
          Multi-Frame Non-Coherent Integration
          (energy accumulation along predicted trajectory, N frames)
                            ↓
          Adaptive Threshold + Candidate Extraction
          (SCR-based threshold, connected component labeling)
                            ↓
          Track-Before-Detect Logic
          (hypothesis-based trajectory confirmation over M frames)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Confirmed Track Output + Threat State Estimate            │
└─────────────────────────────────────────────────────────────┘
```
 
**Tech Stack**
`Python` · `OpenCV` · `NumPy` · `SciPy` · `Thermal simulation engine`
 
**Key Contributions**
- Implemented adaptive background subtraction using temporal median filtering with configurable exponential update rate — robust to slow background drift (sky gradient change across 30s engagement) without over-smoothing transient targets
- Built multi-frame non-coherent integration accumulating signal energy along predicted target trajectories — enables detection at pre-integration SCR below threshold, confirmed on synthetic dim-target sequences (SCR 1.5–3 dB)
- Designed parametric threat simulator generating 8 engagement scenario classes (head-on, crossing, receding, oblique geometries) with configurable atmospheric transmittance and sensor noise parameters
- Implemented image registration pre-processing using optical flow to compensate platform motion before background subtraction
 
**Contextual Metrics**
- ~25% improvement in detection probability on synthetic dim-target sequences (SCR 1.5–3 dB) versus single-frame threshold detection at matched Pfa = 10⁻³
- Background estimator converges to stable clutter estimate within 15–20 frames under moderate sky gradient (Δ < 2 counts/frame)
- Platform motion compensation reduced motion-induced false alarm rate by ~70% on simulated maneuvering platform scenarios
 
**⚖️ Engineering Trade-offs**
 
| Decision | Chosen | Rejected | Reasoning |
|---|---|---|---|
| Background model | Temporal median filter | Gaussian mixture model (GMM) | GMM is statistically rigorous but computationally expensive per-pixel; median filter achieves comparable clutter rejection at 4× lower compute — the target SCR range does not justify GMM's overhead |
| Integration | Non-coherent multi-frame | Coherent integration | Coherent integration requires precise phase knowledge — unavailable for passive IR sensors; non-coherent accumulation is the physically correct choice for incoherent thermal emission |
| Detection logic | Track-before-detect | Detect-then-track | Conventional detect-then-track requires each frame to independently exceed threshold — physically impossible at SCR < 3 dB; TBD accumulates evidence over time before committing |
 
**⚠️ Failure Modes & Learnings**
 
*Failure 1 — Atmospheric shimmer generating false track candidates:*
Turbulence-induced intensity fluctuations produced spatially localized residuals after background subtraction that matched the expected dim-target signature, generating false alarm bursts in high-turbulence scenarios.
 
*Resolution:* Temporal consistency filter — candidates must maintain spatially coherent position across ≥4 consecutive frames to advance to track stage. Atmospheric shimmer is spatially incoherent across frames and rejected; real targets are not.
 
*Failure 2 — Background estimator lag during platform maneuver:*
Rapid sensor platform motion shifted the background faster than the median filter adaptation rate, producing structured residuals across the entire frame — every edge became a false candidate.
 
*Resolution:* Optical flow-based image registration compensating frame-to-frame background motion before subtraction. Eliminated platform-motion-induced false alarm rate by ~70% on maneuvering platform test scenarios.
 
**→** [Repository](https://github.com/sachin-deshik-10/IRST_LIBRARY)
 
---
 
### 📶 5G AI Optimizer — OpenRAN
 
**Problem Statement**
O-RAN disaggregation exposes RAN parameters as software-controllable variables via standardized interfaces, enabling ML-driven optimization. The critical constraint: near-RT RIC control loops operate on 10–100ms timescales, making inference latency a hard architectural requirement. A model with 99% accuracy and 15ms inference is architecturally invalid for this application. Classical rule-based spectrum managers cannot adapt to non-stationary interference at required speed; most deep learning architectures cannot meet the latency budget.
 
**Approach**
Engineered a feature extraction pipeline over O-RAN KPI telemetry streams — lag features, rolling statistics, interference indicators — and benchmarked ML models explicitly on latency-accuracy trade-off rather than accuracy alone. Selected gradient-boosted ensemble as the optimal operating point: ~95% accuracy at ~2.1ms inference, versus LSTM at ~98% accuracy at ~12ms.
 
**Architecture**
```
┌──────────────────────────────────────────────────────────────────┐
│  O-RAN KPI Stream: RSRP, SINR, PRB utilization, CQI, UE count  │
└────────────────────────────┬─────────────────────────────────────┘
                             ↓
           Telemetry Ingestion + Windowing
           (sliding window: last N=10 time slots, 10ms granularity)
                             ↓
           Feature Engineering Pipeline
           (lag-1/3/5 features, 5-slot rolling mean/std,
            interference delta, PRB saturation binary flag)
                             ↓
           ML Inference Engine
           (gradient-boosted ensemble: ~2.1ms CPU inference)
                             ↓
           Spectrum Allocation Output
           (PRB assignment, power control recommendation)
                             ↓
           Closed-Loop Evaluation
           (spectral efficiency, SINR CDF, decision latency)
```
> **System bottleneck:** Feature engineering pipeline at ~0.8ms — not the inference. Profiling revealed rolling statistics recomputed from scratch on each window; moving to incremental update reduced feature compute by 3×.
 
**Tech Stack**
`Python` · `scikit-learn` · `XGBoost` · `Pandas` · `NumPy`
 
**Key Contributions**
- Engineered 23 temporal features from raw KPI streams; feature importance analysis confirmed lag-3 SINR and rolling PRB utilization as primary predictors — interpretable features traceable to RF interference physics
- Benchmarked 6 ML algorithm classes (linear, tree, ensemble, LSTM, MLP, SVM) against joint latency-accuracy objective; documented Pareto frontier identifying gradient boost as optimal for near-RT RIC constraints
- Discovered and resolved feature distribution shift under traffic load spikes by adding instantaneous-delta features alongside rolling statistics — critical for step-change responsiveness
- Profiled feature pipeline and reduced compute from 2.4ms to 0.8ms via incremental rolling statistic updates — necessary to preserve near-RT RIC latency margin
 
**Contextual Metrics**
- ~15–20% improvement in spectral efficiency over static assignment baseline on simulated 20MHz TDD scenario with 3-sector interference
- Gradient boost inference: ~2.1ms on standard x86 CPU — within 10ms near-RT RIC budget with 4.8× headroom for feature computation and protocol overhead
- Feature pipeline: ~0.8ms after incremental update optimization (reduced from 2.4ms baseline)
 
**⚖️ Engineering Trade-offs**
 
| Decision | Chosen | Rejected | Reasoning |
|---|---|---|---|
| Model class | Gradient boosted ensemble | LSTM / Transformer | LSTM achieves ~3% better sequence accuracy but at 12–15ms inference — violates near-RT RIC budget; gradient boost at 2.1ms provides required control loop headroom |
| Feature representation | Normalized relative metrics | Topology-specific absolute values | Absolute RSRP values are cell-layout-dependent; normalized relative metrics (SINR relative to cell average, PRB as fraction of capacity) transfer across sector geometries |
| Evaluation primary metric | Spectral efficiency + latency CDF | Accuracy alone | Accuracy without latency context is meaningless in a control-loop application; latency CDF reveals tail behavior that average latency hides |
| Feature engineering | Manual lag + rolling statistics | End-to-end learned features | Interpretability is operationally required in network management; hand-engineered features allow direct correlation to known RF phenomena and operator debugging |
 
**⚠️ Failure Modes & Learnings**
 
*Failure 1 — Feature lag during traffic load spikes:*
Rolling-average features lagged actual load changes by multiple time slots. During sudden traffic bursts, the model operated on stale context and made allocations optimized for the previous load state.
 
*Resolution:* Added instantaneous-delta features (current slot minus previous slot) alongside rolling statistics. Delta features respond immediately to step changes; rolling features capture steady-state context. Combined representation handled both operating regimes.
 
*Failure 2 — Topology-specific overfitting:*
Model trained on specific cell layout patterns generalized poorly to different sector geometries because features encoded absolute RF values dependent on site-specific propagation.
 
*Resolution:* Replaced absolute with normalized relative metrics. This is a physics-motivated normalization — interference experienced relative to cell average generalizes across layouts where absolute RSRP does not.
 
**→** [Repository](https://github.com/sachin-deshik-10/5G_AI_POWERED_ORAN)
 
---
 
### 📍 Real-Time GPS Telemetry System
 
**Problem Statement**
HTTP polling architectures introduce two structural problems for asset telemetry: per-message TCP connection overhead (200–800 bytes per request vs. ~2 bytes MQTT fixed header overhead), and polling intervals that create artificial latency floors regardless of update rate. For fleet tracking over NB-IoT or LTE-M cellular links with constrained bandwidth and intermittent connectivity, these are not minor inefficiencies — they determine system feasibility.
 
**Approach**
Designed and deployed an MQTT pub/sub telemetry architecture with explicit QoS level selection, hierarchical topic design, and payload serialization optimized for bandwidth-constrained cellular links.
 
**Architecture**
```
┌──────────────────────────────────────────────────────────┐
│  GPS Module (NMEA sentence stream, 1Hz–10Hz)            │
└────────────────────────┬─────────────────────────────────┘
                         ↓
         NMEA Parse + Fix Quality Gate
         (HDOP < 2.5 filter, fix status validation)
                         ↓
         Payload Serialization
         (42-byte binary fixed-format: lat/lon/ts/speed/hdop)
                         ↓
         MQTT Publish — QoS Level 1
         (at-least-once, persistent session, message TTL=30s)
                         ↓
         [BROKER: topic routing, retained message, TTL enforcement]
                         ↓
         Subscriber / Backend Ingestion
         (position store, geofence evaluation, stale message discard)
```
 
**Tech Stack**
`Python` · `Paho MQTT` · `NMEA parser` · `Mosquitto broker` · `IoT edge hardware`
 
**Key Contributions**
- Selected QoS Level 1 over QoS 2 after measuring that QoS 2's 4-message handshake doubled round-trip time on high-latency LTE-M links — application tolerates duplicate delivery, not delivery failure; QoS 1 is the correct choice
- Designed hierarchical topic structure (`fleet/{asset_id}/location`) enabling per-asset subscription isolation and broker-side routing without application logic
- Reduced payload from ~280-byte naive JSON to 42-byte binary fixed-format — 85% bandwidth reduction per message; critical for NB-IoT link budget
- Implemented NMEA fix quality gate (HDOP < 2.5) preventing publication of null-island coordinates (0.0, 0.0) during GPS acquisition phase — a silent data corruption issue
 
**Contextual Metrics**
- End-to-end location update latency: median 380ms, 95th percentile 820ms on LTE-M network under nominal load
- Payload: 42-byte binary vs. ~280-byte naive JSON — 85% reduction in per-message cellular data consumption
- Broker load-tested to 150 concurrent publisher connections on single Mosquitto instance without message queue degradation (baseline: 100 connection target)
 
**⚖️ Engineering Trade-offs**
 
| Decision | Chosen | Rejected | Reasoning |
|---|---|---|---|
| Protocol | MQTT | HTTP REST | MQTT persistent sessions eliminate per-message TCP setup; at 1–5s update intervals, HTTP connection overhead is the dominant cost |
| QoS level | QoS 1 (at-least-once) | QoS 2 (exactly-once) | QoS 2 handshake doubles round-trip time on high-latency cellular; duplicate position at same timestamp is harmless, delivery failure is not |
| Payload format | 42-byte binary | JSON | JSON readability serves development, not production; binary format is correct for bandwidth-constrained links |
| Message TTL | 30-second TTL | Unbounded queue | Stale position data from offline period is operationally worthless; bounded TTL prevents reconnection message storm from overwhelming backend |
 
**⚠️ Failure Modes & Learnings**
 
*Failure 1 — Reconnection message storm:*
After extended offline periods, the broker queued all accumulated messages and delivered them in burst on reconnection, overwhelming backend ingestion and producing a ~45-second latency spike in position updates.
 
*Resolution:* Message TTL enforced at 30 seconds broker-side; client-side buffer capped at 60 messages. Positions older than TTL are discarded. Historical gaps are operationally acceptable; delivery storms are not.
 
*Failure 2 — Null-island coordinate publication:*
GPS module streams NMEA sentences including invalid fixes during satellite acquisition. Without fix quality filtering, `0.0, 0.0` coordinates were published and stored, silently corrupting position history.
 
*Resolution:* HDOP < 2.5 and fix status validation gate added before publish. The error was silent — no exception, no visible failure — making it an example of why input validation at ingestion is non-negotiable.
 
---
 
## 📚 Research Publication
 
<table>
<tr>
<td>
 
**Advanced Data-Driven Analysis and Optimization of LoRa Networks: A Comprehensive Machine Learning Approach**
 
*N. S. Deshik, N. Kanthi, S. M*
 
**IEEE CSITSS 2025** — Bangalore, India
 
[DOI: 10.1109/CSITSS67709.2025.11295011](https://doi.org/10.1109/CSITSS67709.2025.11295011)
 
`LoRa` · `Machine Learning` · `IoT` · `Signal Processing` · `Network Optimization`
 
</td>
</tr>
</table>
 
**Technical Contribution:** Characterized LoRa network degradation modes using field measurement data and constructed ML models to predict optimal spreading factor and transmission power configurations under varying path loss and interference conditions. Key finding: tree-based models outperformed linear regressors because LoRa's link budget is a non-linear function of spreading factor — a physics-layer insight that drove the modeling choice.
 
**Research Gap Addressed:** Existing LoRa optimization literature relies on analytical path loss models assuming free-space propagation. Data-driven characterization captures empirical non-linearities (diffraction, near-far effects, gateway load dynamics) that analytical models cannot represent.
 
---
 
## ⚙️ Technical Expertise
 
**ML Systems Engineering**
Training pipeline architecture · Inference optimization (ONNX, quantization, batching) · Deployment-aware model design · Latency-accuracy trade-off benchmarking · Feature engineering for time-series and signal data
 
**Computer Vision & Sensing**
Small-object detection (YOLO family, anchor engineering) · Multi-object tracking (IoU association, Kalman filter) · Radar DSP (CFAR, pulse compression, range-Doppler) · Infrared/thermal processing · Track-before-detect strategies
 
**IoT & Edge Systems**
MQTT pub/sub architecture · QoS-aware delivery semantics · NMEA sensor data validation · Bandwidth-constrained payload design · Edge inference constraints
 
**Network & RF Systems**
O-RAN telemetry and near-RT RIC control loop constraints · LoRa PHY/MAC · Spectrum optimization under interference · Signal processing fundamentals (FFT, windowing, matched filtering, CFAR)
 
**Languages**
`Python` (primary — ML, DSP, systems) · `C` (embedded/performance-critical) · `JavaScript/TypeScript` (tooling, dashboards)
 
---
 
## 🚀 What Makes My Work Different
 
Most ML engineers optimize model accuracy. Most systems engineers optimize infrastructure. The problems worth solving live at the boundary between both — and that boundary requires fluency in both.
 
```
┌────────────────────────────────────────────────────────────────────┐
│  Layer              │  Common approach        │  My approach       │
├─────────────────────┼─────────────────────────┼────────────────────┤
│  Hardware / Sensors │  Assume clean input      │  Validate + filter │
│  Signal Processing  │  Use library, skip why   │  Understand physics│
│  ML / Inference     │  Maximize benchmark acc. │  Design for deploy │
│  Systems / Ops      │  "Works on my machine"   │  Measure + monitor │
└─────────────────────┴─────────────────────────┴────────────────────┘
```
 
**ECE grounding** means when a system degrades in deployment, I can distinguish between a bad algorithm, a misconfigured sensor, a pipeline data quality issue, and a distribution shift — and target the correct root cause.
 
**Documented trade-offs** — every architectural decision in these projects has a recorded rationale. The "chose X over Y because…" discipline prevents future engineers from silently undoing decisions whose reasons weren't preserved.
 
**Failure-mode documentation** — what broke, why, and how it was fixed is more transferable knowledge than what worked. Systems that haven't been stress-tested haven't been finished.
 
---
 
## 💡 What I Bring
 
**For AI/ML teams:** A practitioner who asks "what are the deployment constraints?" before "what architecture should I use?" — and who can own the full pipeline from signal ingestion to inference output.
 
**For systems teams:** Signal processing and sensor physics fluency. The ability to determine whether a problem requires a better algorithm, better data, or better pipeline design.
 
**For research groups:** Comfort operating at the theory-implementation boundary — translating published methods into working systems, and identifying where real-world conditions invalidate paper assumptions.
 
---
 
## 📊 GitHub Activity
 
<p align="center">
  <img height="165" src="https://github-readme-stats.vercel.app/api?username=sachin-deshik-10&show_icons=true&theme=tokyonight&hide_border=true&bg_color=0d1117&title_color=00D9FF&icon_color=7C3AED&text_color=c9d1d9&count_private=true" />
  <img height="165" src="https://github-readme-streak-stats.herokuapp.com/?user=sachin-deshik-10&theme=tokyonight&hide_border=true&background=0d1117&ring=00D9FF&fire=7C3AED&currStreakLabel=00D9FF" />
</p>
 
<p align="center">
  <img height="145" src="https://github-readme-stats.vercel.app/api/top-langs/?username=sachin-deshik-10&layout=compact&theme=tokyonight&hide_border=true&bg_color=0d1117&title_color=00D9FF&text_color=c9d1d9" />
</p>
 
[![Activity Graph](https://github-readme-activity-graph.vercel.app/graph?username=sachin-deshik-10&theme=tokyo-night&hide_border=true&bg_color=0d1117&color=00D9FF&line=7C3AED&point=FFFFFF)](https://github.com/sachin-deshik-10)
 
---
 
## 📬 Contact
 
| | |
|---|---|
| **Email** | nayakulasachindeshik@gmail.com |
| **LinkedIn** | [sachin-deshik-nayakula](https://www.linkedin.com/in/sachin-deshik-nayakula-62b93b362) |
| **GitHub** | [@sachin-deshik-10](https://github.com/sachin-deshik-10) |
 
---
 
## 🤝 Open To
 
**Research collaborations** in sensor fusion, edge AI, O-RAN intelligence, or real-time signal processing — particularly work where deployment constraints drive architecture choices.
 
**Internships** at organizations building where AI meets sensing, RF, or constrained deployment: defense tech, autonomous systems, telecom systems, or applied AI labs.
 
**Technical discussions** on architecture decisions, signal processing challenges, or ML deployment problems. I find direct technical exchange more useful than introductory calls.
 
---
 
<div align="center">
 
*The systems that matter most operate under constraints that benchmarks don't capture.*
*That's the space I work in.*
 
<br/>
 
<img src="https://komarev.com/ghpvc/?username=sachin-deshik-10&label=Profile+Views&color=00D9FF&style=flat-square" />
 
</div>
 
