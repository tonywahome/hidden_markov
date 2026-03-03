## 6. Analysis and Reflection

### 6.1 Activity Distinguishability

The centroid distances in PCA space reveal a clear hierarchy of how separable each pair of activities is:

| Pair               | Centroid Distance | Confusion          |
| ------------------ | ----------------- | ------------------ |
| Walking ↔ Jumping  | 7.26              | Low — easiest pair |
| Jumping ↔ Still    | 6.01              | Moderate           |
| Standing ↔ Jumping | 4.80              | Moderate           |
| Standing ↔ Walking | 3.14              | Harder             |
| Standing ↔ Still   | 2.54              | Hard               |
| Walking ↔ Still    | 2.01              | Hardest pair       |

Jumping was the easiest to isolate. State 2 (jumping) achieved 100% specificity — no non-jumping window was predicted as jumping. Its PCA centroid sits 6–7 units from every other activity. This is expected: jumping produces unmistakable high-amplitude, high-frequency acceleration spikes that dominate the spectral energy features (the top PCA loadings are acc/gyro standard deviation and FFT magnitude). However, its sensitivity was only 30.8% because its intra-class spread (mean-std = 2.45) is the largest of all activities — jumping contains three mechanically distinct sub-phases (push-off, airborne, landing) that look very different within the same 2-second window, scattering some windows toward the still and standing states in the confusion matrix (7 jumping windows were predicted as still).

Still and standing were the hardest pair to separate, with a centroid distance of only 2.54 — the closest of all pairs. Both are low-motion states with small variance; their feature distributions overlap heavily. The confusion matrix shows this symmetrically: 9 of 13 still windows were predicted as standing, and 2 of 15 standing windows were predicted as still. The intra-class spread of still (mean-std = 1.98) is also high, likely because "still" encompasses both device-on-table and device-held-still conditions, which produce different gravitational projection signatures across axes.

Walking and still had the smallest centroid distance overall (2.01). Despite this, walking achieved 43.75% sensitivity because its intra-class spread is the lowest of all activities (mean-std = 0.59) — the rhythmic stride cycle is highly regular, creating a compact, well-defined cluster. Walking is predominantly confused with still (9/16 windows), which happens at the boundary of short recordings where the subject is starting or stopping.

### 6.2 Transition Probabilities and Behavioural Realism

The learned transition matrix (states ordered as standing, still, jumping, walking):

| From / To    | → Standing | → Still | → Jumping | → Walking |
| ------------ | ---------- | ------- | --------- | --------- |
| **Standing** | 0.9523     | 0.000   | 0.000     | 0.0477    |
| **Still**    | 0.000      | 0.992   | 0.000     | 0.008     |
| **Jumping**  | 0.0161     | 0.000   | 0.9839    | 0.000     |
| **Walking**  | 0.000      | 0.1225  | 0.000     | 0.8775    |

**Three observations:**

1. **Strong diagonal (self-transitions dominate).** All four states have self-transition probabilities above 0.87. This is exactly what the data dictates — every recording contains a single sustained activity, so within-recording transitions are almost always self-transitions. The model has correctly learnt that activities are persistent: once you are jumping, you are almost certainly still jumping one window (1 second of step) later.

2. **Asymmetric cross-transitions reflect physics.** The model learnt two meaningful off-diagonal transitions: standing → walking (4.77%) and walking → still (12.25%), but not walking → standing or still → walking. This matches real human movement: you stop walking by becoming still, and you start moving from a standing posture. The 12.25% walking-to-still transition is elevated because the 50% overlap between consecutive windows means a walking recording ending blends the final window's features toward the still cluster. The total absence of any jumping cross-transition except jumping → standing (1.61%) also makes physical sense — you return to standing after a jump, never directly to walking or still.

3. **Limitation: training data imposed a single-activity structure.** Because no recording ever contains a genuine activity transition (every zip is one activity), the transition matrix is learning boundary artefacts rather than true behavioural transitions. A meaningful transition matrix would require recordings that sequence activities, e.g., stand → walk → jump → stand.

### 6.3 Effect of Sensor Noise and Sampling Rate

**Sampling rate mismatch.** Antony's Samsung SM-A566B recorded at an actual interval of 16.05 ms (~62.3 Hz), while Rob's iPhone 15 Pro recorded at 9.96 ms (~100.4 Hz). To produce a common 100 Hz grid, Antony's signal was linearly interpolated, effectively doubling the sample density. This has two consequences:

- **FFT features are unreliable above ~31 Hz for Antony.** The Nyquist frequency of the original 62 Hz signal is ~31 Hz. The three FFT magnitude features (fft_1, fft_2, fft_3) and dom_freq above 31 Hz reflect interpolated, not measured, signal. PCA component 1 heavily loads on gyro_z_fft_3 and acc_x/y_fft_3, which carry artificially smooth high-frequency content for Antony's windows. This cross-subject inconsistency inflates intra-class variance and makes the still/standing boundary harder to learn.

- **Gyroscope resampling amplifies gyro-noise.** Linear interpolation between gyroscope samples creates artificially flat segments (consecutive interpolated rows share identical gyro values, as visible in the processed CSV). The gyro_sma feature — the top-loading feature on PC1 — will be systematically underestimated for Antony relative to Rob.

**Practical impact.** The per-subject recording breakdown shows Antony contributed 65 jumping windows vs. Rob's 39. Because Antony's jumping windows carry lower effective spectral energy above 31 Hz, the jumping cluster in PCA space is elongated along the PC1 axis, which directly explains the high intra-class spread (2.45) and the 30.8% sensitivity.

### 6.4 Potential Improvements

#### Data Collection

- **Longer mixed-activity recordings** are the highest-impact change. A 2-minute recording cycling through all four activities gives the HMM real transition sequences to learn from, turning it from an expensive nearest-centroid classifier into a genuine temporal model.
- **Match sampling rates at source** by configuring both devices to exactly 10 ms before recording, eliminating the need for upsampling and the spectral artefacts it introduces.
- **More recordings per subject** — 50 recordings of ~10 seconds each gives only ~500 seconds of data total. A minimum of 5–10 minutes per activity per subject is standard for HAR.

#### Feature Engineering

- **Jerk features** (first difference of acceleration) directly capture the explosive acceleration change that distinguishes jumping from all other activities, compressing sub-phase variance within jumping.
- **Resultant magnitude** `sqrt(x²+y²+z²)` is orientation-invariant, removing sensitivity to how the phone is held in the pocket — a major source of intra-class spread for standing and still.
- **Gravity separation** via a low-pass filter (~0.3 Hz) isolates the gravitational component from dynamic acceleration, providing tilt angle as an explicit feature that distinguishes pocket-portrait vs. pocket-landscape orientations.

#### Model Architecture

- **High self-transition prior** (`diag(A)` ≈ 0.9, off-diagonal = 0.1/(n−1)) prevents Baum-Welch from learning spurious transitions from boundary artefacts when recordings are single-activity.
- **Mixture emissions (GMM-HMM)** would model the multi-modal distribution within jumping (push-off, airborne, landing) using 2–3 Gaussians per state instead of one, directly addressing the sensitivity gap.
- **Additional sensors** — a barometer would unambiguously detect the brief airborne phase of a jump (pressure drop), and a magnetometer would resolve orientation ambiguity between standing and still. Both sensors are present on both devices used in this study.
