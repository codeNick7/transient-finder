# System Approach & Methodology Document: Multi-Tier AI Heuristic Validation & False-Positive Analysis

**Document Reference:** `DOC-APP-2026-NEO-02`  
**Date:** November 2026  
**System Target:** Local Laptop Execution (macOS x86_64 / Apple Silicon, 16GB RAM)  
**Classification:** System Architecture & Methodology Specification  

---

## 1. System Philosophy & Objectives

The primary objective of the **Transient-Finder NEO AI System** is to provide an edge-capable, low-latency, and highly transparent validation engine for newly discovered Near-Earth Object candidates and time-domain astronomical transients.

### Core Tenets:
1. **Zero High-Risk False Negatives:** Planetary defense mandates that true Potentially Hazardous Asteroids (PHAs) must never be filtered out or discarded.
2. **Aggressive False-Positive Suppression:** Astronomers and robotic follow-up telescopes cannot waste observing time chasing subtraction artifacts, optical ghosts, or orbital debris.
3. **Physics-Grounded Explainability:** Unlike opaque deep neural networks, every validation verdict must be accompanied by explicit physical triggers, sub-score breakdowns, and diagnostic failure categorizations.
4. **Laptop-First Optimization:** The entire stack must run natively on consumer hardware (e.g., MacBook Pro Intel Core i7 / Apple Silicon, 16GB RAM) without requiring external GPU server clusters or cloud dependencies.

---

## 2. Multi-Tier Vetting Funnel

The system processes raw sky survey alerts through a 6-stage progressive filtration and diagnostic funnel:

```
[ Public Survey Data Ingestion ] (JPL Scout, MPC NEOCP, Fink Broker, JPL CAD)
                 │
                 ▼
[ Tier 1: Coordinate & Cadence Normalization ] (Julian Date, RA/Dec in deg, Disk Caching)
                 │
                 ▼
[ Tier 2: Astrometric Kinematics & Polynomial Fitting ] (RMS Residual, Velocity, Acceleration)
                 │
                 ▼
[ Tier 3: 5-Pillar Scientific Heuristic Engine ] (Astrometry, Orbit, Obs, Debris, Photometry)
                 │
                 ▼
[ Tier 4: Supervised ML & Unsupervised Anomaly Scoring ] (Gradient Boosting + Isolation Forest)
                 │
                 ▼
[ Tier 5: Root-Cause False-Positive Diagnostics ] (Dipole, Satellite, Ghost, Linkage, VarStar)
                 │
                 ▼
[ Tier 6: Longitudinal Re-Validation & Disposition ] (6-Month Drift, P(Recovery), Status)
```

---

## 3. Mathematical Formulation of the 5 Heuristic Pillars

The composite AI confidence score ($S_{composite} \in [0, 100]$) is computed as a weighted combination of five distinct physical evaluations:

$$S_{composite} = \sum_{k=1}^5 w_k \cdot S_k$$

Where the weights $\sum w_k = 1.0$ are calibrated based on empirical survey reliability:
- $w_1 = 0.30$ (Astrometric Consistency)
- $w_2 = 0.25$ (Orbital Feasibility)
- $w_3 = 0.20$ (Observational Reliability)
- $w_4 = 0.15$ (Satellite & Debris Veto)
- $w_5 = 0.10$ (Photometric Stability)

---

### Pillar 1: Astrometric Consistency Score ($S_1$)
Evaluates whether sequential observations conform to smooth celestial motion without erratic jitter:
1. **Astrometric Residual RMS ($\sigma_{RMS}$):**
   Given $N$ observations, residuals between observed coordinates and polynomial fits are calculated in arcseconds:
   $$\sigma_{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( \Delta \alpha_i \cos \delta_i \right)^2 + \left( \Delta \delta_i \right)^2}$$
   - If $\sigma_{RMS} \le 0.25''$: Ideal optical centroiding; penalty = $0$.
   - If $0.25'' < \sigma_{RMS} \le 0.80''$: Moderate jitter; penalty scales linearly.
   - If $\sigma_{RMS} > 0.80''$: Subtraction dipole or bad centroiding; severe penalty applied.
2. **Trajectory Linearity ($R^2$):**
   $$R^2 = 1.0 - \frac{\sum (\Delta \text{pos}_i)^2}{\max(1.0, \Delta \theta_{total}^2)}$$
   For authentic asteroid tracklets, $R^2 \ge 0.98$.
3. **Topocentric Acceleration Bound ($|\ddot{\theta}|$):**
   Natural solar system bodies exhibit smooth apparent acceleration dominated by Earth's reflex motion ($|\ddot{\theta}| < 0.3^\circ/\text{day}^2$). Accelerations $> 0.5^\circ/\text{day}^2$ indicate maneuvering spacecraft, thruster firings, or spurious tracklet linkage.

---

### Pillar 2: Orbital Feasibility Score ($S_2$)
Tests whether orbital elements conform to conservative celestial dynamics:
1. **Semi-Major Axis ($a$):**
   - Natural asteroids: $a \ge 0.5 \text{ AU}$.
   - Objects with $a < 0.1 \text{ AU}$ are flagged as terrestrial orbital debris.
2. **Eccentricity ($e$):**
   - Natural closed orbits: $0 \le e < 1.0$.
   - Hyperbolic trajectories ($e \ge 1.0$) are permitted only if designated as comets or flagged for interstellar anomaly review.
3. **Tisserand Parameter ($T_J$):**
   $$T_J = \frac{a_J}{a} + 2 \cos(i) \sqrt{\frac{a}{a_J}(1 - e^2)}$$
   Where $a_J = 5.2044 \text{ AU}$ (Jupiter's semi-major axis). Orbits with $T_J > 3.0$ receive positive scoring; unphysical or chaotic orbits with $T_J < 1.5$ are penalized.

---

### Pillar 3: Observational Reliability Score ($S_3$)
Rewards extensive observational coverage and penalizes poor observing geometry:
1. **Number of Observations ($N_{obs}$):**
   - $N_{obs} \ge 6$: Full credit ($+25$ points).
   - $N_{obs} = 3$: Baseline confirmation minimum ($+5$ points).
   - $N_{obs} < 3$: Insufficient tracklet length ($-20$ points).
2. **Observational Arc Length ($\tau$):**
   - $\tau \ge 12 \text{ hours}$: Strong orbit constraint ($+25$ points).
   - $\tau < 0.5 \text{ hours}$: Ephemeris degeneration risk ($-15$ points).
3. **Solar Elongation ($\epsilon$):**
   $$\cos \epsilon = \sin \delta \sin \delta_\odot + \cos \delta \cos \delta_\odot \cos(\alpha - \alpha_\odot)$$
   If $\epsilon < 55^\circ$, the target lies within the twilight sky background. Stray light, internal telescope reflections, and bright star diffraction spikes cause heavy false alarm rates.

---

### Pillar 4: Satellite & Space Debris Veto ($S_4$)
Distinguishes natural Near-Earth Objects from artificial satellites:
1. **Maximum Natural Motion Speed:**
   The highest geocentric orbital velocity for a natural asteroid during an ultra-close flyby is bounded by Earth's escape velocity ($\sim 11.2 \text{ km/s}$) plus heliocentric relative speed, resulting in apparent angular rates $\dot{\theta} \le 25\text{–}30^\circ/\text{day}$.
   - Motion rates $\dot{\theta} > 30^\circ/\text{day}$ are virtually guaranteed to be Low Earth Orbit (LEO) debris or operational satellites (e.g. Starlink, OneWeb).
2. **Orbital Inclination Clustering:**
   Megaconstellations cluster tightly around specific inclination bands (e.g., $i \approx 53.0^\circ \pm 3^\circ$, $70^\circ$, or $97^\circ$ Sun-synchronous). Objects exhibiting high apparent rates at these inclinations are immediately vetoed.
3. **Stationary Veto:**
   Detections with $\dot{\theta} < 0.04^\circ/\text{day}$ are non-moving, vetoed as stationary variable stars or bad pixels.

---

### Pillar 5: Photometric Stability Score ($S_5$)
1. Optical magnitude consistency across filters ($16.0 \le V \le 22.5$).
2. Absolute magnitude ($H$) consistency: $H > 30.0$ corresponds to physical diameters smaller than $2 \text{ meters}$, characteristic of discarded satellite shrouds, solar panels, and spent rocket stages rather than detectable asteroids.

---

## 4. Supervised Machine Learning & Anomaly Detection

### Real-Bogus Gradient Boosting Classifier
- **Model:** 100-tree Gradient Boosting Classifier ($lr=0.08, depth=4$).
- **Features:** 20 engineered kinematic, astrometric, and photometric indicators (e.g. $\log_{10}(\dot{\theta})$, $\sigma_{RMS}$, $R^2$, $\ddot{\theta}$, $a, e, i, T_J, \text{MOID}$, $\epsilon$).
- **Output:** Calibrated probability $P(Real) \in [0.0, 1.0]$.
- **Execution Cost:** $< 2 \text{ milliseconds}$ per candidate on laptop CPU.

### Unsupervised Isolation Forest (Interstellar & Rare Event Detection)
- **Model:** Isolation Forest ($contamination=0.05$).
- **Objective:** Detects outlier discoveries without relying on labeled training data.
- **Triggers:** Flags interstellar interlopers ($e > 1.2$, like 1I/'Oumuamua and 2I/Borisov), retrograde NEOs ($i > 90^\circ$), and ultra-close Earth grazers ($\text{MOID} < 0.005 \text{ AU}$).

---

## 5. Root-Cause False-Positive Diagnostic Taxonomy

When an alert is flagged as suspicious, the system executes an automated diagnostic decision tree to classify the exact failure mechanism:

```
                                  [ Candidate Vetting ]
                                            │
                     ┌──────────────────────┴──────────────────────┐
            [ Is Speed > 30°/day                    [ Is Speed < 0.04°/day? ]
             or LEO Inclination Band? ]                            │
                     │                                             ▼
                     ▼                             [ STATIONARY_VARIABLE_STAR ]
        [ ARTIFICIAL_SATELLITE_OR_DEBRIS ]
                     │
                     ├──────────────────────┬──────────────────────┐
                     │                      │                      │
                     ▼                      ▼                      ▼
         [ Is Accel > 0.5°/d²    [ Is RMS > 0.85''? ]    [ Is Elongation < 55°? ]
          or Linearity R² < 0.93? ]         │                      │
                     │                      ▼                      ▼
                     ▼             [ DIPOLE_SUBTRACTION   [ BRIGHT_STAR_DIFFRACTION
        [ SPURIOUS_NOISE_LINKAGE ]     ARTIFACT ]                 GHOST ]
```

1. **`DIPOLE_SUBTRACTION_ARTIFACT`:** Asymmetric PSF resulting from sub-pixel alignment errors between science and reference images.
2. **`ARTIFICIAL_SATELLITE_OR_DEBRIS`:** Fast-moving non-Keplerian orbital objects with terrestrial perigee.
3. **`BRIGHT_STAR_DIFFRACTION_GHOST`:** Low solar elongation twilight alerts contaminated by stellar halos and optical diffraction spikes.
4. **`SPURIOUS_NOISE_LINKAGE`:** Accidental spatial alignment of random background noise fluctuations mimicking a tracklet.
5. **`STATIONARY_VARIABLE_STAR`:** High photometric variability with negligible angular displacement.
6. **`NONE_REAL_CANDIDATE`:** High-confidence astronomical object satisfying all kinematic and physical tests.

---

## 6. Longitudinal Re-Validation Methodology (6-Month Horizon)

For candidates discovered weeks or months prior (e.g. May 2026 objects):
1. **Ephemeris Covariance Expansion:**
   $$\sigma_{\Delta \theta}(t) = \frac{\sigma_0}{\sqrt{N_{obs}}} \left( \frac{t_{elapsed}}{\tau_{arc}} \right)^\gamma$$
2. **Recovery Probability ($P_{recovery}$):**
   Calculated by comparing the predicted 6-month error circle against standard survey search fields of view ($FOV \sim 1.5\text{–}3.0^\circ$):
   - $\sigma_{\Delta \theta} < 60'' \implies P_{recovery} = 0.99$
   - $\sigma_{\Delta \theta} < 600'' \implies P_{recovery} = 0.85$
   - $\sigma_{\Delta \theta} < 3600'' \implies P_{recovery} = 0.50$
   - $\sigma_{\Delta \theta} \ge 18000'' \implies P_{recovery} < 0.05$
3. **Final Disposition Categorization:**
   - `CONFIRMED_PERMANENT`: Multi-station tracking confirmed the orbit; designated into the MPC catalog.
   - `DESIGNATED`: Well-constrained orbit; high probability of recovery in subsequent sky sweeps.
   - `LOST_SHORT_ARC`: Valid physical object, but observational arc was too brief to prevent positional loss.
   - `REJECTED_SPURIOUS`: Conclusively refuted by follow-up imaging or identified as an image artifact.
