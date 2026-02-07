# Kinematic-Constraint-Track-to-Track-Association-with-Incomplete-Radar-Measurements
TTTA Radar–ADS-B Association (Baseline DTW / KC-DTW)

This folder provides two Python files for track-to-track association (TTTA)
between heterogeneous radar measurements and ADS-B trajectories.

FILES

main.py

Experiment runner.

Iterates over selected scenes and error levels.

Loads radar / ADS-B CSV files using the configured filename patterns.

Calls the TTTA association pipeline and saves per-scene/per-level results.

ttta_adsb_radar_kcdtw.py

Core TTTA implementation.

Builds track sequences from CSV.

Computes DTW-based costs for:
(a) baseline DTW with statistically-consistent (Mahalanobis) cost
and dynamic dimension masking for incomplete radar measurements
(b) KC-DTW / KC-IMDTW style cost with additional kinematic constraints
(rate / difference features)

Solves global one-to-one assignment (e.g., Hungarian algorithm).

DATA AVAILABILITY NOTICE

Noise-injected (error-added) data are provided ONLY for:
Scene 1, Scene 3, Scene 7, Scene 9

Raw/original data are provided for:
Scene 1 to Scene 9

If you attempt to run noise/error experiments for scenes other than {1,3,7,9},
missing-file errors may occur.

BEFORE RUNNING (IMPORTANT)

Check and update paths in main.py:

Base data directory (where Scene folders / CSV files are located)

Output directory (must be writable)

Confirm filename patterns match your dataset.
Typical patterns used in this project include:

ADS-B:
scene{sid}ads-b{lv}.csv

Radar:
scene{sid}radar{lv}_miss_none.csv
scene{sid}radar{lv}miss_r.csv
(or other miss* variants)

NOTE: main.py may override the radar pattern. Ensure the pattern used in
main.py matches your actual filenames.

Ensure required dependencies are installed:

numpy

pandas

scipy
