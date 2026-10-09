# Do the simulator's hand-coded mechanisms appear in real AIS data?

Real: 1639 port stays by 717 cargo/tanker vessels at 26 ports, 2026-09-13 to 2026-10-06 (5 km zones, entry and exit observed, 1-240 h). Simulator: 1565 shipments. Levels are NOT comparable (vessel stay vs cargo availability; short-sea vs deep-sea); only direction and rough size of effects are.

| Mechanism assumed in the simulator | Real AIS | Simulator | Verdict |
|---|---|---|---|
| Congestion lengthens the port stage (within port) | Spearman -0.03 (95% CI -0.08 to +0.03); stays 0.92x as long when occupancy is above the port's average (CI 0.84-1.02) | Spearman +0.55; 1.43x | NOT detected in real data |
| Port-stage duration is right-skewed (heavy tail) | median 10.1, IQR 2.5-21.2, p90 37.7, mean/median 1.71 | median 48.9, IQR 37.2-72.3, p90 104.8, mean/median 1.22 | supported; real tail is heavier than simulated |
| Calendar effect: stays starting Fri-Sun are longer | 1.07x vs Mon-Thu (CI 0.96-1.18) | modelled at customs (office hours), not at the port stage | not detected (CI includes no effect) |
| Arrival lateness is asymmetric with a late tail | vs ETA declared 12-24 h ahead: median -0.6 h, p90 +6.6 h, share later than +6 h 0.11, share earlier than -6 h 0.03 | vs schedule: median +8.1 h, p90 +26.2 h, share later than +6 h 0.62, share earlier than -6 h 0.00 | supported (late tail); simulated lateness is much larger but uses a fixed schedule, not an updated ETA |

Reading: occupancy is a crude congestion proxy and these ports are small and rarely full, so the absence of a congestion effect
here does not show that congestion never matters; it shows the simulator's congestion mechanism is NOT externally validated.
No simulator parameter was changed on the basis of these numbers.
