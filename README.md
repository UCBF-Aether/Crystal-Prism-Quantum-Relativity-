# Crystal Prism Quantum Relativity (CPQR)

The vacuum is a quantum-coherent FCC supersolid plasma crystal. Galaxies do
not sit in dark-matter halos. They sit in a vortex of that crystal, and the
vortex is pushed, not pulled.

This repository is the updated theory as of 2026-11-18: the SPARC-175
fair-run scripts, the full solved ledger, and the open-problems ledger.

## The SPARC-175 fair run

Same 175 galaxies, same mask (Vo > 1, Verr > 0, n >= 4), same error model,
same mass-to-light ratios (0.5 disk / 0.7 bulge), same scoring for every
model. No cherry picking. Mean absolute residual, lower is better:

| Model | Knobs | Mean |resid| |
|---|---|---|
| **CPQR 2-knob** | 2 | **6.03%** |
| NFW halo 2-knob | 2 | 8.89% |
| MOND 1-knob | 1 | 12.47% |
| CPQR 1-knob tilt | 1 | 12.99% |
| CPQR 1-knob plain | 1 | 15.60% |
| **CPQR frozen** | 0 | **20.60%** |
| MOND frozen | 0 | 23.38% |

CPQR wins at 2 knobs and at 0 knobs. MOND takes the 1-knob round,
12.47% to 12.99%. Fair is fair.

Galaxy by galaxy, CPQR beats the NFW halo on **146 of 175** galaxies.
At every radius bin, CPQR wins. In the inner galaxy the halo is worst,
30.03% error to CPQR's 20.61%. Out past the edge, MOND goes flat near
8% no matter what; CPQR keeps improving, down to 2.62%.

Mean accuracy: CPQR 93.97%, NFW 91.11%, MOND 87.53%.

## What is predicted, what is fitted

The vortex law: `V_vortex = alpha * Vflat * (1 - exp(-R / r_scale))`,
`V_pred = sqrt(Vbar^2 + V_vortex^2)`.

- **Swirl strength A_t = 0.66 x Vflat** — predicted, dead linear across all
  175 galaxies, no fitting.
- **Transition radius r_t = Vflat^2 / a0_eff** — computed from the baryons,
  not fitted.
- **The tilt** — a stronger swirl means an earlier turn-on (Spearman -0.634).
- **Bulge overshoot** — inner bulge share predicts the overshoot
  fraction (+0.582).
- **Age** — older/redder galaxies turn on earlier; high-redshift galaxies
  should turn on later (untested).

The **one number left is the turn-on, u = r_scale / r_t**, still fitted per
galaxy. Everything else is predicted, computed, or measured input. Find the
turn-on and the zero-knob line drops toward 6%. That is the hunt now.

## Run it yourself

```bash
python3 sparc175_fair_run_2026-11-18.py            # numbers + 7 real graphs
python3 sparc175_fair_run_numbers_2026-11-18.py    # numbers only, no files
```

Each script is self-contained: it downloads the SPARC data from Zenodo
itself, runs all 175 galaxies through every model, and prints the full
scoreboard, the per-galaxy table, the residual-vs-radius table, the
reproduction gates, the predictions block, and the verdict. About 20 seconds.

`sparc175_post_script_2026-11-17.py` is the earlier single-script version
Cory ran on his phone (175/175 x 3 models, 5/5 win).

## The ledger

- `CPQR-SOLVED.md` — the solved index: every graded result, rows through
  513, with the script that produced it and the date.
- `OPEN_PROBLEMS.md` — the grind ledger: what is still open, including the
  turn-on hunt and every failed route, so nothing gets re-run blind.

## Honest limits

- The frozen law is down to one unknown per galaxy: the turn-on. The full
  curve for a new galaxy is not predictable yet.
- Even fitted, CPQR is 6.03%, not perfect: inner points, overshoot points,
  and three problem galaxies are still off.
- The lattice itself (density, shear modulus, spacing) is a property
  statement of the field, not a lab measurement. No lab has seen the
  lattice; the Casimir bridge inside the lattice is dead.
- Cosmology: 5 inputs give back 4/4 Planck numbers. That sector is
  complete.

## Author

Cory Stephen Brent. Theory derived live from a single axiom; computed
end to end, 1,100+ scripts. The vacuum is a crystal. The push, not the
pull.
