# Stale software goldens: prism_domain and prism_heat

Waiting for the owner's OK (CLAUDE.md: goldens are not changed without it).
Nothing under `tests/golden/` was touched.

## Verdict

The two software goldens are stale, not the renderer broken. They were made by the commits
that introduced the scenes (`4063f54` Beauty 1, `53da0fe` Beauty 2) and never regenerated
when two later, deliberate look changes landed. The GPU goldens were regenerated at both of
those commits; the software ones were not. Tier B (llvmpipe / NumPy kernels) renders the
current look correctly.

Recommendation: **regenerate the two software goldens** (patch here, or
`PHASOR_UPDATE_GOLDEN=1 uv run pytest tests/test_beauty.py -k "prism_domain_golden or prism_heat_golden"`
on a no-CUDA machine), after the owner has looked at the `after` pictures. No code fix,
no threshold change: 0.98 is right and the rest of the suite relies on it.

## Evidence

1. Pictures (this folder). `*_stored_current_diff.png` has three rows: stored software
   golden, current render, |difference| x4.
   - prism_heat (SSIM 0.798): the golden is the old two-tone look (orange and cyan plates,
     white-hot drum, a white line). The current render colours by spectrum hue (each isoline
     has the hue of its level, a rainbow trace along the bar, fading panel edges). A different
     look, not noise. Mean brightness 38.1 (golden) vs 24.0 (now): it is also darker, as
     `ccffe76` says ("brightness comes from strength alone, ambient 0: dark ground").
   - prism_domain (SSIM 0.958): same picture, same positions, size and equation. The golden
     has a softer wash and thinner, dimmer phase rays and modulus rings; the current render has
     brighter, more saturated lines (grid more prominent) and a lighter wash (mean 45.0 vs
     48.9). This is the `prism_field` look (`76ff8bb`: "brighter lines, 30% more light, shorter
     haze"). No banding, noise, size or position change.
2. History. `git log` for the two software PNGs shows only `4063f54` and `53da0fe`.
   `ccffe76` (owner: "doesn't look so prismy") rewrote the heat colouring and
   regenerated only `tests/golden/gpu/prism_heat.png`. `76ff8bb` added the `prism_field`
   look, switched `examples/prism_domain.py` to it, and added only
   `tests/golden/gpu/prism_domain.png`.
3. Bisect on this box (llvmpipe, software class). At `53da0fe` both tests pass, so the
   software goldens did reproduce here at their own commit and the platform is not the cause.
   `ccffe76` is the first commit where `prism_heat` fails; `76ff8bb` is the first where
   `prism_domain` fails (its parent `b4035e8` passes both).
4. Current software render against the GPU goldens: SSIM 0.9957 (domain) and 1.0000 (heat).
   So the new look is exactly what the owner approved on the PC, and software agrees with it.
   Against the stale software goldens: 0.958 and 0.798 (the 0.798 matches the CI figure).

## Not verified

- I did not run the tests on CI or on Windows/CUDA; the GPU goldens were not re-rendered
  here (no GPU), only compared. Whether they still pass on the RTX PC is untested by me.
- The fix is a data change, so a later visual change will need the same treatment.
  The process gap (look commits regenerate only the GPU golden) is not fixed here.
  Other software goldens may be stale in the same way; I checked only these two.
- The owner should look at `*_after_current_render.png` before approving; I judged them
  by eye and by SSIM to the GPU goldens only.

## For the owner

- Nothing is required on the Windows PC for this fix. Optionally run
  `uv run pytest tests/test_beauty.py -k "prism_domain_golden or prism_heat_golden"` there to
  confirm the GPU goldens still pass (they should, per the earlier commits; untested by me).
- To apply: `git apply pending/golden_prism_domain_heat/regenerate_software_goldens.patch`
  (binary patch of the two PNGs; `git apply --check` passes on this branch's base).
  Then delete this folder.
