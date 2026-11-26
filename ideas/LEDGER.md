# Idea ledger

Written by the `idea-scout` skill; see `.claude/skills/idea-scout/SKILL.md`. Newest first.
The owner sets each card's status (`new`, `planned`, `made`, `dropped`) and records the
reason in `VERDICTS.md`.

## Topics to try

The owner's queue. One topic per line; the scout takes these first and removes each once run.

## Candidates

### walk_is_a_two_dimensional_paint_budget
2026-11-26 | status: new | converged (geometer + signal engineer) | topic: Polya's theorem (a random walk returns home in 1D and 2D, may escape in 3D)
Still: `ideas/sketches/walk_is_a_two_dimensional_paint_budget.png` (the plane's discs and the three brick piles; no line or space heat maps, no paint square)

- Claim: A random walk needs about R^2 steps to get R away from home in every dimension, so each time the radius doubles it has 4 times the steps to spend but 2, 4 or 8 times the room (line, plane, space). The visits home gained per doubling therefore double in 1D, stay the same for ever in 2D (settling at (2/pi) ln 2 = 0.441), and roughly halve in 3D, which sums to a finite total (1.516 visits counting the start, return probability about 0.34).
- Because: The expected steps to first leave the ball of radius R equal the expected squared exit distance (between R^2 and (R+1)^2) in any dimension, while the ball holds about 2R, pi R^2, (4/3) pi R^3 sites, so visits per site scale as R^(2-d). The link to home is exact: the visits home gained between first leaving radius R and first leaving radius 2R equal the expected visits, before leaving 2R, to the first site outside radius R where the walk crossed (visits from a to b equal visits from b to a).
- Frame: Three panels: a line, a plane, a cut-away space. Each shows the ball of radius 2R as a heat map of expected visits per site before the walk first leaves it, brightest at home and fading to zero at the edge. Beside each, the same square of side 2R (its area is the steps spent), identical in all three. The half-way shell at radius R is marked; its shade is lifted out as a brick and stacked at home. Then everything doubles (R = 2, 4, 8, 16): the square quadruples everywhere; the ball grows 2x, 4x, 8x. On the line the bricks double. On the plane the whole heat map just scales up, the ring keeps the same shade to within a few percent, and equal bricks stack without end. In space the shell grows paler, the bricks shrink, and the pile creeps to a ceiling at 1.516.
- Second beat (the signal engineer's card, merged): the same three piles from the frequency side. One step is an averaging filter; expected visits home are the average over frequency of the gain 1/(1 - H). Per octave toward DC the gain is 4x taller while the band is 2x, 4x, 8x smaller (interval, ring, shell). Exact octave integrals: doubling in 1D, 0.4413 in 2D, halving in 3D.
- Needs: area and volume scale as R^2 and R^3; a random walk after n steps is typically about sqrt(n) from its start.
- Attack: 1 True: pass (symmetry agrees to 5 decimals; 2D bricks 0.442, 0.429, 0.435, 0.437, 0.438, 0.440 for R = 2..64). 2 Picture: pass. 3 Fresh: pass. 4 Payoff: half-fail (the "a to b equals b to a" symmetry and, for the second beat, 1/(1 - H) are unlisted prerequisites). 5 Record: n/a.
- Attacker's conditions: say "the same shade to within a few percent, settling at 0.441", not "exactly"; take the brick from the exit-weighted ring; start the doublings at R = 2 (the first 3D step is x0.37, not a halving).
- Phasor gap: no object for a heat map on grid sites (the still uses one dot per site, 41 s to build 1,000 dots), and none for a cut-away 3D shell.

### sampling_is_a_slanted_projection
2026-11-26 | status: new | topic: sampling and aliasing | lens: linear algebraist
Still: `ideas/sketches/sampling_is_a_slanted_projection.png`

- Claim: Sampling then reconstructing is a projection onto the slow waves, but a slanted one: a fast wave at f0 + fs is not thrown away, its whole amplitude lands on the slow wave at f0 (a·slow + b·fast comes back as (a+b)·slow), whereas a low-pass filter is the perpendicular projection that would send it to zero.
- Because: The difference (fast - slow) is zero at every sample instant, so it spans the kernel of the sampling map, and all signals on a line parallel to it share the same dots. That direction is at 45 degrees to the slow axis, not perpendicular to it, so sliding along it to the slow axis carries the fast coordinate over in full.
- Frame: Left: a plane, "how much slow wave" across and "how much fast wave" up, with one glowing point (a, b). Right: the waveform of that mix with its sample dots. The point is dragged along the down-right diagonal: the curve writhes but the dots do not move. Dragged any other way, the dots move. The plane fills with these parallel diagonals and every point slides down its diagonal onto the slow axis; the pure fast wave at (0, 1) lands at (1, 0), full size. Last, from the same point a second, vertical arrow (the filter) drops to (a, 0): two arrows from one point, one slanted and one plumb; the gap between the landings is the alias.
- Needs: a signal as a weighted sum of two sine waves; a point in a plane as a pair of coordinates.
- Attack: 1 True: pass (a·slow + b·fast equals (a+b)·slow at every sample; slow and fast are orthogonal with equal norm, so 45 degrees and "perpendicular" are literal). 2 Picture: pass. 3 Fresh: pass (oblique projection is in the research sampling literature, not in textbook or video treatments of aliasing). 4 Payoff: pass. 5 Record: n/a. The only card clean on every question in the first round.

### subgroups_are_the_only_hiding_place
2026-11-26 | status: new | topic: the uncertainty principle of signals | lens: abstract algebraist / number theorist

- Claim: On a 12-point clock, three equal spikes at 0, 4, 8 have only 4 nonzero frequencies (3 x 4 = 12, the minimum the product can be, reached on a subgroup), but on a 13-point clock three equal spikes, wherever you put them, have all 13 frequencies nonzero.
- Because: The frequency-k coefficient is the three spike positions' unit arrows, turned k times faster, laid tip to tail; it is zero only if that chain closes, and three unit arrows close only as an equilateral triangle. On {0, 4, 8} every k gives either that triangle or three aligned arrows; on 13 points 120 degrees is never a multiple of 360/13, so no chain can close.
- Frame: Left: a clock of 12 dots with three lit spikes at 0, 4, 8. Right: 12 small tip-to-tail chains, one per frequency, each with a bright bar from its start to its end (the spectrum's height). Eight chains are closed triangles with no bar; four are straight lines of length 3. One spike is nudged from 8 to 9: every triangle springs open and all 12 bars appear at once. Then a 13th dot is squeezed into the clock; the spikes slide over all placements while the chains wriggle, and a faint equilateral target triangle behind each chain is never matched. No bar ever vanishes.
- Closing caption (cited, not shown): for unequal amplitudes, |supp f|·|supp f^| >= n (Donoho-Stark) and, for prime p, |supp f| + |supp f^| >= p + 1 (Tao 2005): at least 11 nonzero frequencies for any 3 spikes on 13 points.
- Needs: a frequency coefficient is a sum of rotating unit arrows (the winding picture, `fourier.Winding`); 13 is prime, 12 = 3 x 4.
- Attack: 1 True: pass after repair (4 nonzero for {0,4,8}, 12 for {0,4,9}, 13 for all 286 triples on 13; the first version claimed more than its mechanism showed). 2 Picture: pass. 3 Fresh: pass (no well-known visual treatment). 4 Payoff: pass. 5 Record: n/a.
- Related: the cut card `too_few_arrows_to_fill_the_glass` (see DEAD.md) gives the general bound whose equality case is this subgroup.

### uncertainty_is_an_area_not_a_box
2026-11-26 | status: new | topic: the uncertainty principle of signals | lens: geometer

- Claim: Duration x bandwidth is only the bounding box of a signal's time-frequency ellipse, and it can be made as large as you like by chirping; what chirping, dispersion, stretching and the Fourier transform cannot change is the ellipse's area.
- Because: Each of those operations acts on the (t, f) plane as a linear map of determinant 1 (chirp: vertical shear; dispersion: horizontal shear; time-stretch: a squeeze; Fourier transform: a quarter turn), so det of the covariance matrix stays fixed. A tilted ellipse of the same area needs a bigger bounding box.
- Quoted fact (a labelled tile on screen, not derived): the area is never below 1/4 second·hertz and is exactly 1/4 for every Gaussian, chirped or not.
- Frame: First the ellipse is built from the two shadows: the pulse on a horizontal axis with its width bar, its spectrum on a vertical axis with its width bar; the bars sweep out a rectangle and an ellipse grows inside it touching all four sides. A thin line through the centre shows how the pulse's pitch drifts with time; the ellipse touches the left and right sides exactly where that line crosses them. The ellipse fills with a fine grid of cells. Then: (1) chirp: the ellipse and its grid shear vertically, the time shadow does not move, the frequency shadow widens, the box swells, the same cells are inside; (2) dispersion: a horizontal shear, the pulse spreads; (3) the opposite shear: the pulse snaps short again (pulse compression is un-shearing); (4) a quarter turn: the shadows swap (the Fourier transform). The box breathes while the cell count never changes.
- Needs: a signal has a spectrum whose width is its bandwidth; a shear keeps area.
- Attack: 1 True: pass after repair (for the chirped Gaussian the product goes 0.0796 to 0.2516 at c = 3 while the area stays 0.2500; the tangency is exact). 2 Picture: pass. 3 Fresh: narrow pass (textbook in time-frequency analysis and optics, absent from the well-known videos). 4 Payoff: pass after repair. 5 Record: n/a.
- Attacker's condition: for a non-Gaussian pulse call the line the best-fit line of the pitch drift, not the pitch itself.

### whole_twists_between_nails
2026-11-26 | status: new | converged (topologist + abstract algebraist) | topic: sampling and aliasing

- Claim: Samples pin a rotating phase only at the sample instants, so they cannot count whole turns between two instants: two rotating signals agree at every sample exactly when their frequencies differ by a whole multiple of the sample rate, and among all these aliases at most one turns less than half a turn per gap (two mirror twins tie only at exactly half a turn).
- Because: Draw the signal as a thread on a cylinder (time along the axis, phase around it). Twisting the cylinder so that the ring at sample n turns n·k whole turns puts every ring back where it was but adds k turns of thread in every gap. A twist that is not whole leaves ring 1 off its place, so nothing else aliases.
- Frame: A translucent cylinder along the time axis; thin rings at the sample instants; a glowing thread winding a quarter turn per gap (1 Hz sampled at 4 Hz); a nail where the thread crosses each ring, with a ghost of each nail fixed in space; on the back wall the thread's shadow, a slow wave. (1) The cylinder is wrung like a towel; the nails leave their ghosts. (2) Pause at half a twist: the even nails are back on their ghosts, the odd ones sit opposite (this 3 Hz thread would fool a sampler half as fast, not this one). (3) At one whole twist every nail lands at once, but the thread makes 1¼ turns per gap and the shadow is a fast wave (5 Hz) through the same dots. (4) One whole twist the other way lands them again (-3 Hz, left-handed). (5) All landed threads together; the shortest lights up. (6) At exactly half a turn per gap a right-handed and a left-handed thread tie: the Nyquist edge.
- From the merged algebraist's card (`aliases_are_polygon_corners_that_reconvene`): in discrete time, keeping every m-th tick makes exactly m pointer speeds indistinguishable, the corners of a regular m-gon of per-tick steps; between kept ticks the m dots form a polygon that collapses to one point every m-th tick. A possible extra beat; the attacker judged the cylinder the better frame.
- Needs: a rotating point traces a sine wave as its shadow; sampling means looking only at evenly spaced instants.
- Attack: 1 True: pass after repair (the first version had the half-twist beat wrong). 2 Picture: pass. 3 Fresh: pass (the standard picture is two sinusoids through the same dots; the twist shows why only whole multiples work). 4 Payoff: pass. 5 Record: n/a. Ranked last of the five: correct and visual, but the fact it explains is close to the definition of aliasing.
- Phasor gap: no object for a curve on a cylinder that is being twisted.
