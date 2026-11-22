# Where things live

| Kind | Where | Status |
|---|---|---|
| Design (binding) | `SPEC.md`, `CLAUDE.md` (root) | the seven contracts; changes need owner approval |
| Decisions | `docs/decisions/` (ADRs 0001-0004) | one file per decision, never rewritten |
| Milestone records | `docs/milestones/MILESTONE_0..13.md` | history: what was built, deferred, deviated |
| AI-facing API docs | `docs/ai/` (+ `llms.txt`) | every python block runs in CI |
| Living plans | `BEAUTY.md`, `ANIMATIONS.md`, `GPU_SPEED.md`, `APP_PLAN.md`, `TESTING.md` (root) | active; code comments cite them |
| Style rubric | `STYLE.md` (root) | read at runtime by the critic tools |
| PC-only tasks | `pc_tasks/` | instructions for the Windows/NVIDIA machine |
| Client kit source | `kit/`, `skills/phasor/` | copied into a person's folder by `phasor kit`; not for contributors |
| Demo scenes | `examples/`, `showcase/` | `showcase/render_all.py` renders the gallery |
| Benchmarks | `benchmarks/` | dated JSON from `phasor bench` |
