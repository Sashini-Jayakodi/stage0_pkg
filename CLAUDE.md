# FYP: Certify the Sign, Not the Frame (POC-2, plan Revision 8)

Track-level certified commit rules and certified readable size for traffic sign recognition (GTSRB core, BelgiumTSC
for validity only). Target venue: IEEE Access. This repo (`stage0_pkg`) holds POC-1 (finished) and POC-2 (in progress).

@docs/HANDOFF.md
@docs/poc2-implementation-plan-r8.md

## Environment
- Windows laptop, PowerShell. Python is `py -3.12` (NOT 3.11: Windows Application Control blocks torch DLLs there).
- GPU: RTX 5050 8 GB, torch 2.14.1+cu132. 16 GB RAM: use mmap for the image cache.
- Tests: `py -3.12 -m pytest tests -q`. Long suites: run in the background and poll.
- Git: POC-1 is tagged `poc1-final`; POC-2 work happens on branch `poc2`.

## Hard rules (never break)
1. Do not edit any POC-1 file (`src/*.py` without the `r8_` prefix, `configs/poc.yaml`, `configs/frozen_g2.yaml`,
   anything in `runs/20261003_stage1/` outside `poc2/`). POC-2 code goes in new files: `src/r8_*.py`, `tests/test_r8_*.py`.
2. Every script checks the SHA-256 of the files it reads against the summary JSON that wrote them, and stops if they differ.
3. Plan section 2 (design, then freeze, then confirm) is the integrity core:
   - Stages 1 and 2 may read outcomes of the 261 design tracks only (seed-0 Fit split, `track_split == 0` in `splits/seed0.npz`).
   - No commit-rule error, coverage or outcome number on the 1,046 pool tracks may be computed, saved or printed before
     `configs/frozen_r8.yaml` is committed (gate F1). Add a code assertion for this in every design-phase stage.
   - After F1, nothing frozen changes. Failed checks are reported as failed.
   - delta = 0.1 and beta = 5% are fixed in every phase.
4. Losses are computed per lambda (tracks-by-lambdas table), never from one score per track (risk R3).
5. Work one stage at a time. A stage is done only when its Checks pass. After each run, write a `LOG.md` entry in the
   section 13 template format (append only, never rewrite old entries).
6. Use confidence tags in explanations: [Certain], [Likely], [Guessing].

## What must never appear in project docs, LOG entries or the paper
- Lines like "Supervisor told?" or similar.
- Supervisors or module assessors listed as stakeholders.
- Research questions that start with "can" or have yes/no answers (they must be WH questions).
- Any mention of the project's time limit.

## Code style
Short comments that say WHY each part exists. Synthetic-data pytest tests for every stage, with brute-force reference
implementations, plus tamper tests (changed input files must be refused). Each stage also gets a
`README_r8_stageN.md` and a `LOG_r8_stageN_entry.md` with blanks for the user to fill from the printout.
