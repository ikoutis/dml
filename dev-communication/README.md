# dev-communication

A dated, async communication channel for the DML (matched mutual learning) project —
the same mechanism as the knowledge-diffusion repo's `dev-communication/`: a place to
hand off tasks to collaborators and record what came back, so the thread survives
across sessions and machines.

## How it works

- **[`log.md`](log.md)** is the running log. It is a single rolling document of dated
  entries, **newest at the top** (reverse-chronological).
- Every entry is headed `## YYYY-MM-DD [HH:MM TZ] — <kind>: <short title>`; the time is
  optional (add it when ordering within a day matters).
- A **task** (`Task:`) says who it is for, why, exactly what to run, and what to report back.
- A **note** (`Note:`) is informational — e.g. pointing at documents now in the repo — and
  asks nothing of the reader.
- A **reply** (`<name>:`) is added at the top as a new entry, citing the task it answers
  (paste results, numbers, errors, questions).
- Keep entries self-contained: link to the exact code/branch/PR, and to any scripts a
  reader needs, by repo-relative path so they resolve on whatever machine you clone to.
- **Write entries — findings above all — in plain, uncompressed prose.** Spell out what
  was expected, what was observed, and what it means, in complete sentences a reader can
  follow on the first pass. Tables carry status and numbers; the *meaning* belongs in the
  prose around them.

## Communication IDs

Task entries carry an incrementing ID in the header — `[D-001]`, `[D-002]`, … (the `D`
prefix keeps them distinct from the knowledge-diffusion repo's `[C-00N]` series, since
the two logs cross-reference each other). **Every experiment batch launched because of
an entry embeds that entry's ID** in two places:

1. the output directory (e.g. `results/suite/…` for the initial suite; later batches
   use `results/campaign/d00N_<experiment>/`), and
2. the run itself: scripts pass `--run_tag d00N`, which suffixes the run_id and adds a
   `run_tag` column to every CSV row.

So the linkage works in both directions: reading a log entry, you can find every run it
ordered; holding any CSV row, the `run_tag` column names the instruction that created
it. Replies cite the ID they answer. When a later entry re-launches an existing script,
it first updates the script's `OUT` prefix and `--run_tag` to its own ID.

## Index

| Document | What |
|---|---|
| [`log.md`](log.md) | The running, dated task/reply log. Start here. |
| [`experiments.md`](experiments.md) | The full experiment-suite design: arms, protocol, predictions H1–H6, experiments R1 + M1–M7, costs, and the mapping onto the original DML paper. |
