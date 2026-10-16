# Archive

## as_run/

These are the scripts used during the first phase of the project, when the pointer head was explored on the frozen backbone (cache reuse, bf16 versus fp32, normalisation, learning-rate schedules, data scaling and so on). "As run" means they are kept exactly as they were executed in the session, not cleaned up or generalised. [as_run/ALL_SCRIPTS.md](as_run/ALL_SCRIPTS.md) is the index: for each script it says what it was for, which section of [RESULTS.md](../../results/RESULTS.md) it produced, and whether it still runs.

They are in an archive, and not next to the maintained code, for three reasons:

- They are a record, not a tool. The numbers in RESULTS.md came from them, so they are kept to make those numbers traceable, but they contain hard-coded paths and a few import a hand-written LoRA module that was replaced by `peft`.
- The maintained code does the same jobs better. Training and evaluation today go through the `jev` package and its command line, and the LoRA and vision studies go through [../engine.py](../engine.py) and [../suite.sh](../suite.sh).
- Keeping them separate stops a reader from running a stale script by mistake and from mistaking one-off exploration code for the supported workflow.

Do not edit these files to fix them. If one of the experiments is worth repeating, redo it on top of `engine.py`.
