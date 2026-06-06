# v3.x training: live monitoring

Runbook for watching the northern Sierra placer v3.x training pipeline
while it runs. All commands assume the repo root as the working directory.

## One-shot status

`scripts/v3_status.py` prints all stages, all folds, the ETA, and the
last 5 log lines in a single snapshot.

```
.venv/bin/python scripts/v3_status.py
```

Auto-refresh every 60 seconds:

```
watch -n 60 .venv/bin/python scripts/v3_status.py
```

## Per-fold AUC tracker

`scripts/v3_fold_watch.py` is the live per-fold AUC view. It polls every
30 seconds, prints one line per new fold with the running pos-weighted
AUC by group, and backfills folds that already landed when it starts.

```
.venv/bin/python scripts/v3_fold_watch.py
```

Ctrl-C to stop. Stopping the watcher does not affect the trainer.

## Raw log firehose

Tail the active training log directly:

```
tail -f /tmp/v36_train.log
```

(Substitute the current log filename when the version bumps.) Warnings
drown the signal once sklearn starts spamming; filter them out:

```
tail -f /tmp/v36_train.log | grep -v "warnings.warn\|UserWarning"
```

## Process inspection

Find the running trainer PID, its parent, and start time:

```
ps -ef | grep train_predict_250m | grep -v grep
```

The parent column tells you whether the trainer is attached to your
shell, to nohup, or to tmux, which matters for how you reattach or kill
it.

## Checkpoint directory state

Summarize how many folds have landed per population per stage:

```
ls data/derived/northern_sierra_placer/_k5_checkpoints/ \
  | awk -F"__" '{print $1, $2}' \
  | sort | uniq -c
```

The count column tells you fold progress at a glance; the population
plus stage tells you which group is still outstanding.

## Pause, resume, clean shutdown

Checkpoints persist across any restart, so the kill-and-resume pattern
costs only the in-flight fold.

- Pause without losing in-memory state: `kill -SIGSTOP <pid>`. The
  process freezes; nothing writes, nothing reads.
- Resume from pause: `kill -SIGCONT <pid>`. Picks up where it left off.
- Clean shutdown plus restart: `kill -SIGKILL <pid>`, then re-launch
  the trainer the same way it was started. Already-completed fold
  checkpoints are reused on restart; only the partially-written fold
  is recomputed.

SIGSTOP is the right tool when you want to free CPU briefly (other
heavy job, video call) without losing the resident dataset. SIGKILL
plus restart is the right tool when something is wrong with the run
itself and you want to change config or code before continuing.
