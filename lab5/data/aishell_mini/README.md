# AISHELL mini

This dataset was rebuilt from a local AISHELL-1 copy for undergraduate speech information processing labs.

## Build

- Build time: 2026-05-21T00:54:35
- Source: local AISHELL-1 wav files
- Output directory: `/workplace/nankai/zhangyanzhe_space/Playground/mini-aishell1/aishell_mini`

```json
{
  "aishell_root": "AISHELL-1",
  "bonus_train_per_speaker": 8,
  "bonus_val_per_speaker": 3,
  "copy_mode": "copy",
  "dry_run": false,
  "enroll_per_speaker": 3,
  "gender_balance": true,
  "id_test_per_speaker": 4,
  "known_speakers": 16,
  "max_duration": 8.0,
  "min_duration": 2.0,
  "negatives_per_positive": 1,
  "output_dir": "aishell_mini",
  "overwrite": true,
  "same_gender_negatives": true,
  "seed": 42,
  "unknown_speakers": 4,
  "verbose": false,
  "verify_dev_per_speaker": 3,
  "verify_test_per_speaker": 3
}
```

## Statistics

- Known speakers: 16
- Unknown speakers: 4
- Selected utterances: 400
- Gender distribution: {0: 11, 1: 9}
- Role distribution: {'bonus_train': 128, 'identification_test': 64, 'enroll': 48, 'verification_dev': 48, 'verification_test': 48, 'bonus_val': 48, 'unknown_test': 16}
- Dev verification trials: {0: 48, 1: 48}
- Test verification trials: {0: 48, 1: 48}

## Files

- `metadata/speakers.csv`: speaker metadata and known/unknown labels.
- `metadata/utterances.csv`: one row per selected utterance with a single role.
- `protocols/enroll.csv`: enrollment utterances for known speakers.
- `protocols/identification_test.csv`: closed-set known tests plus open-set unknown tests.
- `protocols/verification_trials_dev.csv`: development trials for threshold scanning.
- `protocols/verification_trials_test.csv`: held-out test trials.
- `protocols/bonus_train.csv`: train/val utterances for the optional closed-set classifier bonus.

## Important Notes

- AISHELL-1 is an ASR corpus, not an official speaker verification benchmark.
- AISHELL mini enrollment, identification tests, and verification trials are reconstructed for undergraduate teaching.
- FAR, FRR, and approximate EER on this dataset are for teaching analysis only and do not represent formal benchmark performance.
- The CN-Celeb checkpoint is only a later pretrained model source; this dataset does not contain CN-Celeb data.
- This script does not resample or rewrite audio. It only copies or symlinks original AISHELL wav files.
