# NeuCodec 48k

Scaling NeuCodec to 48k and reduce from 50TPS to 25TPS.

## Training

1. Use uv,

```bash
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
```

2. Prepare dataset,

A simple file with list of files, example, [example.txt](example.txt),

```
/path/audio1.wav
/path/audio2.wav
```

3. Run finetune,

Dry run,

```bash
python3 train.py \
log_dir=48k \
train.trainer.devices=2 \
train.trainer.max_steps=1000 \
train.trainer.min_steps=1000 \
train.trainer.val_check_interval=100 \
+train.accumulate_grad_batches=4 \
dataset.train.filelist="train.txt" \
dataset.train.batch_size=8 \
dataset.val.filelist="test.txt"
```

