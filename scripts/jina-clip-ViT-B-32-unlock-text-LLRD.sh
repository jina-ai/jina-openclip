export CUDA_VISIBLE_DEVICES=2

python -m training.main \
    --train-data="/shared/datasets/laion-400m/laion400m-data-new/{00000..01720}.tar::/shared/datasets/laion-400m/laion400m-data/{00000..01200}.tar" \
    --train-num-samples 1572864 \
    --val-data="/shared/datasets/laion-400m/laion400m-data-new/{01721..01722}.tar" \
    --val-num-samples 12000 \
    --dataset-type webdataset \
    --batch-size 1024 \
    --grad-checkpointing \
    --warmup 3072 \
    --epochs 50 \
    --lr 5e-4 \
    --text-lr-decay 0.98 \
    --vision-lr-decay 1.0 \
    --precision amp \
    --workers 4 \
    --model "jina-clip-ViT-B-32" \
    --force-custom-text \
    --log-every-n-steps 20 \
    --report-to "wandb" \
    --name "jina-clip-ViT-B-32-unlock-text-LLRD" \
    --wandb-project-name "jina-clip-laion400m-short-LLRD" \
    --clip-benchmark-frequency 1 \
    --mteb-frequency 1 \
    --evaluate-on-start