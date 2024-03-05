export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7,8

torchrun --nproc_per_node 8 -m training.main \
    --train-data="/home/admin/andreas/laion/train-part-2/{00000..14233}.tar::/home/admin/andreas/laion/train-part-3/{00000..05290}.tar::pipe:aws s3 cp s3://laion-400m-data/train/{00002..26002}.tar -" \
    --train-num-samples 24576000 \
    --val-data="pipe:aws s3 cp s3://laion-400m-data/data/{00000..00001}.tar -" \
    --val-num-samples 15000 \
    --dataset-type webdataset \
    --batch-size 1024 \
    --warmup 15000 \
    --epochs 120 \
    --lr 5e-4 \
    --precision amp \
    --workers 16 \
    --model "jina-clip-ViT-B-32-3towers-text" \
    --name "jina-clip-ViT-B-32-3towers-text-cosine-embedding-loss" \
    --grad-checkpointing \
    --force-custom-text \
    --log-every-n-steps 20 \
    --report-to "wandb" \
    --wandb-project-name "jina-clip-laion400m-full" \
    --clip-benchmark-frequency 2 \
    --mteb-frequency 2\
    --hf-load-pretrained \
    --evaluate-on-start \
    --3towers-cos-embeddings-loss-weight 1.0 \
    --3towers-contrastive-loss-weight 1.0 \