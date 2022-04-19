
# CUDA_LAUNCH_BLOCKING=1 python3 train_n_evaluate.py --batch-size=2 --bert-model=bert-base-uncased
# --bert-encoding-dim=768 --bert-max-input-len=512 --train-print-every=1 --update-every-n-batches=2 --lr=1e-05

#CUDA_LAUNCH_BLOCKING=1 python3 train_n_evaluate.py --batch-size=1 --bert-model=bert-base-uncased \
#--bert-encoding-dim=768 --bert-max-input-len=512 --train-print-every=40 --update-every-n-batches=40 --lr=5e-05 \
#--max-challenge-len=7 --include-in-span-layer=True --fine-tune-bert=True

CUDA_LAUNCH_BLOCKING=1 python3 train_n_evaluate.py --batch-size=2 --bert-model=bert-base-uncased \
--bert-encoding-dim=768 --bert-max-input-len=512 --train-print-every=50 --update-every-n-batches=10 \
--lr=5e-05 --warmup-steps-pct=10 --lr-decay-gamma=0.95 --lr-decay-steps=1000 --weight-decay=0.01 \
--max-challenge-len=5 --epochs=10 --checkpoint-every=2000 \
--train-data-filepath=../data/coqa-train-v1.0_preprocessed.json --dev-data-filepath=../data/coqa-dev-v1.0_preprocessed.json \
--logs-file=../experiment_6/logs.txt --predictions-dir=../experiment_6/predictions --models-dir=../experiment_6/models \
--history-window-size=3

#CUDA_LAUNCH_BLOCKING=1 python3 train_n_evaluate.py --batch-size=2 --bert-model=bert-base-uncased \
#--train-data-filepath=coqa-train-v1.0.json --dev-data-filepath=coqa-dev-v1.0.json \
#--bert-encoding-dim=768 --bert-max-input-len=512 --train-print-every=50 --update-every-n-batches=20 \
#--lr=6e-05 --warmup-steps-pct=10 --lr-decay-gamma=0.95 --lr-decay-steps=200 --weight-decay=0.01 \
#--max-challenge-len=5 --epochs=2 --checkpoint-every=1000 --include-in-span-layer=True --in-span-loss-lambda=0.15 \
#--train-data-filepath=coqa-train-v1.0_preprocessed.json --dev-data-filepath=coqa-dev-v1.0_preprocessed.json \
#--logs-file=experiment_6/logs.txt --predictions-dir=experiment_6/predictions --models-dir=experiment_6/models