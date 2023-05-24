import os
import sys
import random
sys.path.append("./model/")
sys.path.append("./preprocess/")

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
import evaluate
import torch

from preprocess.roundataset import roundataset
from model.modeling_deberta_visual import DebertaForPhotobookListener
from model.configuration_deberta_visual import DebertaWithVisualConfig
from model.variables import (
    EPOCHS, CKPT_DIR, RND_SEED,
    BATCH_SIZE, PEAK_LR, WARMUP_STEPS, WEIGHT_DECAY,
    PRETRAINED_MODEL_NAME, DLS
)

metric = evaluate.load("accuracy")

config_json = sys.argv[1]
CKPT_DIR = sys.argv[2] if len(sys.argv) > 2 else CKPT_DIR
RND_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else RND_SEED

# NOTE (Shih-Lun): borrowed from 
# https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
def set_rnd_seed():
    np.random.seed(RND_SEED)
    random.seed(RND_SEED)
    torch.manual_seed(RND_SEED)
    torch.cuda.manual_seed(RND_SEED)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(RND_SEED)
    print(f"[info] Random seed set as {RND_SEED}")


def compute_metrics(eval_pairs):
    predictions, labels = eval_pairs
    # print (labels[0])
    predictions = np.argmax(predictions[..., 1:], axis=-1) + 1

    true_predictions = []
    true_labels = []

    # fetch last timestep outputs only
    bsize, seqlen = predictions.shape[0], predictions.shape[1]
    for b in range(bsize):
        for pos in range(seqlen - 1, -1, -1):
            if labels[b, pos, 0] != -100:
                true_predictions.extend(predictions[b, pos])
                true_labels.extend(labels[b, pos])
                break
    
    # print (true_predictions, true_labels)
    results = metric.compute(predictions=true_predictions, references=true_labels)

    # print (results)

    return results

if __name__ == '__main__':
    set_rnd_seed()

    train_dset = roundataset(
        'data/train_clean_sections.pickle',
        'data/image_feats.pickle',
        separate_images=False,
        dense_learning_signals=DLS,
    )
    print ("[info] train set loaded, len =", len(train_dset))
    val_dset = roundataset(
        'data/valid_clean_sections.pickle',
        'data/image_feats.pickle',
        separate_images=False,
        dense_learning_signals=DLS,
    )
    print ("[info] valid set loaded, len =", len(val_dset))
    test_dset = roundataset(
        'data/test_clean_sections.pickle',
        'data/image_feats.pickle',
        separate_images=False,
        dense_learning_signals=DLS,
    )
    print ("[info] test dset loaded, len =", len(test_dset))

    print ("[info] using config:", config_json)

    model = DebertaForPhotobookListener(
                DebertaWithVisualConfig.from_json_file(
                    config_json
                )
            )
    model.deberta.load_hf_pretrained_deberta(PRETRAINED_MODEL_NAME)

    trainer = Trainer(
        model,
        TrainingArguments(
            output_dir=CKPT_DIR,
            do_train=True,
            do_eval=True,
            per_device_eval_batch_size=BATCH_SIZE,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=PEAK_LR,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=EPOCHS,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            metric_for_best_model="eval_accuracy",
            save_total_limit=3,
            dataloader_num_workers=8,
            logging_steps=50,
            load_best_model_at_end=True,
            gradient_accumulation_steps=1
        ),
        train_dataset=train_dset,
        eval_dataset=val_dset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics["train_samples"] = len(train_dset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # # Evaluation
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(val_dset)

    trainer.log_metrics("val", metrics)
    trainer.save_metrics("val", metrics)

    metrics = trainer.evaluate(eval_dataset=test_dset)
    metrics["test_samples"] = len(test_dset)

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
