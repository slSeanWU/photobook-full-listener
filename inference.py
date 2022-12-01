import os
import sys
sys.path.append("./model/")
sys.path.append("./preprocess/")

from transformers import Trainer, TrainingArguments
import numpy as np
import evaluate

from preprocess.roundataset import roundataset
from model.modeling_deberta_visual import DebertaForPhotobookListener
from model.configuration_deberta_visual import DebertaWithVisualConfig
from model.variables import (
    EPOCHS, CKPT_DIR,
    BATCH_SIZE, PEAK_LR, WARMUP_STEPS, WEIGHT_DECAY,
    PRETRAINED_MODEL_NAME,
)

metric = evaluate.load("accuracy")
ckpt_dir = sys.argv[1]

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
    test_dset = roundataset(
        'data/test_clean_sections.pickle',
        'data/image_feats.pickle'
    )
    print ("[info] test set loaded, len =", len(test_dset))

    model = DebertaForPhotobookListener.from_pretrained(ckpt_dir)

    trainer = Trainer(
        model,
        TrainingArguments(
            output_dir=ckpt_dir,
            do_train=False,
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
        ),
        eval_dataset=test_dset,
        compute_metrics=compute_metrics,
    )


    # # Evaluation
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(test_dset)

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    # predictions, labels, metrics = trainer.predict(test_dset, metric_key_prefix="predict")
    # predictions = np.argmax(predictions, axis=2)

    # trainer.log_metrics("test", metrics)
    # trainer.save_metrics("test", metrics)