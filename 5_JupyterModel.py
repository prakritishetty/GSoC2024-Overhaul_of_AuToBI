import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from datasets import load_dataset
dataset = load_dataset('./dataset.py', split='train', trust_remote_code=True)
print(dataset)
example = dataset[0]
print(example)
dataset = dataset.train_test_split(test_size=0.2)
print(dataset)
print(dataset["train"][0])

from transformers import AutoFeatureExtractor
model_id = "microsoft/wavlm-base-plus"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize = True, return_attention_mask = True)

sampling_rate = feature_extractor.sampling_rate
sampling_rate

from datasets import Audio
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

max_duration = 20.0

def preprocess_function(examples):
    audio_arrays = [augment_audio(x["array"]) for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
    )
    inputs["labels"] = examples["label"]
    return inputs

dataset_encoded = dataset.map(
    preprocess_function,
    batched= True,
    batch_size = 20,
    num_proc = 4,
    # remove_columns=dataset["train"].column_names,
)
print(dataset_encoded)

unique_labels_train = set(dataset_encoded["train"]["label"])
unique_labels_test = set(dataset_encoded["test"]["label"])
print(f"Unique labels in train set: {unique_labels_train}")
print(f"Unique labels in test set: {unique_labels_test}")


from collections import Counter
label_counts = Counter(dataset_encoded["train"]["label"])
print(f"Label distribution in train set: {label_counts}")
label_counts_test = Counter(dataset_encoded["test"]["label"])
print(f"Label distribution in test set: {label_counts_test}")

from transformers import AutoModelForAudioClassification
model = AutoModelForAudioClassification.from_pretrained(
     model_id,
     problem_type = "single_label_classification",
     num_labels=3
 ).to(device)


import evaluate
import numpy as np
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

import torch
import torch.nn as nn
def compute_class_weights(dataset):
    label_counts = Counter(dataset["train"]["label"])
    total_samples = sum(label_counts.values())
    class_weights = torch.tensor([
        (total_samples / (len(label_counts) * count)) ** 1.5  # Increase the exponent
        for count in label_counts.values()
    ], dtype=torch.float)
    return class_weights
  
class_weights = compute_class_weights(dataset_encoded).to(device)

import numpy as np
def convert_time_to_int64(example):
    example['start_time'] = np.int64(float(example['start_time']) * 1e9)
    example['end_time'] = np.int64(float(example['end_time']) * 1e9)
    return example

dataset_encoded['train'] = dataset_encoded['train'].map(convert_time_to_int64)
dataset_encoded['test'] = dataset_encoded['test'].map(convert_time_to_int64)

print("Train dataset:")
print(dataset_encoded['train'].features['start_time'])
print(dataset_encoded['train'].features['end_time'])
print("\nTest dataset:")
print(dataset_encoded['test'].features['start_time'])
print(dataset_encoded['test'].features['end_time'])

from collections import Counter
label_counts = Counter(dataset_encoded["train"]["labels"])
print(f"Label distribution in train set: {label_counts}")


from imblearn.over_sampling import RandomOverSampler
X = np.arange(len(dataset_encoded["train"])).reshape(-1, 1)
y = dataset_encoded["train"]["labels"]
# Create an instance of RandomOverSampler
ros = RandomOverSampler(random_state=42)
# Now use the instance to fit_resample
X_resampled, y_resampled = ros.fit_resample(X, y)
resampled_dataset = dataset_encoded["train"].select(X_resampled.flatten())

from collections import Counter
label_counts = Counter(resampled_dataset["label"])
print(f"Label distribution in train set: {label_counts}")


from transformers import WavLMModel
import torch.nn as nn
class CustomAudioClassifier(nn.Module):
    def __init__(self, model_id, num_labels):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_id)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values, attention_mask):
        outputs = self.wavlm(input_values=input_values, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled_output)

model = CustomAudioClassifier(model_id, num_labels=3).to(device)
print(model.classifier)

from typing import Dict, List, Union, Any, Optional, Tuple
from transformers import Trainer
from transformers import get_scheduler

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(input_values=inputs["input_values"], attention_mask=inputs["attention_mask"])
        loss_fn = FocalLoss(alpha=class_weights)
        loss = loss_fn(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        labels = inputs.pop("labels")
        with torch.no_grad():
            outputs = model(input_values=inputs["input_values"], attention_mask=inputs["attention_mask"])
            loss = None
            if labels is not None:
                loss_fn = FocalLoss(alpha=class_weights)
                loss = loss_fn(outputs, labels)
        return (loss, outputs, labels)


import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (1-pt)**self.gamma * CE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


loss_fn = FocalLoss(alpha=class_weights)


from transformers import TrainingArguments

total_train_examples = len(resampled_dataset)
model_name = model_id.split("/")[-1]
batch_size = 16
steps_per_epoch = total_train_examples // batch_size
logging_steps = int(steps_per_epoch * 0.01)
gradient_accumulation_steps = 2
num_train_epochs = 5

training_args = TrainingArguments(
    f"{model_name}-finetuned-dummy",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = 1e-5,
    per_device_train_batch_size = 3,
    num_train_epochs = num_train_epochs,
    warmup_ratio = 0.1,
    logging_steps = logging_steps,
    load_best_model_at_end = True,
    metric_for_best_model = "accuracy",
    fp16= True,
max_grad_norm=1.0,
    
)



try:
    from transformers import Trainer
    import traceback
    from transformers import TrainerCallback

    class DetailedPrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_local_process_zero:
                print(f"Step {state.global_step}:")
                for k, v in logs.items():
                    print(f"  {k}: {v}")
                print("\n")
                
    class PerClassAccuracyCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is not None and 'eval_predictions' in metrics:
                predictions = metrics['eval_predictions']
                labels = metrics['eval_labels']
                for class_id in range(3):  # Assuming 3 classes
                    class_preds = (predictions == class_id)
                    class_labels = (labels == class_id)
                    class_accuracy = (class_preds == class_labels).float().mean()
                    print(f"Class {class_id} accuracy: {class_accuracy:.4f}")

    
    trainer = CustomTrainer(
    model,
    training_args,
    train_dataset = resampled_dataset,
    eval_dataset = dataset_encoded["test"],
    tokenizer = feature_extractor,
    compute_metrics = compute_metrics,
    callbacks=[PerClassAccuracyCallback()]   
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

except Exception:
    print(traceback.format_exc())

import pandas as pd
import numpy as np


predictions = trainer.predict(dataset_encoded["test"])
raw_logits = predictions.predictions
print("Raw logits shape:", raw_logits.shape)
print("Sample raw logits:", raw_logits[:5])


predicted_labels = np.argmax(predictions.predictions, axis=1)

original_test_data = dataset_encoded["test"].to_pandas()
original_test_data = original_test_data.drop(columns=['audio']) 
original_test_data = original_test_data.drop(columns=['input_values'])
original_test_data = original_test_data.drop(columns=['attention_mask'])


original_test_data["predicted_label"] = predicted_labels

print(original_test_data) 


# original_test_data.to_csv("test_results_with_predictions.csv", index=False)


