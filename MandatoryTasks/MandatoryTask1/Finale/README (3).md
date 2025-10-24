# English Question Answering with French Translation System

A machine learning system that processes English questions and provides answers in French by combining fine-tuned question answering models with neural machine translation.

## System Architecture

The system operates in two stages:
1. **Question Answering Module**: RoBERTa-based model fine-tuned on NewsQA dataset extracts answers from English contexts
2. **Translation Module**: mBART50 M2M model translates English answers to French

## Key Features

- Fine-tuned RoBERTa model for accurate answer extraction
- Neural machine translation for a quality French output


## Requirements

### Dependencies

```bash
transformers==4.28.0
torch>=1.6.0
datasets
easynmt>=2.0.2
numpy
pandas
tqdm
nltk
```


## Installation

```bash
# Install core dependencies
pip install transformers==4.28.0
pip install easynmt
pip install datasets

# Install NLP utilities
pip install nltk
python -m nltk.downloader punkt punkt_tab
```

## Dataset Preparation

The system uses the NewsQA dataset from Hugging Face:

```python
from datasets import load_dataset
import pandas as pd

# Loading the dataset
splits = {
    'train': 'data/train-00000-of-00001-ec54fbe500fc3b5c.parquet',
    'validation': 'data/validation-00000-of-00001-3cf888b12fff1dd6.parquet'
}

df_train = pd.read_parquet("hf://datasets/lucadiliello/newsqa/" + splits["train"])
df_val = pd.read_parquet("hf://datasets/lucadiliello/newsqa/" + splits["validation"])
```

### Dataset Structure

```python
{
    'context': 'NEW DELHI, India (CNN) -- A high court in northern India...',
    'question': 'When was Pandher sentenced to death?',
    'answers': ['February.'],
    'key': '724f6eb9a2814e4fb2d7d8e4de846073',
    'labels': [{'end': [269], 'start': [261]}]
}
```

## Model Training

### Data Preprocessing

```python
from transformers import RobertaTokenizerFast

MODEL_NAME = "deepset/roberta-base-squad2"
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

MAX_LENGTH = 256
DOC_STRIDE = 64

def prepare_features(examples):
    tokenized_list = {
        "input_ids": [],
        "attention_mask": [],
        "start_positions": [],
        "end_positions": []
    }

    for i in range(len(examples["context"])):
        context = examples["context"][i]
        question = examples["question"][i]
        answer = examples["answers"][i][0]
        answer_text = answer["text"]
        answer_start = answer["answer_start"]
        answer_end = answer_start + len(answer_text)

        encodings = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        # Process overflow tokens and map answer positions
        overflow_sample_mapping = encodings.pop("overflow_to_sample_mapping")
        offset_mapping = encodings.pop("offset_mapping")

        for j, offsets in enumerate(offset_mapping):
            input_ids = encodings["input_ids"][j]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            start_token, end_token = cls_index, cls_index
            for idx, (start_off, end_off) in enumerate(offsets):
                if start_off <= answer_start < end_off:
                    start_token = idx
                if start_off < answer_end <= end_off:
                    end_token = idx

            tokenized_list["input_ids"].append(input_ids)
            tokenized_list["attention_mask"].append(encodings["attention_mask"][j])
            tokenized_list["start_positions"].append(start_token)
            tokenized_list["end_positions"].append(end_token)

    return tokenized_list
```

### Training Configuration

```python
from transformers import RobertaForQuestionAnswering, TrainingArguments, Trainer

model = RobertaForQuestionAnswering.from_pretrained(MODEL_NAME)

args = TrainingArguments(
    output_dir="./newsqa_fast",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    num_train_epochs=0.3,
    weight_decay=0.01,
    fp16=True,
    logging_steps=10000,
    report_to="none",
    disable_tqdm=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()
trainer.save_model("./newsqa_roberta_final")
tokenizer.save_pretrained("./newsqa_roberta_final")
```

### Training Results

Training metrics from NewsQA dataset:

| Metric | Value |
|--------|-------|
| Training Loss | 0.8536 |
| Evaluation Loss | 0.8010 |
| Training Samples/Second | 34.272 |


## Example Usage

### Basic Usage with Context and Question being provided

```python
from transformers import pipeline
from easynmt import EasyNMT

# Initialize Question Answering pipeline
qa = pipeline(
    "question-answering",
    model="./newsqa_roberta_final",
    tokenizer="./newsqa_roberta_final",
    framework="pt",
)

# Initialize Translation model
translator = EasyNMT('mbart50_m2m')

# Input data
context = """The Amazon rainforest is often called the "lungs of the Earth" 
because it produces a large portion of the planet's oxygen. However, 
deforestation has caused a significant decrease in its size over the past decades."""

question = "Why is the Amazon rainforest called the lungs of the Earth?"

# Get answer in English
result = qa(question=question, context=context)
answer_text = result["answer"]

# Translate to French
answer_fr = translator.translate(answer_text, target_lang='fr')

print("Answer (English):", answer_text)
print("Answer (French):", answer_fr)
```

### Output

```
Answer (English): it produces a large portion of the planet's oxygen.
Answer (French): Il produit une grande partie de l'oxygène de la planète.
```


## System Workflow

```
┌─────────────────┐
│  English Input  │
│  Question +     │
│  Context        │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  RoBERTa QA Model           │
│  - Tokenization             │
│  - Answer Span Detection    │
│                             │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│  English Answer │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  mBART50 M2M Translator     │
│  - English to French        │
│  - Semantic Preservation    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│  French Answer  │
└─────────────────┘


```


## Example Use Cases

### Use Case 1: Educational Platform

```python
context = """Python is a high-level programming language known for its 
simplicity and readability. It was created by Guido van Rossum and first 
released in 1991. Python supports multiple programming paradigms including 
procedural, object-oriented, and functional programming."""

question = "Who created Python?"

result = qa(question=question, context=context)
answer_fr = translator.translate(result["answer"], target_lang='fr')

print(f"Answer: {answer_fr}")
# Output: "Guido van Rossum"
```

### Use Case 2: News Summarization

```python
context = """The 2024 Olympic Games were held in Paris, France. The event 
featured over 10,000 athletes from 206 countries competing in 329 events 
across 32 sports. France topped the medal table with 64 medals."""

question = "How many athletes participated in the 2024 Olympics?"

result = qa(question=question, context=context)
answer_fr = translator.translate(result["answer"], target_lang='fr')

print(f"Réponse: {answer_fr}")
# Output: "plus de 10 000 athlètes"
```

### Use Case 3: Customer Support

```python
context = """Our standard warranty covers manufacturing defects for 2 years 
from the date of purchase. Extended warranty plans are available for up to 
5 years. To file a warranty claim, contact our support team with your 
purchase receipt and product serial number."""

question = "How long is the standard warranty?"

result = qa(question=question, context=context)
answer_fr = translator.translate(result["answer"], target_lang='fr')

print(f"Réponse: {answer_fr}")
# Output: "2 ans à compter de la date d'achat"
```



## Resources used 

- Hugging Face for Transformers library and model hosting
- EasyNMT project for simplified translation interface
- NewsQA dataset creators for training data
- deepset.ai for pre-trained RoBERTa-SQuAD2 model


## Version History

- **v1.0.0** (2025-10-24): Initial release
  - RoBERTa-based QA model
  - mBART50 translation integration
  - Basic training pipeline
  - Documentation and examples

