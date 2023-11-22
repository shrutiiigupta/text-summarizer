from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from transformers import pipeline

# Pick model
model_name = "google/pegasus-xsum"

# Load pretrained tokenizer
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)

with open('textin.txt') as f:
    example_text = f.read()

# Define PEGASUS model
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Create tokens
tokens = pegasus_tokenizer(example_text, truncation=True, max_length=512, padding="max_length")

#
# # Summarize text
# encoded_summary = pegasus_model.generate(**tokens)
#
# # Decode summarized text
# decoded_summary = pegasus_tokenizer.decode(
#       encoded_summary[0],
#       skip_special_tokens=True
# )


# Define summarization pipeline
summarizer = pipeline(
    "summarization",
    model=model_name,
    tokenizer=pegasus_tokenizer,
    framework="pt"
)

summary = summarizer(example_text, min_length=100, max_length=160)
print(summary[0]["summary_text"])
