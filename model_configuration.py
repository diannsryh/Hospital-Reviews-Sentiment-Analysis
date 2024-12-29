# ---- Load Pre-trained ALBERT Model ----
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=3)

# ---- Preprocessing Data: Tokenization and Padding ----
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='tf')
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='tf')

# ---- Label Preprocessing ----
label_mapping = {'Positive': 0, 'Negative': 1, 'Neutral': 2}

# Normalize and convert labels to numeric
train_labels = [label.strip().capitalize() for label in train_labels]
test_labels = [label.strip().capitalize() for label in test_labels]

train_labels_numeric = [label_mapping.get(label, -1) for label in train_labels]
test_labels_numeric = [label_mapping.get(label, -1) for label in test_labels]

# Validate labels
print("Unique train labels:", set(train_labels_numeric))
print("Unique test labels:", set(test_labels_numeric))

# Handle invalid labels
if -1 in train_labels_numeric or -1 in test_labels_numeric:
    raise ValueError("Invalid labels found in dataset. Check label preprocessing.")

# ---- Class Distribution ----
train_counter = Counter(train_labels_numeric)
test_counter = Counter(test_labels_numeric)
print("Train class distribution:", train_counter)
print("Test class distribution:", test_counter)

# Compute class weights to handle imbalance
class_weights = {
    0: 1.0 / train_counter[0],
    1: 1.0 / train_counter[1],
    2: 1.0 / train_counter[2]
}

# ---- Create tf.data.Dataset for Training and Testing ----
train_dataset = tf.data.Dataset.from_tensor_slices(((train_encodings['input_ids'], train_encodings['attention_mask']), train_labels_numeric))
test_dataset = tf.data.Dataset.from_tensor_slices(((test_encodings['input_ids'], test_encodings['attention_mask']), test_labels_numeric))