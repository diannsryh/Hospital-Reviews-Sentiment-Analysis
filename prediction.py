new_texts = ['Facilities are clean', 'The service was excellent', 'Staff were rude and unhelpful', 'The experience was fine overall',
             'The quality of service was good', 'The experience was disappointing']
new_encodings = tokenizer(new_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='tf')

new_input_ids = new_encodings['input_ids'].numpy()
new_attention_mask = new_encodings['attention_mask'].numpy()

predictions = model.predict([new_input_ids, new_attention_mask])
logits = predictions.logits

predicted_labels = tf.argmax(logits, axis=1).numpy()
predicted_sentiments = [list(label_mapping.keys())[list(label_mapping.values()).index(label)] for label in predicted_labels]
print("Predicted sentiments:", predicted_sentiments)