  optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  history = model.fit(
      train_dataset.batch(BATCH),
      epochs=EPOCH,
      validation_data=test_dataset.batch(BATCH),
      class_weight=class_weights
  )

  eval_results = model.evaluate(test_dataset.batch(BATCH))
  print("Test loss:", eval_results[0])
  print("Test accuracy:", eval_results[1])
  model.save_pretrained(MODEL_PATH)
  print(f"Model saved to {MODEL_PATH}")