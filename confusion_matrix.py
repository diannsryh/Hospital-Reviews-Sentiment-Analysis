class_names = ['Positive', 'Neutral', 'Negative']
y_true = []
for _, y in test_dataset:
    y_true.append(y.numpy())
y_true = np.array(y_true)

y_pred = model.predict(test_dataset.batch(BATCH))

if isinstance(y_pred, dict):
    y_pred = y_pred['logits']

if len(y_pred.shape) == 1:
    y_pred_classes = (y_pred > 0.5).astype(int)
else:
    y_pred_classes = np.argmax(y_pred, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Value", fontsize=8)
plt.ylabel("Actual Value", fontsize=8)
plt.title("Confusion Matrix", fontsize=10)
plt.show()

print("\n")
print("Classification Report: \n", classification_report(y_true, y_pred_classes, target_names=class_names))