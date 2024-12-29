import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history['accuracy'], label='Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].legend()
axes[0].set_title ('Accuracy vs Validation Accuracy')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')

axes[1].plot(history.history['loss'], label='Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].legend()
axes[1].set_title('Loss vs Validation Loss')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')

plt.tight_layout()
plt.show()