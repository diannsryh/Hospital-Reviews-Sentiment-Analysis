EPOCH = 50
BATCH = 32
LEARNING_RATE = 1e-6
MAX_LENGTH = 256
MODEL_PATH = "models/my-albert-202312181131.h5"

FILE_PATH = '/content/datasets/processed_hospital_reviews.csv'
df = pd.read_csv(FILE_PATH)

# Initialization of features and class labels
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].values,
    df['sentiment'].values,
    test_size=0.2,
    random_state=42
)