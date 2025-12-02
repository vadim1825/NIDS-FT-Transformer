import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, regularizers, Model, Input
import time
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg') # Встановлюємо бекенд для збереження файлів без відображення вікна
import matplotlib.pyplot as plt

class AddCLSToken(layers.Layer):
    def __init__(self, embedding_dim, **kwargs):
        super(AddCLSToken, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        # Ініціалізація [CLS]-токена як навчальної змінної
        self.cls_token = self.add_weight(
            shape=(1, 1, embedding_dim),
            initializer=tf.random_normal_initializer(),
            trainable=True,
            name='cls_token'
        )

    def call(self, x):
        # Отримання розміру батчу
        batch_size = tf.shape(x)[0]
        # Розширення [CLS]-токена до розміру батчу
        cls_token = tf.tile(self.cls_token, [batch_size, 1, 1])
        # Конкатенація [CLS]-токена з вхідними вбудовуваннями
        x = tf.concat([cls_token, x], axis=1)
        return x

    def get_config(self):
        config = super(AddCLSToken, self).get_config()
        config.update({'embedding_dim': self.embedding_dim})
        return config

pd.options.display.expand_frame_repr = False
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

NUM_CLASSES = 4
EMBEDDING_DIM = 32
NUM_HEADS = 2
FFN_DIM = 64
NUM_TRANSFORMER_BLOCKS = 1
DROPOUT_RATE = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 10
REGULARIZATION = 0.005

pre_drop_df=pd.read_csv('UNSW_NB15_training_set_csv.csv')
df = pre_drop_df.drop(columns=['label', 'id']).copy()

df = df[df['attack_cat'].isin(['Normal', 'Exploits', 'Fuzzers', 'Generic'])]

le = LabelEncoder()

df['attack_cat'] = le.fit_transform(df['attack_cat'])

print(df)

print(df['attack_cat'].value_counts())

Y = df['attack_cat']
X = df.drop(columns=['attack_cat'])
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

numerical_cols = list(X.select_dtypes(include=np.number).columns)
categorical_cols = list(X.select_dtypes(exclude=np.number).columns)

print(f"Числові ознаки ({len(numerical_cols)}): {numerical_cols[:5]}...")
print(f"Категоріальні ознаки ({len(categorical_cols)}): {categorical_cols}")

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    combined_data = pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str)
    label_encoders[col].fit(combined_data)
    X_train[col] = label_encoders[col].transform(X_train[col].astype(str))
    X_val[col] = label_encoders[col].transform(X_val[col].astype(str))
    X_test[col] = label_encoders[col].transform(X_test[col].astype(str))

smote = SMOTE(random_state=42)
X_train_values = X_train.values
y_train_values = y_train.values
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_values, y_train_values)

X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
y_train = pd.Series(y_train_resampled)

print("Розподіл класів після SMOTE:")
print(pd.Series(y_train_resampled).value_counts())
print("\n")

y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
y_train = np.array(y_train_cat)
y_val = np.array(y_val_cat)
y_test = np.array(y_test_cat)

def create_ft_transformer(input_cols, categorical_cols, label_encoders, num_classes, embedding_dim, num_heads, ffn_dim, num_transformer_blocks, dropout_rate):
    inputs = []
    embeddings = []

    # Feature Tokenizer
    for col in input_cols:
        input_layer = Input(shape=(1,), name=f"input_{col}")
        inputs.append(input_layer)
        if col in categorical_cols:
            vocab_size = len(label_encoders[col].classes_) + 1
            print(f"Колонка: {col}, vocab_size: {vocab_size}")
            embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
            embedding = layers.Reshape((embedding_dim,))(embedding)
        else:
            embedding = layers.Dense(embedding_dim, activation=None)(input_layer)
            embedding = layers.LayerNormalization(epsilon=1e-6)(embedding)
        embeddings.append(embedding)

    x = layers.Concatenate(axis=-1)(embeddings)
    x = layers.Reshape((len(input_cols), embedding_dim))(x)

    x = AddCLSToken(embedding_dim)(x)

    for _ in range(num_transformer_blocks):
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate
        )(x_norm, x_norm)
        x = layers.Add()([x, attn_output])
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = layers.Dense(ffn_dim, activation='relu', kernel_regularizer=regularizers.l2(REGULARIZATION))(x_norm)
        ffn_output = layers.Dense(embedding_dim, kernel_regularizer=regularizers.l2(REGULARIZATION))(ffn_output)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        x = layers.Add()([x, ffn_output])

    x = x[:, 0, :]
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(REGULARIZATION))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

results = []
reports = []
histories = []
class_names = [f'Class {i} ({le.inverse_transform([i])[0]})' for i in range(NUM_CLASSES)]

for run in range(3):
    print(f"\n=== Запуск {run + 1}/3 ===")

    tf.random.set_seed(run)
    np.random.seed(run)

    ft_model = create_ft_transformer(
        input_cols=X_train.columns,
        categorical_cols=categorical_cols,
        label_encoders=label_encoders,
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        dropout_rate=DROPOUT_RATE
    )

    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
    ft_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(name='auc')]
    )

    train_inputs = [X_train[col].values for col in X_train.columns]
    val_inputs = [X_val[col].values for col in X_train.columns]
    test_inputs = [X_test[col].values for col in X_train.columns]

    checkpoint_path = f"models/ft_transformer_best_model_run_{run}.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=f"/logs/ft_run_{run}", histogram_freq=1)
    ]

    start_time = time.time()
    history = ft_model.fit(
        train_inputs, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_inputs, y_val),
        callbacks=callbacks,
        verbose=1
    )
    end_time = time.time()
    print(f"Час навчання для запуску {run + 1}: {end_time - start_time:.2f} секунд")
    histories.append(history.history)

    test_metrics = ft_model.evaluate(test_inputs, y_test, verbose=0)
    results.append(test_metrics)
    print(f"\nОцінка для запуску {run + 1}:")
    print(f"Test loss: {test_metrics[0]:.4f}")
    print(f"Test accuracy: {test_metrics[1]*100:.2f}%")
    print(f"Test precision: {test_metrics[2]:.4f}")
    print(f"Test recall: {test_metrics[3]:.4f}")
    print(f"Test AUC: {test_metrics[4]:.4f}")

    y_pred_prob = ft_model.predict(test_inputs, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    reports.append(report)
    print(f"\nClassification Report для запуску {run + 1}:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    for cls, cls_name in enumerate(class_names):
        idx_cls = np.where(y_true == cls)[0]
        y_pred_cls = y_pred[idx_cls]
        mistakes = idx_cls[y_pred_cls != cls]
        if len(mistakes) > 0:
            print(f"\nАналіз помилок для {cls_name} (помилково класифіковано: {len(mistakes)} з {len(idx_cls)}):")
            mistake_counts = pd.Series(y_pred[mistakes]).value_counts()
            for mistaken_cls, count in mistake_counts.items():
                print(f"  Клас {mistaken_cls} ({le.inverse_transform([mistaken_cls])[0]}): {count} випадків")

    ft_model.save(f"models/ft_transformer_run_{run}.keras")

results = np.array(results)
print("\n=== Середні метрики (3 запусків) ===")
print(f"Середні метрики (loss, accuracy, precision, recall, AUC): {np.mean(results, axis=0)}")
print(f"Стандартне відхилення: {np.std(results, axis=0)}")

metrics_per_class = {name: {'precision': [], 'recall': [], 'f1-score': []} for name in class_names}
for report in reports:
    for name in class_names:
        metrics_per_class[name]['precision'].append(report[name]['precision'])
        metrics_per_class[name]['recall'].append(report[name]['recall'])
        metrics_per_class[name]['f1-score'].append(report[name]['f1-score'])

print("\n=== Середні метрики Classification Report ===")
for name in class_names:
    print(f"\n{name}:")
    print(f"Precision: {np.mean(metrics_per_class[name]['precision']):.3f} ± {np.std(metrics_per_class[name]['precision']):.3f}")
    print(f"Recall: {np.mean(metrics_per_class[name]['recall']):.3f} ± {np.std(metrics_per_class[name]['recall']):.3f}")
    print(f"F1-score: {np.mean(metrics_per_class[name]['f1-score']):.3f} ± {np.std(metrics_per_class[name]['f1-score']):.3f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i, h in enumerate(histories):
    plt.plot(h['accuracy'], label=f'Тренування (запуск {i+1})', alpha=0.3)
    plt.plot(h['val_accuracy'], label=f'Валідація (запуск {i+1})', alpha=0.3, linestyle='--')
plt.title('Точність моделі (усі запуски)')
plt.xlabel('Епохи')
plt.ylabel('Точність')
plt.legend()

plt.subplot(1, 2, 2)
for i, h in enumerate(histories):
    plt.plot(h['loss'], label=f'Тренування (запуск {i+1})', alpha=0.3)
    plt.plot(h['val_loss'], label=f'Валідація (запуск {i+1})', alpha=0.3, linestyle='--')
plt.title('Функція втрат (усі запуски)')
plt.xlabel('Епохи')
plt.ylabel('Втрати')
plt.legend()

plt.tight_layout()
plt.savefig('images/ft_transformer_training_metrics_3runs.png')
plt.show()