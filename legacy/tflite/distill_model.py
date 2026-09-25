# distill_model.py

import argparse
import os

# Use the legacy Keras 2 backend (SavedModel dirs, custom-model compiled_metrics).
# Requires `tf-keras` (installed via the `tf` extra).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

# --- Configuration --
BASELINE_MODEL_PATH = "models/baseline_model"
# Path to save the distilled student model (SavedModel directory)
STUDENT_MODEL_PATH = "models/student_model"
# Alpha controls the balance between student and distillation loss.
ALPHA = 0.1
# Temperature softens probability distributions for distillation.
TEMPERATURE = 10
# --- NEW CONFIGURATION ---
# We will train the student for a sufficient number of epochs to absorb knowledge.
EPOCHS = 30
# A standard batch size for this type of task.
BATCH_SIZE = 64
# --- END NEW CONFIGURATION ---


# --- NEW HELPER FUNCTION ---
def load_and_preprocess_data():
    """
    Loads and preprocesses the CIFAR-10 dataset.

    Returns:
        A tuple of (x_train, y_train), (x_test, y_test) with normalized pixel values.
    """
    print("Loading and preprocessing CIFAR-10 data...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to the [0, 1] range.
    # This is optimal for the student model's training.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    print("Data loaded and normalized successfully.")
    return (x_train, y_train), (x_test, y_test)


# --- Student Model Definition (Unchanged) ---
def create_student_model(input_shape=(32, 32, 3), num_classes=10):
    student_model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dense(num_classes),
        ],
        name="student",
    )
    return student_model


# --- Distiller Class with train_step and test_step (Unchanged) ---
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=10,
    ):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            teacher_predictions = self.teacher(x, training=False)
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            ) * (self.temperature**2)
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars, strict=True))
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {
                "student_loss": student_loss,
                "distillation_loss": distillation_loss,
                "total_loss": total_loss,
            }
        )
        return results

    def test_step(self, data):
        x, y = data
        student_predictions = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, student_predictions)
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

    def call(self, x, training=False):
        return self.student(x, training=training)


# --- Main Workflow ---
def main():
    p = argparse.ArgumentParser(description="Knowledge distillation into a compact CNN student.")
    p.add_argument("--baseline", default=BASELINE_MODEL_PATH)
    p.add_argument("--save-path", default=STUDENT_MODEL_PATH)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N train/test samples (fast smoke test).",
    )
    args = p.parse_args()

    print("--- Knowledge Distillation Workflow ---")

    # --- Setup Student, Teacher, and Distiller ---
    student = create_student_model()
    if not os.path.exists(args.baseline):
        print(f"Error: Baseline model not found at {args.baseline}")
        return
    teacher = keras.models.load_model(args.baseline)
    teacher.trainable = False
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=args.alpha,
        temperature=args.temperature,
    )

    # Task: Write a custom training loop for the Distiller model.
    print("[TASK] Preparing data and starting the training process...")

    # Load and preprocess the dataset.
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    if args.limit is not None:
        x_train, y_train = x_train[: args.limit], y_train[: args.limit]
        x_test, y_test = x_test[: args.limit], y_test[: args.limit]

    # Train the distiller.
    # The `fit` method will now use our custom `train_step` and `test_step` logic.
    print(f"Starting training for {args.epochs} epochs...")
    distiller.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
    )

    print("--- Distillation training complete. ---")

    # Save the trained student model as a standalone artifact
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    print(f"Saving distilled student model to: {args.save_path}")
    distiller.student.save(args.save_path)
    print("Student model saved successfully.")


if __name__ == "__main__":
    main()
