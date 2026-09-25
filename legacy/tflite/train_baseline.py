import argparse
import os

# These scripts use the Keras 2 API (SavedModel directories, tf-mot). TF 2.16+
# ships Keras 3 by default, so opt into the legacy Keras backend before importing
# TensorFlow. Requires the `tf-keras` package (installed via the `tf` extra).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf  # noqa: E402


def load_dataset():
    """
    Loads the CIFAR-10 dataset using TensorFlow's built-in datasets API.

    This function handles the download (if not already cached) and loading
    of the dataset into memory.

    Returns:
        A tuple containing two tuples:
        - The first tuple contains the training data and labels (x_train, y_train).
        - The second tuple contains the test data and labels (x_test, y_test).
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)


def get_data_augmentation_pipeline():
    """
    Creates and returns a Keras Sequential model for data preprocessing.

    The pipeline first rescales pixel values from [0, 255] to [0, 1]
    and then applies random transformations to increase dataset diversity
    and reduce overfitting.

    Returns:
        tf.keras.Sequential: Preprocessing + augmentation layers pipeline.
    """
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255, input_shape=(32, 32, 3)),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation_and_normalization",
    )
    return data_augmentation


def build_resnet50_model(
    preprocessing_pipeline: tf.keras.Model, weights: str | None = "imagenet"
) -> tf.keras.Model:
    """
    Build a ResNet50-based classifier for CIFAR-10 using transfer learning.

    Args:
        preprocessing_pipeline: Keras model performing rescaling and augmentation.
        weights: Pretrained weights for the ResNet50 backbone ("imagenet" or
            None). Use None for a fast smoke test to skip the weight download.

    Returns:
        Compiled tf.keras.Model ready for training.
    """
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = preprocessing_pipeline(inputs, training=True)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_tensor=x,
    )

    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the ResNet50 CIFAR-10 baseline (TensorFlow).")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--save-path", default="models/baseline_model")
    p.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="ResNet50 backbone weights. Use 'none' for a fast smoke test.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N train/test samples (fast smoke test).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # 1) Load dataset
    (train_images, train_labels), (test_images, test_labels) = load_dataset()
    if args.limit is not None:
        train_images, train_labels = train_images[: args.limit], train_labels[: args.limit]
        test_images, test_labels = test_images[: args.limit], test_labels[: args.limit]
    print(f"Dataset loaded: {len(train_images)} train / {len(test_images)} test images.")

    # 2) Build preprocessing pipeline
    preprocessing_pipeline = get_data_augmentation_pipeline()
    print("Preprocessing pipeline created.")

    # 3) Build ResNet50 model
    weights = None if args.weights == "none" else "imagenet"
    model = build_resnet50_model(preprocessing_pipeline, weights=weights)
    print("ResNet50 model built successfully.")

    # 4) Compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    print("Model compiled successfully.")

    # 5) Print model summary
    print("\n--- Model Summary ---")
    model.summary()

    # 6) Train the model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_images,
        train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(test_images, test_labels),
    )
    print("--- Model Training Finished ---")

    final_val_accuracy = history.history["val_accuracy"][-1]
    print(f"\nFinal validation accuracy: {final_val_accuracy:.4f}")

    # 7) Save the trained model (TensorFlow SavedModel format)
    print(f"\n--- Saving model to {args.save_path} ---")
    model.save(args.save_path)
    print("--- Model Saved Successfully ---")


if __name__ == "__main__":
    main()
