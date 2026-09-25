# prune_model.py

import argparse
import os

# tf-mot targets the Keras 2 API, so opt into the legacy Keras backend before
# importing TensorFlow / tf-mot. Requires the `tf-mot` extra (installs
# tensorflow-model-optimization + tf-keras).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflow_model_optimization as tfmot  # noqa: E402

# --- Configuration ---
BASELINE_MODEL_PATH = "models/baseline_model"
# Path where we will save our final, pruned model.
PRUNED_MODEL_SAVE_PATH = "models/pruned_model"
FINE_TUNE_EPOCHS = 3
BATCH_SIZE = 64


def load_dataset():
    """
    Loads the CIFAR-10 dataset. We now need both training and test sets.
    The test set will be used for validation during fine-tuning.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)


def main():
    """
    Main function to orchestrate the model pruning process.
    """
    p = argparse.ArgumentParser(description="Prune the baseline model with tf-mot.")
    p.add_argument("--baseline", default=BASELINE_MODEL_PATH)
    p.add_argument("--save-path", default=PRUNED_MODEL_SAVE_PATH)
    p.add_argument("--epochs", type=int, default=FINE_TUNE_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--final-sparsity", type=float, default=0.50)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N train/test samples (fast smoke test).",
    )
    args = p.parse_args()

    print("--- Starting Model Pruning Workflow ---")

    if not os.path.exists(args.baseline):
        print(f"Error: Baseline model not found at {args.baseline}")
        return

    # Load the baseline model
    print(f"Loading baseline model from: {args.baseline}")
    baseline_model = tf.keras.models.load_model(args.baseline)
    print("Baseline model loaded successfully.")

    # Load data
    (train_images, train_labels), (test_images, test_labels) = load_dataset()
    if args.limit is not None:
        train_images, train_labels = train_images[: args.limit], train_labels[: args.limit]
        test_images, test_labels = test_images[: args.limit], test_labels[: args.limit]
    print(f"Loaded {len(train_images)} training images and {len(test_images)} test images.")

    # Define the pruning schedule
    print("\n[TASK] Defining the pruning schedule...")
    num_train_samples = len(train_images)
    end_step = np.ceil(num_train_samples / args.batch_size).astype(np.int32) * args.epochs

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=args.final_sparsity,
            begin_step=0,
            end_step=int(end_step),
            frequency=100,
        )
    }
    print(
        f"Pruning schedule defined to reach {args.final_sparsity:.0%} sparsity in {end_step} steps."
    )

    # Apply the pruning wrapper
    print("\n[TASK] Applying the pruning wrapper to the model...")
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(baseline_model, **pruning_params)
    print("Pruning wrapper applied successfully.")

    # Re-compile the prunable model
    print("\n[TASK] Re-compiling the prunable model...")
    model_for_pruning.compile(
        optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )
    print("Prunable model re-compiled successfully.")

    # Fine-tune the pruned model
    print("\n[TASK] Fine-tuning the prunable model...")
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    print(f"Starting fine-tuning for {args.epochs} epochs...")
    model_for_pruning.fit(
        train_images,
        train_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(test_images, test_labels),
        callbacks=callbacks,
    )
    print("--- Fine-tuning complete. ---")

    # Strip the pruning wrappers
    print("\n[TASK] Stripping pruning wrappers to get the final model...")
    model_stripped = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    print("Pruning wrappers stripped successfully.")

    # Task: Save the pruned and fine-tuned model.
    print(f"\n[TASK] Saving the final stripped and pruned model to: {args.save_path}")

    # Create the directory to save the model if it doesn't exist.
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # We call `save()` on our final, stripped model. This saves the standard
    # Keras model with its new sparse weights into the robust SavedModel format.
    model_stripped.save(args.save_path)

    print("--- Final pruned model saved successfully. ---")

    # --- NEW CODE ENDS HERE ---


if __name__ == "__main__":
    main()
