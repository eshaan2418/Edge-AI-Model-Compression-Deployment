# quantize_model.py

import argparse
import os

# Use the legacy Keras 2 backend (SavedModel directories). Requires `tf-keras`.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

# --- Configuration --
BASELINE_MODEL_PATH = "models/baseline_model"
TFLITE_DYNAMIC_QUANT_MODEL_PATH = "models/quantized_dynamic_range.tflite"
# --- NEW CONFIGURATION --
# Define the path for our new, full-integer TFLite model.
TFLITE_INT8_MODEL_PATH = "models/quantized_integer_only.tflite"
# --- END NEW CONFIGURATION --


# --- Helper Functions (representative_dataset_generator, etc.) ---
def load_training_data():
    """
    Loads the CIFAR-10 training dataset.
    We only need the images for our representative dataset.
    """
    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    return x_train


def representative_dataset_generator(training_images, num_samples=200):
    """
    Creates a generator that yields a small number of samples from the
    training data. This is used for calibrating the full integer quantization.
    """
    # This print statement will now appear during the .convert() call
    print(f"Feeding {num_samples} samples to the converter for calibration...")
    for i in range(num_samples):
        sample = training_images[i].astype("float32")
        sample = np.expand_dims(sample, axis=0)
        yield [sample]


# --- Main Workflow ---
def main():
    """
    Main function to orchestrate the model quantization process.
    """
    p = argparse.ArgumentParser(description="Quantize the baseline SavedModel to TFLite.")
    p.add_argument("--baseline", default=BASELINE_MODEL_PATH)
    p.add_argument(
        "--num-calibration-samples",
        type=int,
        default=200,
        help="Representative samples for full-integer calibration (lower = faster).",
    )
    args = p.parse_args()
    baseline_path = args.baseline

    print("--- Starting Model Quantization Workflow ---\n")

    if not os.path.exists(baseline_path):
        print(f"Error: Baseline model not found at {baseline_path}")
        return

    os.makedirs(os.path.dirname(TFLITE_DYNAMIC_QUANT_MODEL_PATH), exist_ok=True)

    # --- Section 1: Dynamic Range Quantization ---
    print("--- Section 1: Dynamic Range Quantization ---")
    converter_dynamic = tf.lite.TFLiteConverter.from_saved_model(baseline_path)
    converter_dynamic.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_dynamic = converter_dynamic.convert()
    with open(TFLITE_DYNAMIC_QUANT_MODEL_PATH, "wb") as f:
        f.write(tflite_model_dynamic)
    print(f"Dynamically quantized model saved to: {TFLITE_DYNAMIC_QUANT_MODEL_PATH}\n")

    # --- Section 2: Full Integer Quantization ---
    print("--- Section 2: Full Integer Quantization ---")

    train_images = load_training_data()
    n_samples = args.num_calibration_samples

    def representative_dataset():
        return representative_dataset_generator(train_images, n_samples)

    print("[TASK] Representative dataset generator for calibration is ready.")

    # Configure the converter for full integer quantization
    print("\n[TASK] Configuring the converter for FULL INTEGER quantization...")
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(baseline_path)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_dataset
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    print("Converter configured successfully.")

    # --- NEW CODE STARTS HERE ---

    # Task: Save the full-integer quantized model as a new .tflite file.
    print("\n[TASK] Converting and saving the full-integer quantized model...")
    print("This may take a moment as it runs the calibration process...")

    try:
        # The .convert() call now triggers the calibration with our generator.
        tflite_model_quant_int8 = converter_int8.convert()
        print("\nModel converted successfully after calibration.")
    except Exception as e:
        print(f"An error occurred during full integer conversion: {e}")
        return

    # Ensure the 'models' directory exists.
    os.makedirs(os.path.dirname(TFLITE_INT8_MODEL_PATH), exist_ok=True)

    # Save the final, fully-integer model to its own file.
    with open(TFLITE_INT8_MODEL_PATH, "wb") as f:
        f.write(tflite_model_quant_int8)

    print(f"Full-integer quantized model saved to: {TFLITE_INT8_MODEL_PATH}")

    # --- Size Comparison ---
    dynamic_quant_size_mb = os.path.getsize(TFLITE_DYNAMIC_QUANT_MODEL_PATH) / (1024 * 1024)
    int8_quant_size_mb = os.path.getsize(TFLITE_INT8_MODEL_PATH) / (1024 * 1024)

    print("\n--- Size Comparison ---")
    print(f"Dynamic Range TFLite model size: {dynamic_quant_size_mb:.2f} MB")
    print(f"Full Integer TFLite model size : {int8_quant_size_mb:.2f} MB")
    print("Note: The file sizes are similar because in both models the weights are stored as int8.")

    # --- NEW CODE ENDS HERE ---


if __name__ == "__main__":
    main()
