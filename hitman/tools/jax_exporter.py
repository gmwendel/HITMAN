"""
JAX Exporter Module for HITMAN

Automatically extracts neural network weights from trained TensorFlow/Keras models 
and packs them into a strictly structured .npz file designed for consumption by 
pure-JAX inference engines like `sbi_inference`.
"""

import os
import numpy as np
import tensorflow as tf

def extract_layers(model, model_prefix, weights_dict):
    """
    Recursively extracts weights from all layers in a given tf.keras.Model.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            # Recurse into nested models
            extract_layers(layer, f"{model_prefix}_{layer.name}", weights_dict)
        else:
            weights = layer.get_weights()
            if weights:
                for i, w in enumerate(weights):
                    weights_dict[f"{model_prefix}_{layer.name}_{i}"] = w
            
            # Special handling for specific layer types that don't expose 
            # state natively through get_weights() in all Keras versions
            if "normalization" in layer.name.lower():
                try:
                    weights_dict[f"{model_prefix}_{layer.name}_mean"] = layer.mean.numpy()
                    weights_dict[f"{model_prefix}_{layer.name}_var"] = layer.variance.numpy()
                except Exception as e:
                    pass

def export_to_jax(network_dir: str, out_filename: str = "jax_weights.npz"):
    """
    Loads trained ShapeNet and AcceptanceNet from the given directory,
    extracts all weights, and saves them to a compressed .npz archive.
    
    Args:
        network_dir (str): Directory containing 'ShapeNet' and 'AcceptanceNet' SavedModels.
        out_filename (str): Name of the output .npz file to save in the same directory.
    """
    print(f"--- Exporting TF models to JAX-compatible format ---")
    
    # Standard custom objects used in HITMAN
    def mish(x): return x * tf.math.tanh(tf.math.softplus(x))
    def gelu_approx(x): return tf.nn.gelu(x, approximate=True)
    custom_objects = {'mish': mish, 'gelu_approx': gelu_approx}

    shape_path = os.path.join(network_dir, "ShapeNet")
    acc_path = os.path.join(network_dir, "AcceptanceNet")
    hitnet_path = os.path.join(network_dir, "hitnet")
    chargenet_path = os.path.join(network_dir, "chargenet")
    poisson_path = os.path.join(network_dir, "poisson_chargenet")
    
    out_file = os.path.join(network_dir, out_filename)

    weights_dict = {}
    
    if os.path.exists(shape_path):
        print("Exporting ShapeNet weights...")
        shape_model = tf.keras.models.load_model(shape_path, custom_objects=custom_objects, compile=False)
        extract_layers(shape_model, "shape", weights_dict)
    elif os.path.exists(hitnet_path):
        print("Exporting HitNet weights...")
        hit_model = tf.keras.models.load_model(hitnet_path, custom_objects=custom_objects, compile=False)
        extract_layers(hit_model, "shape", weights_dict) # Maps generically to 'shape' for inference
    else:
        print(f"Warning: No ShapeNet or HitNet found.")

    if os.path.exists(acc_path):
        print("Exporting AcceptanceNet weights...")
        acc_model = tf.keras.models.load_model(acc_path, custom_objects=custom_objects, compile=False)
        extract_layers(acc_model, "acc", weights_dict)
    elif os.path.exists(poisson_path):
        print("Exporting Poisson ChargeNet weights...")
        poisson_model = tf.keras.models.load_model(poisson_path, custom_objects=custom_objects, compile=False)
        extract_layers(poisson_model, "acc", weights_dict) # Maps generically to 'acc' for inference
    elif os.path.exists(chargenet_path):
        print("Exporting Standard ChargeNet weights...")
        charge_model = tf.keras.models.load_model(chargenet_path, custom_objects=custom_objects, compile=False)
        extract_layers(charge_model, "acc", weights_dict)
    else:
        print(f"Warning: No AcceptanceNet or ChargeNet found.")

    if len(weights_dict) > 0:
        np.savez_compressed(out_file, **weights_dict)
        print(f"Successfully saved {len(weights_dict)} weight matrices to {out_file}")
    else:
        print("No weights extracted. JAX export failed.")
