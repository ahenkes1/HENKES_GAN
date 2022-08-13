"""Code of the publication 'Three-dimensional microstructure generation using
generative adversarial neural networks in the context of continuum
micromechanics' published in
https://doi.org/10.1016/j.cma.2022.115497
by Alexander Henkes and Henning Wessels from TU Braunschweig.
This code utilizes JIT compilation with XLA.
Use the following command to use it:
'XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.2 python3 main.py'.
Be sure to point to the correct cuda path!
"""
import argparse
import logging
import os
import sys
import tensorflow as tf
import time
import random
import numpy as np
from tensorflow.python.framework import random_seed


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # 0:ALL, 1:INFO, 2:WARNING, 3:ERROR


timestr = time.strftime("%d_%m_%Y-%H_%M_%S")

logger = tf.get_logger()
logger.setLevel(logging.INFO)

# Seeds
# Python RNG
random.seed(42)
# Numpy RNG
np.random.seed(42)
# TF RNG
random_seed.set_seed(42)

print(f"\nTensorflow version: {tf.__version__}")


def gpu(parser_args):
    """Initialize GPUs."""
    MIXED = bool(parser_args.mixed)

    print(f"\nUse mixed precision: {str(MIXED)}")

    if MIXED:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.backend.set_floatx("float32")

    GPUS = parser_args.gpus
    tf.config.set_soft_device_placement(True)

    print(f"\n{80 * '-'}")
    if not tf.config.list_physical_devices("GPU"):
        input("No GPU found. Do you want to proceed?")
    else:
        print(f"GPU Available: " f"{tf.config.list_physical_devices('GPU')}")

    if GPUS == 1:
        physical_devices = tf.config.list_physical_devices("GPU")
        dist_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        tf.config.set_visible_devices(physical_devices[-1], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")

    else:
        physical_devices = tf.config.list_physical_devices("GPU")
        dist_strategy = tf.distribute.MirroredStrategy()
        tf.config.set_visible_devices(physical_devices[-GPUS:], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")

    print(
        f"Physical GPUs: {len(physical_devices)}, "
        f"Logical GPUs: {len(logical_gpus)}"
        f"\nRunning on {len(logical_gpus)} GPU.\n"
        f"Distribution strategy: {dist_strategy}\n"
    )

    return dist_strategy


def main(parser_args, strategy):
    """Main function for MicroGAN."""
    import data_generation
    import gan
    import plot

    SAVE_PATH = "save_files/" + timestr
    try:
        os.mkdir(SAVE_PATH)
        os.mkdir(SAVE_PATH + "/figs/")
        os.mkdir(SAVE_PATH + "/logs/")
        os.mkdir(SAVE_PATH + "/models/")
    except OSError:
        print(f"{80 * '-'}")
        print("Creation of the directory %s failed" % SAVE_PATH)
        sys.exit("DIRECTORY WRONG!")
    else:
        print(f"{80 * '-'}")
        print("Successfully created the directory %s " % SAVE_PATH)
        print("Successfully created the directory %s " % SAVE_PATH + "/figs/")
        print("Successfully created the directory %s " % SAVE_PATH + "/logs/")
        print(
            "Successfully created the directory %s " % SAVE_PATH + "/models/"
        )

    BATCH_SIZE = int(parser_args.batch)
    LATENT = int(parser_args.latent)
    VOXEL = int(parser_args.voxel)
    SAMPLE_SIZE = int(parser_args.samples)
    RADIUS = parser_args.radius
    VOLUME_FRAC = parser_args.volfrac
    LR_D = parser_args.lrd
    LR_G = parser_args.lrg
    ITER = int(parser_args.iter)
    FILTER_G = parser_args.filter_g
    FILTER_D = parser_args.filter_d
    KERNEL = parser_args.kernel
    GP_WEIGHT = parser_args.gpw
    CF_WEIGHT = 1.0
    MIXED = bool(parser_args.mixed)

    data = data_generation.generate_dataset(
        samples=SAMPLE_SIZE,
        batch_size=BATCH_SIZE,
        image_size=VOXEL,
        radius=RADIUS,
        inclusion_fraction=VOLUME_FRAC,
    )
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    dataset = data["dataset"]
    dataset = dataset.with_options(options)
    dataset = strategy.experimental_distribute_dataset(dataset)
    normalizer = data["normalizer"]
    mean = normalizer.mean
    std = tf.sqrt(normalizer.variance)
    try:
        tf.debugging.check_numerics(mean, message="Mean")
    except Exception as e:
        sample_lst = []
        for _ in range((1000 // BATCH_SIZE)):
            sample = next(iter(dataset))
            sample_lst.append(sample)

        mean = tf.math.reduce_mean(sample_lst)
        std = tf.math.reduce_std(sample_lst)
        print(
            f"\n\n{e.message}\nChanging to mean = {mean} and "
            f"to standard deviation = {std}\n\n"
        )

    if MIXED:
        mean = tf.cast(mean, tf.float32)
        std = tf.cast(std, tf.float32)

    iterator = iter(dataset)
    maximum = None
    minimum = None
    for _ in range(1):
        sample = next(iterator)
        maximum = tf.cast(tf.reduce_max(sample), dtype=tf.float32)
        minimum = tf.cast(tf.reduce_min(sample), dtype=tf.float32)

    maximum = (maximum - mean) / std
    minimum = (minimum - mean) / std

    with strategy.scope():
        generator = gan.make_generator(
            input_shape=(LATENT,),
            image_size=VOXEL,
            filter_start=FILTER_G,
            kernel=KERNEL,
            out_max=maximum,
            out_min=minimum,
        )

    # Does not work with Docker
    # tf.keras.utils.plot_model(
    #     generator,
    #     to_file=SAVE_PATH + "/models/generator.png",
    #     show_shapes=True,
    # )

    with strategy.scope():
        discriminator = gan.make_discriminator_growing(
            input_shape=(VOXEL, VOXEL, VOXEL, 1),
            filter_start=FILTER_D,
            kernel=KERNEL,
        )

    # Does not work with Docker
    # tf.keras.utils.plot_model(
    #     discriminator,
    #     to_file=SAVE_PATH + "/models/discriminator.png",
    #     show_shapes=True,
    # )

    print(f"{80 * '-'}\nStart saving hyperparameters...")
    hyperparameter = str(
        f"BATCH SIZE: {BATCH_SIZE}\n"
        f"LATENT SIZE: {LATENT}\n"
        f"VOXEL SIZE: {VOXEL}\n"
        f"SAMPLE_SIZE: {SAMPLE_SIZE}\n"
        f"RADIUS: {RADIUS}\n"
        f"VOLUME FRACTION: {VOLUME_FRAC}\n"
        f"LEARNING RATE DISCRIMINATOR: {LR_D}\n"
        f"LEARNING RATE GENERATOR: {LR_G}\n"
        f"ITERATIONS: {ITER}\n"
        f"KERNEL: {KERNEL}\n"
        f"FILTER GENERATOR: {FILTER_G}\n"
        f"FILTER DISCRIMINATOR: {FILTER_D}\n"
        f"GP WEIGHT: {GP_WEIGHT}\n"
        f"CF WEIGHT: {CF_WEIGHT}\n"
    )
    with open(SAVE_PATH + "/models/hyperparameter.txt", "w") as f:
        f.write(hyperparameter)
    print("Saving hyperparameters finished!")

    with strategy.scope():
        _ = gan.train(
            dataset=dataset,
            generator=generator,
            discriminator=discriminator,
            z_dim=LATENT,
            batch_size=BATCH_SIZE,
            discriminator_learning_rate=LR_D,
            generator_learning_rate=LR_G,
            iterations=ITER,
            path=SAVE_PATH,
            mean=mean,
            std=std,
            gp_weight=GP_WEIGHT,
            cf_weight=CF_WEIGHT,
            distribution_strategy=strategy,
        )

    z = tf.random.uniform([100, LATENT], minval=-1, maxval=1)
    prediction = generator.predict(z)

    cf = tf.size(prediction[prediction > 0]) / tf.size(prediction)

    plot.plot_prediction(
        path=SAVE_PATH,
        prediction=prediction,
        cf=cf,
        suffix="uniform",
        binary=False,
    )

    z = tf.random.normal([100, LATENT], mean=0, stddev=1)
    prediction = generator.predict(z)

    cf = tf.size(prediction[prediction > 0]) / tf.size(prediction)

    plot.plot_prediction(
        path=SAVE_PATH,
        prediction=prediction,
        cf=cf,
        suffix="normal_1",
        binary=False,
    )

    z = tf.random.normal([100, LATENT], mean=0, stddev=10)
    prediction = generator.predict(z)

    cf = tf.size(prediction[prediction > 0]) / tf.size(prediction)

    plot.plot_prediction(
        path=SAVE_PATH,
        prediction=prediction,
        cf=cf,
        suffix="normal_10",
        binary=False,
    )

    print(f"{80 * '-'}\nStart saving models...\n")
    generator.save(SAVE_PATH + "/models/generator")
    discriminator.save(SAVE_PATH + "/models/discriminator")
    print("\nSaving models finished!\n")

    plot.create_tif(path=SAVE_PATH, data=prediction)

    return None


def get_input():
    long_description = str(
        "Code of the publication 'Three-dimensional microstructure generation "
        "using generative adversarial neural networks in the context of "
        "continuum micromechanics' published in "
        "https://doi.org/10.1016/j.cma.2022.115497 by "
        "Alexander Henkes and Henning Wessels from TU Braunschweig. "
        "This code utilizes JIT compilation with XLA. Use the following "
        "command to use it: 'XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local"
        "/cuda-11.2 python3 main.py'. Be sure to point to the correct cuda "
        "path!"
    )

    parser = argparse.ArgumentParser(
        description=long_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--iter",
        default=20000,
        type=float,
        help="Number of training iterations",
    )

    parser.add_argument(
        "--lrd",
        default=1e-4,
        type=float,
        help="Learning rate of discriminator",
    )

    parser.add_argument(
        "--lrg",
        default=5e-5,
        type=float,
        help="Learning rate of generator",
    )

    parser.add_argument(
        "--gpw", default=10.0, type=float, help="Weight for gradient penalty."
    )

    parser.add_argument("--batch", default=8, type=int, help="Batch size")

    parser.add_argument(
        "--volfrac",
        default=0.2,
        type=float,
        help="Volume fraction of inclusions",
    )

    parser.add_argument(
        "--radius",
        default=4,
        type=int,
        help="Radius of inclusions.",
    )

    parser.add_argument(
        "--samples",
        default=1000,
        type=int,
        help="Number of samples",
    )

    parser.add_argument(
        "--latent", default=128, type=int, help="Latent vector size"
    )

    parser.add_argument(
        "--voxel",
        default=32,
        type=int,
        help="Number of voxels per axis",
    )

    parser.add_argument(
        "--kernel",
        default=3,
        type=int,
        help="Size of convolutional kernels.",
    )

    parser.add_argument(
        "--filter_g",
        default=32,
        type=int,
        help="Starting size of filters for generator.",
    )

    parser.add_argument(
        "--filter_d",
        default=16,
        type=int,
        help="Starting size of filters for discriminator.",
    )

    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs.")

    parser.add_argument(
        "--mixed", default=1, type=int, help="Activate mixed precision."
    )

    parser.add_argument(
        "--debug", default=0, type=int, help="Activate eager debug mode."
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_input()
    if args.debug == 1:
        tf.config.run_functions_eagerly(True)
        tf.print(
            f"\n\n{79 * '='}\n{20 * ' '}DEBUG MODE ON!\n" f"{79 * '='}\n\n\n"
        )
    else:
        print("Graph Mode")

    distributed_strategy = gpu(parser_args=args)
    main(parser_args=args, strategy=distributed_strategy)
