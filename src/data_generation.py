"""Generate input data for GAN."""

from datetime import datetime
import tensorflow as tf
import tqdm


@tf.function
def create_bin_sphere(matrix_size, center, radius):
    x = tf.cast(tf.linspace(0, matrix_size - 1, matrix_size), dtype=tf.float32)
    y = tf.cast(tf.linspace(0, matrix_size - 1, matrix_size), dtype=tf.float32)
    z = tf.cast(tf.linspace(0, matrix_size - 1, matrix_size), dtype=tf.float32)
    center = tf.cast(center, dtype=tf.float32)
    coords = tf.meshgrid(x, y, z)
    distance = tf.sqrt(
        (coords[0] - center[0]) ** 2
        + (coords[1] - center[1]) ** 2
        + (coords[2] - center[2]) ** 2
    )
    return tf.cast(distance <= radius, tf.int32)


@tf.function
def spherical(inclusion_fraction=0.3, image_size=32, radius=4):
    """Create spherical inclusion.

    Generate random spherical inclusion in a three dimensional unit cell.
    Args:
        - inclusion_fraction: The volume fraction of the inclusion phase.
        - no_of_voxels: Dimensions of the unit cell.

    Returns:
        microstructure: Voxelized microstructure with spherical inclusion.
    """
    no_of_voxels = image_size
    microstructure = tf.zeros(
        shape=(no_of_voxels, no_of_voxels, no_of_voxels), dtype=tf.int32
    )

    no_of_spheres = 0
    real_cf = None
    while (
        tf.size(microstructure[microstructure == 1]) / tf.size(microstructure)
        <= inclusion_fraction
    ):
        r = radius
        sphere_center = tf.random.uniform(
            minval=0,
            maxval=no_of_voxels - 1,
            shape=(3, 1),
            dtype=tf.int32,
        )
        sphere = create_bin_sphere(no_of_voxels, sphere_center, r)
        if (
            tf.reduce_any(tf.cast(microstructure[sphere == 1], tf.bool))
            is False
        ):
            pass
        else:

            microstructure = tf.where(
                condition=tf.equal(sphere, 1), x=1, y=microstructure
            )

            no_of_spheres += 1
            real_cf = tf.convert_to_tensor(
                [
                    tf.size(microstructure[microstructure == 1])
                    / tf.size(microstructure)
                ],
                dtype=tf.float32,
            )

    microstructure = tf.reshape(
        microstructure, shape=(no_of_voxels, no_of_voxels, no_of_voxels, 1)
    )

    real_cf = tf.reshape(real_cf, shape=(1,))

    microstructure = tf.cast(x=microstructure, dtype=tf.float32)

    return {"microstructure": microstructure, "volume_fraction": real_cf}


def generate_dataset(
    samples=None,
    batch_size=None,
    image_size=32,
    radius=4,
    inclusion_fraction=0.3,
):
    """Generate a tf.data.Dataset for efficient training.

    The dataset is shuffled, batched and prefetched.

    Args:
        - samples: Total number of samples.

    Returns:
        dataset: tf.data.Dataset of microstructures.
        batch_size: Batch size for the dataset.

    """
    print(f"\n{80 * '-'}\nGenerating dataset ...")
    start_time_generation = datetime.now()
    dataset = []

    print("Radius will be overwritten ...")
    if image_size == 32:
        radius = 4
    elif image_size == 64:
        radius = 8
    elif image_size == 128:
        radius = 16
    else:
        print("Image size not supported!")

    tqdm_iter = None
    for tqdm_iter in tqdm.tqdm(range(samples)):
        dataset.append(
            spherical(
                image_size=image_size,
                inclusion_fraction=inclusion_fraction,
                radius=radius,
            )["microstructure"]
        )

    dataset = tf.data.Dataset.from_tensor_slices(tensors=dataset)
    dataset = dataset.shuffle(buffer_size=samples).repeat()
    dataset = dataset.batch(
        batch_size=batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(data=dataset, steps=samples)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    print(f"Dataset finished! Created {tqdm_iter + 1} samples.")
    print(f"Generation time: {datetime.now() - start_time_generation}")
    print(f"\n{80 * '-'}")

    return {"dataset": dataset, "normalizer": normalizer}
