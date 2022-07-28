"""Create neural networks for GAN."""

import tensorflow as tf
import tqdm


def res_block_discriminator(
    input_layer=None,
    filters=None,
    pooling=None,
    kernel=None,
    initializer=None,
    strides=2,
):
    """Residual block for ResNet in the discriminator.

    Residual block for ResNet using 3D convolution and skip connections.

    Args:
        - input_layer: The input from preceeding layer.
        - filters: Number of filters for convolutional layers.
        - pooling: Activate average (mean) pooling in block.

    Returns:
        - output_layer: Output of residual block.
    """
    INITIALIZER = initializer

    residual = tf.keras.layers.Conv3D(
        filters=filters,
        kernel_size=1,
        padding="valid",
        strides=strides,
        kernel_initializer=INITIALIZER,
    )(input_layer)

    x = tf.keras.layers.Conv3D(
        filters=filters,
        kernel_size=kernel,
        padding="same",
        strides=strides,
        kernel_initializer=INITIALIZER,
    )(input_layer)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv3D(
        filters=filters,
        kernel_size=kernel,
        padding="same",
        strides=1,
        kernel_initializer=INITIALIZER,
    )(x)

    x = tf.keras.layers.add([residual, x])
    x = tf.keras.layers.LeakyReLU()(x)

    if pooling:
        x = tf.keras.layers.AveragePooling3D()(x)

    output_layer = x
    return output_layer


def make_generator(
    input_shape,
    image_size,
    filter_start=4,
    kernel=3,
    out_max=None,
    out_min=None,
):
    """Creates generator for GAN.

    The generator is a reversed convolutional neural network, mapping random
    noise to image space.

    Args:
        - input_shape: The input shape coresponding to the random noise vector.

    Returns:
        - generator: A tf.keras model.
    """
    START = filter_start
    FILTER = None
    no_layers = None

    if image_size == 32:
        no_layers = 3
        FILTER = [START * 8, START * 4, START * 2, START]

    elif image_size == 64:
        no_layers = 4
        FILTER = [START * 16, START * 8, START * 4, START * 2, START]

    elif image_size == 128:
        no_layers = 5
        FILTER = [
            START * 32,
            START * 16,
            START * 8,
            START * 4,
            START * 2,
            START,
        ]
    else:
        print("Image size not supported!")

    KERNEL = kernel
    INITIALIZER = "orthogonal"

    input_generator = tf.keras.layers.Input(shape=input_shape)

    xg = input_generator

    for w_layers in range(8):
        xg = tf.keras.layers.Dense(
            units=input_shape[0],
            use_bias=True,
            activation="leaky_relu",
            kernel_initializer=INITIALIZER,
        )(xg)

    xg = tf.keras.layers.Dense(
        units=2 * 2 * 2 * FILTER[0],
        use_bias=True,
        activation="leaky_relu",
        kernel_initializer=INITIALIZER,
    )(xg)
    w = tf.keras.layers.Reshape((2, 2, 2, FILTER[0]))(xg)
    xg = w

    for block in range(no_layers):
        STRIDES = int(2 ** (block + 1))
        xg = tf.keras.layers.Conv3DTranspose(
            filters=FILTER[block + 1],
            kernel_size=KERNEL,
            strides=2,
            padding="same",
            use_bias=True,
            activation="leaky_relu",
            kernel_initializer=INITIALIZER,
        )(xg)

        x_w = tf.keras.layers.Conv3DTranspose(
            filters=FILTER[block + 1],
            kernel_size=1,
            strides=STRIDES,
            padding="same",
            activation="leaky_relu",
            use_bias=True,
            kernel_initializer=INITIALIZER,
        )(w)
        xg = tf.keras.layers.add([x_w, xg])

    xg = tf.keras.layers.Conv3DTranspose(
        filters=1,
        kernel_size=KERNEL,
        strides=2,
        padding="same",
        activation="linear",
        use_bias=True,
        kernel_initializer=INITIALIZER,
    )(xg)

    outputs = tf.keras.layers.Activation("tanh", dtype="float32", name="tanh")(
        xg
    )

    out_max = tf.constant(
        out_max,
        dtype="float32",
    )
    out_min = tf.constant(
        out_min,
        dtype="float32",
    )
    one = tf.constant(
        1.0,
        dtype="float32",
    )
    two = tf.constant(
        2.0,
        dtype="float32",
    )

    outputs = (outputs + one) * ((out_max - out_min) / two) + out_min

    generator = tf.keras.Model(
        inputs=input_generator, outputs=outputs, name="generator"
    )
    generator.build(input_shape=(None, input_shape[0]))
    generator.summary(line_length=120)
    print()

    return generator


def make_discriminator_growing(input_shape, filter_start=4, kernel=3):
    """Creates discriminator for GAN.

    The discriminator or critique is a convolutional neural network,
    classifiying images based on a dataset, whether they are real or fake.

    Args:
        - input_shape: The input shape coresponding to the random noise vector.
        - filter_start: The starting filter size, grows by 2^no_layers.
        - kernel: Kernel size for convolutional layers.

    Returns:
        - discriminator: A tf.keras model.
    """

    START = filter_start
    FILTER = None
    no_layers = None
    image_size = input_shape[0]
    if image_size == 32:
        no_layers = 4
        FILTER = [START, START * 2, START * 4, START * 8, START * 16]

    elif image_size == 64:
        no_layers = 5
        FILTER = [
            START,
            START * 2,
            START * 4,
            START * 8,
            START * 16,
            START * 32,
        ]

    elif image_size == 128:
        no_layers = 6
        FILTER = [
            START,
            START * 2,
            START * 4,
            START * 8,
            START * 16,
            START * 32,
            START * 64,
        ]

    else:
        print("Image size not supported!")

    KERNEL = kernel
    INITIALIZER = "orthogonal"

    input_discriminator = tf.keras.layers.Input(
        shape=(input_shape[0], input_shape[1], input_shape[2], 1)
    )

    xd = input_discriminator

    res_block = None
    for res_block in range(no_layers):
        xd = res_block_discriminator(
            input_layer=xd,
            filters=FILTER[res_block],
            kernel=KERNEL,
            strides=1,
            initializer=INITIALIZER,
            pooling=True,
        )

    xd = res_block_discriminator(
        input_layer=xd,
        filters=FILTER[res_block + 1],
        kernel=KERNEL,
        strides=1,
        initializer=INITIALIZER,
        pooling=False,
    )

    xd = tf.keras.layers.Flatten()(xd)

    xd = tf.keras.layers.Dense(units=1, kernel_initializer=INITIALIZER)(xd)

    outputs = tf.keras.layers.Activation("linear", dtype="float32")(xd)

    discriminator = tf.keras.Model(
        inputs=input_discriminator, outputs=outputs, name="discriminator"
    )

    discriminator.build(
        input_shape=(None, input_shape[0], input_shape[1], input_shape[2], 1)
    )

    discriminator.summary(line_length=150)

    return discriminator


def get_loss_fn():
    """Wasserstein loss function.

    Creates the Wasserstein loss function without gradient penalty for GAN
    training. Without gradient penalty, the Lipschitz-continuity is likewise
    violated, rendering it meaningless. Otherwise, the Wasserstein loss
    function gives an approximation of the distance of two probability
    functions.

    Args:

    Returns:
        - d_loss_fn: Discriminator part of the loss function.
        - g_loss_fn: Generator part of the loss function.
    """

    def d_loss_fn(real_logits, fake_logits):
        """Discriminator Wasserstein loss.

        Args:
            - real_logits: Prediction for real microstructures.
            - fake_logits: Prediction for fake microstructures.

        Returns:
            - d_loss: Discriminator loss function.
        """
        d_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
        return d_loss

    def g_loss_fn(fake_logits):
        """Generator Wasserstein loss.

        Args:
            - fake_logits: Prediction for fake microstructures.

        Returns:
            - g_loss: Generator loss function.
        """
        g_loss = -tf.reduce_mean(fake_logits)
        return g_loss

    return d_loss_fn, g_loss_fn


def gradient_penalty(
    discriminator, real_images, fake_images, batch_size, gp_weight
):
    """Gradient penalty term for Wasserstein loss.

    The gradient penalty term ensures Lipschitz-continuity of the GAN to
    render the Wasserstein-loss meaningfull.

    Args:
        - generator: Generator network.
        - real_images: Real images from the training dataset.
        - fake_images: Fake images generated by the generator.
        - batch_size: Training batch size.
        - gp_weight: Weight term for penalty term.

    Returns:
        - gradient_penalty_term: Gradient penalty term for loss function.
    """
    real_images = tf.cast(real_images, tf.float32)
    fake_images = tf.cast(fake_images, tf.float32)
    alpha = tf.random.uniform([batch_size, 1, 1, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    inter = real_images + (alpha * diff)
    with tf.GradientTape() as tape:
        tape.watch(inter)
        predictions = discriminator(inter, training=True)
    gradients = tape.gradient(predictions, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3, 4]))
    gradient_penalty_term = tf.reduce_mean((slopes - 1.0) ** 2)
    return gradient_penalty_term * gp_weight


def volume_fraction_penalty(fake_images, mean, cf_weight):
    """Calculate inclusion volume fraction discrepancy.

    Args:
        - fake_images: Images produced by generator.
        - cf_weight: Weight term for penalty term.

    Returns:
        - cf: Volume fraction loss term.
    """
    fake_images = tf.cast(fake_images, tf.float32)
    mean = tf.cast(mean, tf.float32)
    cf = tf.reduce_mean(
        tf.where(
            condition=(fake_images > 0),
            x=tf.ones(shape=fake_images.shape),
            y=tf.zeros(shape=fake_images.shape),
        )
    )

    cf_loss = tf.square((mean - cf) * 100.0)
    cf_loss = tf.reduce_mean(cf_loss) * cf_weight
    return {"cf_loss": cf_loss, "cf": cf}


def histogram_penalty(ref=None, pred=None):
    """Compare histogram with reference and form penalty factor."""
    histogram_reference = tf.histogram_fixed_width(
        values=ref, value_range=[-2, 2], nbins=5
    )

    histogram_prediction = tf.histogram_fixed_width(
        values=pred, value_range=[-2, 2], nbins=5
    )

    histogram_penalty_square = tf.square(
        histogram_reference - histogram_prediction
    )
    histogram_penalty_mean_square = tf.reduce_mean(histogram_penalty_square)

    return histogram_penalty_mean_square


def train_step(
    strategy=None,
    generator=None,
    discriminator=None,
    real_images=None,
    z_dim=None,
    batch_size=None,
    d_loss_fn=None,
    g_loss_fn=None,
    w_optimizer=None,
    g_optimizer=None,
    d_optimizer=None,
    gp_weight=None,
    cf_weight=None,
    mean=None,
    std=None,
):
    """Single train step for GAN.

    Args:

    Returns:
    """
    if not strategy:
        tf.print("Stragety error!")
    DEVICES = tf.distribute.get_replica_context().strategy.num_replicas_in_sync
    batch_size = batch_size // DEVICES
    GP_WEIGHT = gp_weight
    CF_WEIGHT = cf_weight
    z = tf.random.normal([batch_size, z_dim], mean=0.0, stddev=1.0)

    real_images = (real_images - mean) / std

    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_images = generator(z, training=True)
        real_images = tf.reshape(real_images, fake_images.shape)

        fake_mean = tf.reduce_mean(fake_images)
        fake_std = tf.math.reduce_std(fake_images)

        fake_logits = discriminator(fake_images, training=True)
        real_logits = discriminator(real_images, training=True)

        d_loss_r = d_loss_fn(real_logits, fake_logits)
        g_loss_r = g_loss_fn(fake_logits)

        gp_loss = gradient_penalty(
            discriminator=discriminator,
            real_images=real_images,
            fake_images=fake_images,
            batch_size=batch_size,
            gp_weight=GP_WEIGHT,
        )

        cf_tuple = volume_fraction_penalty(
            fake_images=fake_images, mean=mean, cf_weight=CF_WEIGHT
        )
        cf = cf_tuple["cf"]

        cf_loss = tf.constant(
            value=0.0, dtype=gp_loss.dtype, shape=gp_loss.shape
        )

        d_loss = d_loss_r + gp_loss - cf_loss
        g_loss = g_loss_r + cf_loss

        d_loss_avg = tf.nn.compute_average_loss(
            per_example_loss=d_loss, global_batch_size=batch_size
        )
        g_loss_avg = tf.nn.compute_average_loss(
            per_example_loss=g_loss, global_batch_size=batch_size
        )

        scaled_d_loss = d_optimizer.get_scaled_loss(d_loss_avg)
        scaled_g_loss = g_optimizer.get_scaled_loss(g_loss_avg)

    scaled_d_gradients = d_tape.gradient(
        scaled_d_loss, discriminator.trainable_variables
    )
    scaled_g_gradients = g_tape.gradient(
        scaled_g_loss, generator.trainable_variables
    )

    d_gradients = d_optimizer.get_unscaled_gradients(scaled_d_gradients)
    g_gradients = g_optimizer.get_unscaled_gradients(scaled_g_gradients)

    d_optimizer.apply_gradients(
        zip(d_gradients, discriminator.trainable_variables)
    )
    w_optimizer.apply_gradients(
        zip(g_gradients[:16], generator.trainable_variables[:16])
    )
    g_optimizer.apply_gradients(
        zip(g_gradients[16:], generator.trainable_variables[16:])
    )

    return {
        "g_loss": g_loss,
        "g_loss_r": g_loss_r,
        "d_loss": d_loss,
        "d_loss_r": d_loss_r,
        "gp": gp_loss,
        "cf_loss": cf_loss,
        "cf": cf,
        "fake_mean": fake_mean,
        "fake_std": fake_std,
    }


# if 'jit_compile=True', add:
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.7 python3 main.py
# else, set 'jit_compile=False'
@tf.function(jit_compile=True)
def distributed_train_step(
    strategy=None,
    generator=None,
    discriminator=None,
    real_images=None,
    z_dim=None,
    batch_size=None,
    d_loss_fn=None,
    g_loss_fn=None,
    w_optimizer=None,
    g_optimizer=None,
    d_optimizer=None,
    gp_weight=None,
    cf_weight=None,
    mean=None,
    std=None,
    physics=None,
):
    per_replica_losses = strategy.run(
        train_step,
        args=(
            strategy,
            generator,
            discriminator,
            real_images,
            z_dim,
            batch_size,
            d_loss_fn,
            g_loss_fn,
            w_optimizer,
            g_optimizer,
            d_optimizer,
            gp_weight,
            cf_weight,
            mean,
            std,
        ),
    )

    out = strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
    )
    return out


def train(
    dataset=None,
    generator=None,
    discriminator=None,
    z_dim=None,
    batch_size=None,
    generator_learning_rate=None,
    discriminator_learning_rate=None,
    iterations=None,
    path=None,
    mean=None,
    std=None,
    gp_weight=None,
    cf_weight=None,
    distribution_strategy=None,
    physics=None,
):
    """Training routine for GAN.

    Args:

    Returns:

    """
    strategy = distribution_strategy
    print(f"\n{80 * '-'}\nStart training ...\n")
    d_loss_fn, g_loss_fn = get_loss_fn()

    BETA_1 = 0.9
    CLIPNORM = 1.0

    lr_w = generator_learning_rate * 0.01
    lr_g = generator_learning_rate
    lr_d = discriminator_learning_rate

    with strategy.scope():
        w_optim = tf.keras.optimizers.Nadam(
            learning_rate=lr_w,
            beta_1=BETA_1,
            beta_2=0.999,
            global_clipnorm=CLIPNORM,
        )

        g_optim = tf.keras.optimizers.Nadam(
            learning_rate=lr_g,
            beta_1=BETA_1,
            beta_2=0.999,
            global_clipnorm=CLIPNORM,
        )

        d_optim = tf.keras.optimizers.Nadam(
            learning_rate=lr_d,
            beta_1=BETA_1,
            beta_2=0.999,
            global_clipnorm=CLIPNORM,
        )

        w_optim = tf.keras.mixed_precision.LossScaleOptimizer(w_optim)
        g_optim = tf.keras.mixed_precision.LossScaleOptimizer(g_optim)
        d_optim = tf.keras.mixed_precision.LossScaleOptimizer(d_optim)

    g_loss_raw_metrics = tf.metrics.Mean(name="g_loss_raw")
    d_loss_raw_metrics = tf.metrics.Mean(name="d_loss_raw")
    gp_metrics = tf.metrics.Mean(name="gp")
    cf_metrics = tf.metrics.Mean(name="cf")
    mean_metrics = tf.metrics.Mean(name="mean")
    std_metrics = tf.metrics.Mean(name="std")

    dataset = iter(dataset)

    losses = []

    train_log_dir = path + "/logs/"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    with tqdm.trange(int(iterations)) as pbar:
        for step in pbar:
            real_images = next(dataset)
            train_dic = distributed_train_step(
                strategy=strategy,
                generator=generator,
                discriminator=discriminator,
                real_images=real_images,
                z_dim=z_dim,
                batch_size=batch_size,
                d_loss_fn=d_loss_fn,
                g_loss_fn=g_loss_fn,
                w_optimizer=w_optim,
                g_optimizer=g_optim,
                d_optimizer=d_optim,
                gp_weight=tf.constant(
                    value=gp_weight, shape=(1,), dtype=tf.float32
                ),
                cf_weight=tf.constant(
                    value=cf_weight, shape=(1,), dtype=tf.float32
                ),
                mean=tf.constant(value=mean, shape=(1,), dtype=tf.float32),
                std=tf.constant(value=std, shape=(1,), dtype=tf.float32),
                physics=physics,
            )

            g_loss_r = train_dic["g_loss_r"]
            d_loss_r = train_dic["d_loss_r"]
            gp = train_dic["gp"]
            cf = train_dic["cf"]
            fake_mean = train_dic["fake_mean"]
            fake_std = train_dic["fake_std"]

            glrm = g_loss_raw_metrics(g_loss_r)
            dlrm = d_loss_raw_metrics(d_loss_r)
            gpm = gp_metrics(gp)
            cfm = cf_metrics(cf)
            meanm = mean_metrics(fake_mean)
            stdm = std_metrics(fake_std)

            pbar.set_postfix(
                gr=glrm.numpy(),
                dr=dlrm.numpy(),
                gp=gpm.numpy(),
                cf=cfm.numpy(),
                m=meanm.numpy(),
                s=stdm.numpy(),
            )

            with train_summary_writer.as_default():
                tf.summary.scalar("Dr", d_loss_raw_metrics.result(), step=step)
                tf.summary.scalar("Gr", g_loss_raw_metrics.result(), step=step)
                tf.summary.scalar("GP", gp_metrics.result(), step=step)
                tf.summary.scalar("cf", cf_metrics.result(), step=step)
                tf.summary.scalar("mean", mean_metrics.result(), step=step)
                tf.summary.scalar("std", std_metrics.result(), step=step)

            g_loss_raw_metrics.reset_states()
            d_loss_raw_metrics.reset_states()
            gp_metrics.reset_states()
            cf_metrics.reset_states()

    print(f"\nTraining finished!\n{80 * '-'}")
    return losses
