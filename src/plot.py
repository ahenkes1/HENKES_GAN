"""Plotting routine for GAN."""
import matplotlib.pyplot as plt
import tensorflow as tf
import tifffile
import sys

sys.path.append(r"../results")
print(sys.path)


def plot_loss(path=None, loss_values=None):
    print(f"{80 * '-'}\nStart plotting loss...\n")

    loss_g = []
    loss_d = []
    loss_d_abs = []
    loss_t = []
    for loss in loss_values:
        loss_g.append(loss[0])
        loss_d.append(loss[1])
        loss_d_abs.append(abs(loss[1]))
        loss_t.append(loss[2])

    plt.figure(1)
    plt.plot(loss_g, label="G")
    plt.plot(loss_d, label="D")
    plt.plot(loss_t, label="T")
    plt.legend()
    plt.savefig(fname=path + "/figs/loss_all.pdf")

    plt.figure(2)
    plt.plot(loss_g, label="G")
    plt.legend()
    plt.savefig(fname=path + "/figs/loss_G.pdf")

    plt.figure(3)
    plt.plot(loss_d, label="D")
    plt.legend()
    plt.savefig(fname=path + "/figs/loss_D.pdf")

    plt.figure(4)
    plt.plot(loss_t, label="T")
    plt.legend()
    plt.savefig(fname=path + "/figs/loss_T.pdf")

    plt.figure(5)
    plt.semilogy(loss_d_abs, label="D_abs")
    plt.legend()
    plt.savefig(fname=path + "/figs/log_abs_loss_D.pdf")

    print(f"Plotting loss finished!")
    print(f"{80 * '-'}\n")
    return None


def plot_prediction(
    path=None, prediction=None, cf=None, binary=False, suffix=None
):
    print(f"{80 * '-'}\nStart plotting predictions...\n")

    prediction = prediction[:, :, :, :, 0]

    if binary:
        prediction[prediction > 0] = 1
        prediction[prediction <= 0] = 0

    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(20, 20))

    i = 0
    for row in axes:
        for ax in row:
            ax.imshow(prediction[i, 16, :, :], cmap=plt.get_cmap("gray"))
            i += 1
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        fname=(
            path
            + "/figs/prediction_"
            + str(cf.numpy())
            + "_"
            + str(suffix)
            + ".pdf"
        )
    )

    print(f"Plotting predictions finished!")
    print(f"{80 * '-'}\n")
    # plt.show()
    return None


def create_tif(path=None, data=None):
    """Create .tif file of generated microstructures.

    Creates .tif files of the GAN generated microstructures for medical
    imaging software.

    Args:
        - path: The save path of the file.

    Returns:
    """
    for i in range(10):
        tifffile.imsave(
            (path + "/figs/prediction_" + str(i) + ".tif"), data[i, :, :, :, 0]
        )
    return None


def load_model(path=None):
    """Load trained and saved generator.

    Args:
        - path: Save-path for generator model.

    Returns:
    """
    model = tf.keras.models.load_model(path)
    return model
