import argparse

import mlx.core as mx
import mlx.optimizers as optim

from data import fol
from models import trm
from training.trainer import Trainer

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--dataset",
    type=str,
    default="fol",
    choices=["mnist", "cifar10", "fol"],
    help="dataset to use",
)
parser.add_argument("-b", "--batch_size", type=int, default=1024, help="batch size")
parser.add_argument("-e", "--epochs", type=int, default=15, help="number of epochs")
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")


def main(args):
    if args.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(args.seed)

    if args.dataset == "mnist":
        from data.vision import mnist
        train_data, test_data, meta = mnist(args.batch_size)
    elif args.dataset == "cifar10":
        from data.vision import cifar10
        train_data, test_data, meta = cifar10(args.batch_size)
    elif args.dataset == "fol":
        train_data, test_data, meta = fol.fol_dataset(args.batch_size, steps_per_epoch=20)
    else:
        raise NotImplementedError(f"{args.dataset=} is not implemented.")
    
    # n_inputs = next(train_data)["image"].shape[1:] # Not needed for FOL or can be derived differently
    # train_data.reset() # Generators might not have reset


    config = trm.ModelConfig(
        vocab_size=meta.get("vocab_size", 256), # Default or from meta
        depth=2,
        dim=64,
        heads=4,
        n_outputs=2, # Binary classification for Entailment
    )
    model = trm.Model(config)
    model.summary()

    n_steps = args.epochs * meta["steps_per_epoch"]
    n_linear = n_steps * 0.10
    linear = optim.linear_schedule(0, args.lr, steps=n_linear)
    cosine = optim.cosine_decay(args.lr, n_steps - n_linear, 0)
    lr_schedule = optim.join_schedules([linear, cosine], [n_linear])
    optimizer = optim.AdamW(
        learning_rate=lr_schedule, betas=(0.9, 0.999), weight_decay=0.01
    )

    manager = Trainer(model, optimizer)
    manager.train(train_data, val=test_data, epochs=args.epochs)

    #! plotting
    import matplotlib.pyplot as plt
    print("Saving model to fol_model.safetensors")
    model.save_weights("fol_model.safetensors")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    lw = 2
    ax.plot(mx.array(manager.train_acc_trace) * 100, label="train", color="r", lw=lw)
    ax.plot(mx.array(manager.val_acc_trace) * 100, label="val", color="b", lw=lw)
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    # plt.show() # blocking


if __name__ == "__main__":
    main(parser.parse_args())
