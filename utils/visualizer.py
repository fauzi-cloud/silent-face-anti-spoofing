import matplotlib.pyplot as plt

def plot_metrics(history):
    plt.plot(history["loss"], label="Loss")
    plt.plot(history["accuracy"], label="Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
