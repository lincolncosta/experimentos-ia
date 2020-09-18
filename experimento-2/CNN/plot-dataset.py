import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


num = 10
images = X_train[:num]
labels = Y_train[:num]

num_row = 2
num_col = 5

# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col, 2*num_row))
for i in range(num):
    ax = axes[i//num_col, i % num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Amostra: {}'.format(labels[i]))
plt.tight_layout()
plt.savefig('mnist-samples.png')
