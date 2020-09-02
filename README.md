# One-Shot-Learning-and-Siamese-Network
One-Shot-Learning-and-Siamese-Network


One-shot learning allows deep learning algorithms to measure the similarity and difference between two images.

1?Take an input and extract its embedding (mapping to a vector of continuous numbers) by passing it through a neural network.
2>Repeat step 1 with a different input.
3>Compare the two embeddings to check whether there is a similarity between the two data points. These two embeddings act as a latent feature representation of the data. In our case, images with the same person should have similar embeddings.

note: download the facenet.h5 before using this project.

#Reference https://github.com/susantabiswas/FaceRecog
