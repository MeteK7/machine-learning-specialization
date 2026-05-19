import os

folders = [
    "machine-learning-specialization",

    "machine-learning-specialization/01_supervised_learning/linear_regression",
    "machine-learning-specialization/01_supervised_learning/logistic_regression",
    "machine-learning-specialization/01_supervised_learning/gradient_descent",

    "machine-learning-specialization/02_advanced_learning_algorithms/neural_networks_basics",
    "machine-learning-specialization/02_advanced_learning_algorithms/activation_functions",

    "machine-learning-specialization/02_advanced_learning_algorithms/softmax",

    "machine-learning-specialization/02_advanced_learning_algorithms/image_classifier_mnist/results",

    "machine-learning-specialization/02_advanced_learning_algorithms/overfitting_regularization",
    "machine-learning-specialization/02_advanced_learning_algorithms/backpropagation",

    "machine-learning-specialization/03_unsupervised_learning/k_means",
    "machine-learning-specialization/03_unsupervised_learning/anomaly_detection",

    "machine-learning-specialization/assets/images",
    "machine-learning-specialization/assets/diagrams",
]

files = [
    "machine-learning-specialization/README.md",

    "machine-learning-specialization/02_advanced_learning_algorithms/softmax/softmax_lab_notes.ipynb",
    "machine-learning-specialization/02_advanced_learning_algorithms/softmax/softmax_playground.py",

    "machine-learning-specialization/02_advanced_learning_algorithms/image_classifier_mnist/mnist_classifier.ipynb",
    "machine-learning-specialization/02_advanced_learning_algorithms/image_classifier_mnist/mnist_classifier.py",
    "machine-learning-specialization/02_advanced_learning_algorithms/image_classifier_mnist/README.md",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    with open(file, "w") as f:
        f.write("")

print("Project structure created successfully!")