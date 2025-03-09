import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import torch


class MLP(torch.nn.Module):
    # Define neural network
    def __init__(self, input_dim, output_dim):
        assert isinstance(input_dim, int) and input_dim > 0
        assert isinstance(output_dim, int) and output_dim > 0
        
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 64)
        self.fc4 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.softmax(self.fc4(x), dim=1)
        return x


def stats_model(model_name, x_train, x_test, y_train, y_test):
    '''
    Predict data based on statistical models
    '''
    models = {"SVC":SVC,
              "SVR":SVR,
              "NBC":GaussianNB,
              "RFC":RandomForestClassifier}

    assert model_name in models.keys()
    assert isinstance(x_train, np.ndarray)
    assert isinstance(x_test,  np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test,  np.ndarray)
    
    # Predict Anxiety, Depression, Insomnia, OCD, Effect one by one
    y_pred = []
    for i in range(5):
        model = models[model_name]()
        y_train_col, y_test_col = y_train[:, i], y_test[:, i]
        model.fit(x_train, y_train_col)
        y_pred_col = model.predict(x_test).reshape(-1, 1)
        y_pred.append(y_pred_col)

    # Combine prediction result to ndarray
    y_pred = np.concatenate(y_pred, axis=1)
    return y_pred
        

def nn_model(model_name, x_train, x_test, y_train, y_test):
    '''
    Predict data based on neural network models
    '''
    models = {"MLP": MLP}
    assert model_name in models.keys()
    assert isinstance(x_train, np.ndarray)
    assert isinstance(x_test,  np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test,  np.ndarray)

    # Function for NN training
    def train(model, x_train, y_train, criterion, optimizer, epochs):
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train.reshape(-1))
            loss.backward()
            optimizer.step()

    # Train models and predict Anxiety, Depression, Insomnia, OCD, Effect one by one
    y_pred = []  
    for i in range(5):
        # Get number of dimensions
        input_dim = x_train.shape[1]
        # symptoms have 11 classes, music effect has 3 classes
        output_dim = 11 if i < 4 else 3
        
        # Model, loss function and optimizer
        model = models[model_name](input_dim, output_dim)
        cross_entropy = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.005,
                                    momentum=0.6)

        # Train model
        x_train_tensor = torch.tensor(x_train, dtype=torch.float)
        x_test_tensor  = torch.tensor(x_test,  dtype=torch.float)
        y_train_col_tensor = torch.tensor(y_train[:, i], dtype=torch.long)
        y_test_col_tensor  = torch.tensor(y_test[:, i],  dtype=torch.long)
        train(model, x_train_tensor, y_train_col_tensor, cross_entropy, optimizer, 1000)

        # Predict
        with torch.no_grad():
            y_pred_col = model(x_test_tensor).argmax(dim=1).numpy().reshape(-1,1)
        y_pred.append(y_pred_col)

    # Combine prediction result to ndarray
    y_pred = np.concatenate(y_pred, axis=1)
    return y_pred


def model_metric(model_name, y_test, y_pred):
    '''
    Calculate mean squared error and accuracy.
    '''
    assert model_name in ["SVC", "SVR", "RFC", "NBC", "MLP"]
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)

    y_names = ["Anxiety", "Depression", "Insomnia", "OCD", "Effect"]
    mse, acc = [], []
    for i in range(5):
        mse.append(mean_squared_error(y_test[:, i], y_pred[:, i]))
        acc.append(np.mean(np.round(y_pred[:, i]) == y_test[:, i]) * 100)
        print(f"MSE loss on {y_names[i]:10}: {mse[i]:.4f} \
              \tAccuracy on {y_names[i]:10}: {acc[i]:.2f}%.")

    return (mse, acc)


def draw_scatter_model(model_name, y_test, y_pred):
    '''
    Visualize model results, compare true labels and predicted labels.
    '''
    model_fullnames = {"SVC": "Support Vector Regressor",
                       "SVR": "Support Vector Classifier",
                       "RFC": "Random Forest Classifier",
                       "NBC": "Naive Bayes Classifier",
                       "MLP": "Multi-layer Perceptron"}
    
    colors = {"SVR": "blue",
              "SVC": "orange",
              "RFC": "green",
              "NBC": "red",
              "MLP": "purple"}

    assert model_name in model_fullnames.keys()
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_pred, np.ndarray)

    y_names = ["Anxiety", "Depression", "Insomnia", "OCD", "Effect"]
    fullname = model_fullnames[model_name]
    color = colors[model_name]

    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        min_y, max_y = np.min(y_test[:, i]), np.max(y_test[:, i])
        ax[i].set_title(y_names[i])
        ax[i].plot([min_y, max_y], [min_y, max_y], color="grey", linestyle="--")
        ax[i].scatter(y_pred[:, i], y_test[:, i], color=color, alpha=0.5)

    fig.supxlabel(f"{model_name} Predicted Value")
    fig.supylabel("True Value")
    fig.suptitle(fullname)
    plt.tight_layout()
    plt.show()


def draw_multihist_model(mse_dict, acc_dict):
    '''
    Compare mean squared error and accuracy between different models.
    '''
    assert isinstance(mse_dict, dict)
    assert isinstance(acc_dict, dict)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    y_names = ["Anxiety", "Depression", "Insomnia", "OCD", "Effect"] 
    bar_width = 0.15

    # Draw MSE loss bar graph
    for idx, (key, val) in enumerate(mse_dict.items()):
        offset = idx * bar_width
        bars = ax[0].bar(x      = offset + np.arange(len(mse_dict)),
                         height = val,
                         width  = bar_width,
                         label  = key)
        
    ax[0].legend(loc="upper left")
    ax[0].set_ylabel("MSE loss")
    ax[0].set_xticks(2*bar_width + np.arange(len(mse_dict)), y_names)

    # Draw Accuracy bar graph
    for idx, (key, val) in enumerate(acc_dict.items()):
        offset = idx * bar_width
        bars = ax[1].bar(x      = offset + np.arange(len(acc_dict)),
                         height = val,
                         width  = bar_width,
                         label  = key)

    ax[1].legend(loc="upper left")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xticks(2*bar_width + np.arange(len(acc_dict)), y_names)

    fig.suptitle("Models performance")
    plt.tight_layout()
    plt.show()
