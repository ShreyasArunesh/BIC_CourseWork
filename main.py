from MLP import MLP
from MLP import FCLayer
import numpy as np
# x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])


input_array = np.loadtxt("dataset/data_banknote_authentication.txt",delimiter=',')
np.random.shuffle(input_array)
x_train = input_array[:,:-1]
x_train = np.reshape(x_train,(-1,1,4))
y_train = input_array[:,-1]
y_train = np.reshape(y_train,(-1,1,1))


hyperparameters = [
    # # Accuracy vs layers and neurone [1-10][2-10]
    # {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1,"learning_rate":0.5},
    # {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":6,"no_neurons":6,"loss":"mse","epochs":1,"learning_rate":0.5},
    #
    # # Accuracy vs activation function in hidden layer [tanh,relu]
    # {"hl_act": "relu","ol_act": "relu","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":100,"learning_rate":0.5},
    # {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":100,"learning_rate":0.5},
    #
    # # Accuracy vs activation function in output layer [tanh,sigmoid,relu]
    # {"hl_act": "tanh","ol_act": "sigmoid","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":100,"learning_rate":0.5},
    # {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":100,"learning_rate":0.5},
    # {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":100,"learning_rate":0.5},
    #
    # # Accuracy vs learning rate [0.1-0.9]
    # {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":100,"learning_rate":0.5},
    #
    # # Accuracy vs epoch [1-1000,100]
    # {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":100,"learning_rate":0.5},

    # Accuracy vs loss [mse,log]
    # {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":100,"learning_rate":0.5},
    {"hl_act": "tanh","ol_act": "sigmoid","no_hiddenlayers":3,"no_neurons":4,"loss":"binary_cross_entropy","epochs":10,"learning_rate":0.5},
]

results = []
time = []

for set_index,set in enumerate(hyperparameters):
    print(f"Training on set :  {set}\n")
    avg_acc_set_10 = []
    for i in range(10):
        # start time
        print( f"Random Weights Training subset {i+1} Processing...\n")
        model = MLP()
        model.add_layer(FCLayer(x_train.shape[-1], set["no_neurons"], set["hl_act"]))
        for l in range(set["no_hiddenlayers"]):
            model.add_layer(FCLayer(set["no_neurons"], set["no_neurons"], set["hl_act"]))
        model.add_layer(FCLayer(set["no_neurons"], 1, set["ol_act"]))
        model.compile(set["loss"])
        model.fit(x_train=x_train, y_train=y_train,epochs=set["epochs"], learning_rate=set["learning_rate"])

        pred = model.predict(x_train)

        # end time
        avg_acc_set_10.append(np.mean(np.around(pred, 0) == y_train))

    # time.append(end time - start time)

    results.append({f"set-{set_index}":avg_acc_set_10})

    print(f"\nAverage Accuracy over 10 Training (random weights) : {results[-1]}")


#
# model = MLP()
# model.add_layer(FCLayer(2, 4, 'tanh'))
# model.add_layer(FCLayer(4, 1, 'sigmoid'))
#
# model.fit(x_train, y_train,100, 0.5)
# model.compile(log)