from MLP import MLP
from MLP import FCLayer
import numpy as np
from time import time
# x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])


input_array = np.loadtxt("dataset/data_banknote_authentication.txt",delimiter=',')
np.random.shuffle(input_array)
x_train = input_array[:,:-1]
x_train = np.reshape(x_train,(-1,1,4))
y_train = input_array[:,-1]
y_train = np.reshape(y_train,(-1,1,1))


Output_file = open("output.txt","w+")

hyperparameters = [
     # Accuracy vs layers and neurone [1-10][2-10]
    {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":2,"no_neurons":2,"loss":"mse","epochs":500,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":3,"no_neurons":3,"loss":"mse","epochs":500,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":4,"no_neurons":4,"loss":"mse","epochs":500,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":5,"no_neurons":5,"loss":"mse","epochs":500,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":6,"no_neurons":6,"loss":"mse","epochs":500,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":7,"no_neurons":7,"loss":"mse","epochs":500,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":8,"no_neurons":8,"loss":"mse","epochs":500,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":9,"no_neurons":9,"loss":"mse","epochs":500,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":10,"no_neurons":10,"loss":"mse","epochs":500,"learning_rate":0.1},


    # Accuracy vs activation function in hidden layer [tanh,relu]
    {"hl_act": "relu","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.1},

    # Accuracy vs activation function in output layer [tanh,sigmoid,relu]
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "sigmoid","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.1},

    # Accuracy vs learning rate [0.1-0.9]
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.2},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.3},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.4},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.5},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.6},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.7},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.8},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.9},

    # Accuracy vs epoch [1-1000,100]
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":100,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":200,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":400,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":600,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":800,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.1},

    # Accuracy vs loss [mse,log]
    {"hl_act": "tanh","ol_act": "relu","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "tanh","no_hiddenlayers":3,"no_neurons":4,"loss":"mse","epochs":1000,"learning_rate":0.1},
    {"hl_act": "tanh","ol_act": "sigmoid","no_hiddenlayers":3,"no_neurons":4,"loss":"binary_cross_entropy","epochs":1000,"learning_rate":0.1},
]

avg_acccuracy = []
avg_total_time = []
for set_index,set in enumerate(hyperparameters):
    print(f"Training on set :  {set}\n")
    Output_file.write(f"Training on set :  {set}\n\n")
    acc_set_10 = []
    total_time_10 = []
    for i in range(10):
        # start time
        start = time()
        print(f"Random Weights Training subset {i + 1} Processing...\n")
        model = MLP()
        model.add_layer(FCLayer(x_train.shape[-1], set["no_neurons"], set["hl_act"]))
        for l in range(set["no_hiddenlayers"]):
            model.add_layer(FCLayer(set["no_neurons"], set["no_neurons"], set["hl_act"]))
        model.add_layer(FCLayer(set["no_neurons"], 1, set["ol_act"]))
        model.compile(set["loss"])
        model.fit(x_train=x_train, y_train=y_train,epochs=set["epochs"], learning_rate=set["learning_rate"])

        pred = model.predict(x_train)

        # end time
        total_time_10.append (time() - start)
        acc_set_10.append(np.around(pred, 0) == y_train)
        Output_file.write(f"Random Weights Training subset {i + 1} Processed \n")

    # avg_acc_set_10.append(avg)

    avg_acccuracy.append({f"set-{set_index}":np.mean(acc_set_10)})
    avg_total_time.append({f"set-{set_index}":np.mean(total_time_10)})

    print(f"\nAverage Accuracy of a sets over 10 Training (random weights) : {avg_acccuracy[-1]}")
    print( f"\nAverage Time taken to run that sets over 10 Training (random weights) : {avg_total_time[-1]} seconds \n\n")

    Output_file.write(f"\nAverage Accuracy of a sets over 10 Training (random weights) : {avg_acccuracy[-1]}")
    Output_file.write (f"\nAverage Time taken to run that sets over 10 Training (random weights) : {avg_total_time} seconds \n\n")

Output_file.close()