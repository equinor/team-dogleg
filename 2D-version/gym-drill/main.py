import agent_training as train
import sys
import warnings  

# Verifies input and assigns kwargs
def handle_kwargs(kwargs):
    try:
        action = str(kwargs[1])
    except IndexError:
        raise IndexError("An action (train/load/retrain) must be specified!")
    verify_arg_len(kwargs)
    
    try:
        name = kwargs[2]
        algorithm = str(kwargs[3]).upper()
    except IndexError:
        raise IndexError("A model name and an algorithm must be specified!")
    verify_algorithm(algorithm)

    if action == "load":
        timesteps = None
        new_save_name = None
    
    elif action == "train":
        try:
            timesteps = int(kwargs[4])
        except IndexError:
            print("No timestep lenght specified. Running with 10k total timesteps.")
            timesteps = 10000
        except ValueError:
            raise ValueError("Timesteps must be integer!")

        new_save_name = None
    elif action == "retrain":
        try:
             timesteps = int(kwargs[4])
        except IndexError:
            print("No timestep lenght specified. Running with 10k total timesteps.")
            timesteps = 10000
        except ValueError:
            raise ValueError("Timesteps must be integer!")
        try:
            new_save_name = str(kwargs[5])
        except IndexError:
            print("No new save name has been specified, will override current model")
            new_save_name = name
    else:
        raise ValueError("Invalid command line argument!")

    return action, name, algorithm, timesteps, new_save_name

def verify_arg_len(kwargs):
    if kwargs[1] == "train":
        try:
            assert len(kwargs) <= 4
        except AssertionError:
            print("Training mode only supports a maximum of 4 arguments")
    
    elif kwargs[1] == "retrain":
        try:
            assert len(kwargs) <= 5 
        except AssertionError:
            print("Retraining mode only supports a maximum of 5 arguments")

    elif kwargs[1] == "load":
        try:
            assert len(kwargs) <= 6
        except AssertionError:
            print("Loading mode only supports a maximum of 6 arguments")

def verify_algorithm(algorithm):
    try:
        assert algorithm == "DQN" or algorithm == "PPO2"
        #assert algorithm == "dqn" or algorithm == "ppo2"
    except AssertionError:
        print(str(algorithm), "is not a valid algorithm. Must be either DQN og PPO2.")
        sys.exit()

if __name__ == '__main__':    
    try: 
        action, name, algorithm, timesteps, new_save_name = handle_kwargs(sys.argv)
    except Exception as e:
        print(str(e))
        sys.exit()

    if action == "train":
        if algorithm == "DQN":
            train.train_new_DQN(timesteps,name)
        elif algorithm =="PPO2":
            train.train_new_PPO2(timesteps,name)

    elif action == "retrain":
        if algorithm == "DQN":
            train.train_existing_DQN(name,timesteps,new_save_name) # Add support to change kwargs of function
        elif algorithm == "PPO2":
            train.train_existing_PPO2(name,timesteps,new_save_name)

    elif action == "load":
        if algorithm == "DQN":
            #model = train.get_trained_DQN_model(name)
            train.display_agent(name)
            # show model
        elif algorithm == "PPO2":
            model = train.get_trained_PPO2_model()
            train.display_agent(model)
