from flask import Flask,request
import gym
import gym_drill
from gym_drill.envs.Target import TargetBall
from gym_drill.envs.Hazard import Hazard
import agent_training as at

BENCHMARK_MODEL = "./trained_models/final_test"

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route("/my_first_post_request",methods = ["POST","GET"])
def req():
    if request.method == "POST":
        try:
            data = request.get_json()
            name = data["Name"]
            age = data["Age"]
            return "My name is: " + str(name) + " and I'm this old: " + str(age)
        except Exception as e:
            return str(e)            

@app.route("/get_well_path",methods = ["POST","GET"])
def get_well_path():
    if request.method == "GET":
        return "What are you hoping to get here, huh?"
    else:
        try:
            data = request.get_json()
            targets = data["Targets"]
            hazards = data["Hazards"]

            target_list = []
            for target_dict in targets:
                x = target_dict["x"]
                y = target_dict["y"]
                r = target_dict["r"]
                target_candidate = TargetBall(x,y,r)
                target_list.append(target_candidate)
            
            hazard_list = []
            for hazard_dict in hazards:
                x = hazard_dict["x"]
                y = hazard_dict["y"]
                r = hazard_dict["r"]
                hazard_candidate = Hazard(x,y,r)
                hazard_list.append(hazard_candidate)

            model = at.get_trained_model(BENCHMARK_MODEL)
            

        except Exception as e:
            return str(e)

if __name__ == '__main__':
    app.run(debug=True)