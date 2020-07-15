"""
Implement custom policies/neural network configurations
"""
import gym

from stable_baselines.common.policies import FeedForwardPolicy, register_policy

# Taken from documentation just to test
class CustomPolicy(FeedForwardPolicy):
    def __init__(self,*args,**kwargs):
        super(CustomPolicy,self).__init__(*args, **kwargs, net_arch=[0,dict(pi=[128, 128, 128], vf=[128, 128, 128])], feature_extraction="mlp")

register_policy('CustomPolicy', CustomPolicy)