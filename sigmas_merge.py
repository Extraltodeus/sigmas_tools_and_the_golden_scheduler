import torch
from copy import deepcopy

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
from math import *

def tensor_to_graph_image(tensor):
    plt.figure()
    plt.plot(tensor.numpy())
    plt.title("Graph from Tensor")
    plt.xlabel("Index")
    plt.ylabel("Value")
    with BytesIO() as buf:
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf).copy()
    plt.close()
    return image

def fibonacci_normalized_descending(n):
    fib_sequence = [0, 1]
    for _ in range(n):
        if n > 1:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    max_value = fib_sequence[-1]
    normalized_sequence = [x / max_value for x in fib_sequence]
    descending_sequence = normalized_sequence[::-1]
    return descending_sequence

class sigmas_merge:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"forceInput": True}),
                "sigmas_2": ("SIGMAS", {"forceInput": True}),
                "proportion_1": ("FLOAT", {"default": 0.5, "min": 0,"max": 1,"step": 0.01})
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def simple_output(self, sigmas_1, sigmas_2, proportion_1):
        return (sigmas_1*proportion_1+sigmas_2*(1-proportion_1),)
    
class sigmas_mult:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "factor": ("FLOAT", {"default": 1, "min": 0,"max": 100,"step": 0.01})
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def simple_output(self, sigmas, factor):
        return (sigmas*factor,)
    
class sigmas_to_graph:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "print_as_list" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def simple_output(self, sigmas,print_as_list):
        if print_as_list:
            print(sigmas.tolist())
            sigmas_percentages = ((sigmas-sigmas.min())/(sigmas.max()-sigmas.min())).tolist()
            sigmas_percentages_w_steps = [(i,round(s,4)) for i,s in enumerate(sigmas_percentages)]
            print(sigmas_percentages_w_steps)
        sigmas_graph = tensor_to_graph_image(sigmas.cpu())
        numpy_image = np.array(sigmas_graph)
        numpy_image = numpy_image / 255.0
        tensor_image = torch.from_numpy(numpy_image)
        tensor_image = tensor_image.unsqueeze(0)
        images_tensor = torch.cat([tensor_image], 0)
        return (images_tensor,)

class sigmas_concat:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"forceInput": True}),
                "sigmas_2": ("SIGMAS", {"forceInput": True}),
                "sigmas_1_until": ("INT", {"default": 10, "min": 0,"max": 1000,"step": 1}),
                "rescale_sum" : ("BOOLEAN", {"default": True}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def simple_output(self, sigmas_1, sigmas_2, sigmas_1_until,rescale_sum):
        result = torch.cat((sigmas_1[:sigmas_1_until], sigmas_2[sigmas_1_until:]))
        if rescale_sum:
            result = result*torch.sum(result).item()/torch.sum(sigmas_1).item()
        return (result,)

class the_golden_scheduler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 0,"max": 100000,"step": 1}),
                "sgm" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    
    def simple_output(self,model,steps,sgm):
        s = model.model.model_sampling
        sigmin = s.sigma(s.timestep(s.sigma_min))
        sigmax = s.sigma(s.timestep(s.sigma_max))
        
        if sgm:
            steps+=1
        phi = (1 + 5 ** 0.5) / 2
        sigmas = [(1-x/(steps-1))**phi*sigmax+(x/(steps-1))**phi*sigmin for x in range(steps)]
        if sgm:
            sigmas = sigmas[:-1]
        sigmas = torch.tensor(sigmas+[0])
        return (sigmas,)

class sigmas_min_max_out_node:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("FLOAT","FLOAT",)
    RETURN_NAMES = ("Sigmas_max","Sigmas_min",)
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def simple_output(self,model):
        s = model.model.model_sampling
        sigmin = s.sigma(s.timestep(s.sigma_min)).item()
        sigmax = s.sigma(s.timestep(s.sigma_max)).item()
        return (sigmax,sigmin,)


class manual_scheduler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "custom_sigmas_manual_schedule": ("STRING", {"default": "((1 - cos(2 * pi * (1-y**0.5) * 0.5)) / 2)*sigmax+((1 - cos(2 * pi * y**0.5 * 0.5)) / 2)*sigmin"}),
                "steps": ("INT", {"default": 20, "min": 0,"max": 100000,"step": 1}),
                "sgm" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    
    def simple_output(self,model, custom_sigmas_manual_schedule,steps,sgm):
        if sgm:
            steps+=1
        s = model.model.model_sampling
        sigmin = s.sigma(s.timestep(s.sigma_min))
        sigmax = s.sigma(s.timestep(s.sigma_max))
        phi  = (1 + 5 ** 0.5) / 2
        sigmas = []
        s = steps
        fibo = fibonacci_normalized_descending(s)
        for j in range(steps):
            y = j/(s-1)
            x = 1-y
            f = fibo[j]
            try:
                f = eval(custom_sigmas_manual_schedule)
            except:
                print("could not evaluate {custom_sigmas_manual_schedule}")
                f = 0
            sigmas.append(f)
        if sgm:
            sigmas = sigmas[:-1]
        sigmas = torch.tensor(sigmas+[0])
        return (sigmas,)
    
def remap_range_no_clamp(value, minIn, MaxIn, minOut, maxOut):
            finalValue = ((value - minIn) / (MaxIn - minIn)) * (maxOut - minOut) + minOut;
            return finalValue;

class get_sigma_float:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sigmas": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("FLOAT",)
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def simple_output(self,sigmas,model):
        sigfloat = float((sigmas[0]-sigmas[-1])/model.model.latent_format.scale_factor)
        return (sigfloat,)

def remap_range_no_clamp(value, minIn, MaxIn, minOut, maxOut):
            finalValue = ((value - minIn) / (MaxIn - minIn)) * (maxOut - minOut) + minOut;
            return finalValue;

class sigmas_gradual_merge:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"forceInput": True}),
                "sigmas_2": ("SIGMAS", {"forceInput": True}),
                "proportion_1": ("FLOAT", {"default": 0.5, "min": 0,"max": 1,"step": 0.01})
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def simple_output(self,sigmas_1,sigmas_2,proportion_1):
        result_sigmas = deepcopy(sigmas_1)
        for idx,s in enumerate(result_sigmas):
            current_factor = remap_range_no_clamp(idx,0,len(result_sigmas)-1,proportion_1,1-proportion_1)
            result_sigmas[idx] = sigmas_1[idx]*current_factor+sigmas_2[idx]*(1-current_factor)
        return (result_sigmas,)
    
NODE_CLASS_MAPPINGS = {
    "Merge sigmas by average": sigmas_merge,
    "Merge sigmas gradually": sigmas_gradual_merge,
    "Multiply sigmas": sigmas_mult,
    "Split and concatenate sigmas": sigmas_concat,
    "The Golden Scheduler": the_golden_scheduler,
    "Manual scheduler": manual_scheduler,
    "Get sigmas as float": get_sigma_float,
    "Graph sigmas": sigmas_to_graph,
    "Output min/max sigmas": sigmas_min_max_out_node,
}
