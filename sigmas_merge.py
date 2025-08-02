import enum
from copy import deepcopy

import matplotlib
import matplotlib.scale
import torch
from asteval import Interpreter

matplotlib.use("Agg")
from io import BytesIO
from math import *

import comfy.samplers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import norm


def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys


class GraphScale(enum.StrEnum):
    linear = "linear"
    log = "log"


def tensor_to_graph_image(tensor, color="blue", scale: GraphScale = GraphScale.linear):
    SCALE_FUNCTIONS: dict[str, matplotlib.scale.ScaleBase] = {
        GraphScale.linear: matplotlib.scale.LinearScale,
        GraphScale.log: matplotlib.scale.LogScale,
    }
    plt.figure()
    plt.plot(tensor.numpy(), marker="o", linestyle="-", color=color)
    plt.title("Graph from Tensor")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.yscale(scale)
    with BytesIO() as buf:
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf).copy()
    plt.close()
    return image


def fibonacci_normalized_descending(n):
    if n <= 0:
        return []
    if n == 1:
        fib_sequence = [1]
    else:
        fib_sequence = [1, 1]
        for _ in range(2, n):
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
                "proportion_1": (
                    "FLOAT",
                    {"default": 0.5, "min": 0, "max": 1, "step": 0.01},
                ),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    def simple_output(self, sigmas_1, sigmas_2, proportion_1):
        return (sigmas_1 * proportion_1 + sigmas_2 * (1 - proportion_1),)


class sigmas_mult:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "factor": ("FLOAT", {"default": 1, "min": 0, "max": 100, "step": 0.01}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    def simple_output(self, sigmas, factor):
        return (sigmas * factor,)


class sigmas_to_graph:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # quick list from comfyroll studio https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/blob/main/nodes/nodes_graphics_filter.py
        # that also appears to be based on the Color Tint node by hnmr293
        # trimmed of incompatible colors
        col = [
            "black",
            "red",
            "green",
            "blue",
            "cyan",
            "magenta",
            "yellow",
            "purple",
            "lime",
            "navy",
            "teal",
            "orange",
            "maroon",
            "lavender",
            "olive",
        ]

        scale_options = [option.value for option in GraphScale]

        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "color": (col, {"default": "blue"}),
                "print_as_list": ("BOOLEAN", {"default": False}),
                "scale": (scale_options, {"default": GraphScale.linear}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    def simple_output(self, sigmas, color, print_as_list, scale):
        # Convert scale string to GraphScale enum
        if isinstance(scale, str):
            scale_enum = GraphScale(scale)
        else:
            scale_enum = scale
        if print_as_list:
            print(sigmas.tolist())
            sigmas_percentages = (
                (sigmas - sigmas.min()) / (sigmas.max() - sigmas.min())
            ).tolist()
            sigmas_percentages_w_steps = [
                (i, round(s, 4)) for i, s in enumerate(sigmas_percentages)
            ]
            print(sigmas_percentages_w_steps)
        sigmas_graph = tensor_to_graph_image(
            sigmas.cpu(), color=color, scale=scale_enum
        )
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
                "sigmas_1_until": (
                    "INT",
                    {"default": 10, "min": 0, "max": 1000, "step": 1},
                ),
                "rescale_sum": ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    def simple_output(self, sigmas_1, sigmas_2, sigmas_1_until, rescale_sum):
        result = torch.cat((sigmas_1[:sigmas_1_until], sigmas_2[sigmas_1_until:]))
        if rescale_sum:
            result = result * torch.sum(result).item() / torch.sum(sigmas_1).item()
        return (result,)


class the_golden_scheduler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 0, "max": 100000, "step": 1}),
                "sgm": ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    def simple_output(self, model, steps, sgm):
        s = model.get_model_object("model_sampling")
        sigmin = s.sigma(s.timestep(s.sigma_min))
        sigmax = s.sigma(s.timestep(s.sigma_max))

        if sgm:
            steps += 1
        phi = (1 + 5**0.5) / 2
        sigmas = [
            (1 - x / (steps - 1)) ** phi * sigmax + (x / (steps - 1)) ** phi * sigmin
            for x in range(steps)
        ]
        if sgm:
            sigmas = sigmas[:-1]
        sigmas = torch.tensor(sigmas + [0])
        return (sigmas,)


class GaussianTailScheduler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 0, "max": 100000, "step": 1}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    def simple_output(self, model, steps):
        s = model.model.model_sampling
        sigmin = s.sigma(s.timestep(s.sigma_min))
        sigmax = s.sigma(s.timestep(s.sigma_max))
        sigmas = [
            (sigmax - sigmin) * 2 * (1 - norm.cdf((x / (steps - 1)) * 3.2905)) + sigmin
            for x in range(steps)
        ]
        sigmas = torch.tensor(sigmas + [0])
        return (sigmas,)


class aligned_scheduler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                # "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default":"simple"}),
                "model_type": (["SD1", "SDXL", "SVD"],),
                "force_sigma_min": ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    def simple_output(self, model, steps, model_type, force_sigma_min):
        timestep_indices = {
            "SD1": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24, 0],
            "SDXL": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13, 0],
            "SVD": [995, 920, 811, 686, 555, 418, 315, 174, 109, 12, 0],
        }
        indices = timestep_indices[model_type]
        indices = [999 - i for i in indices]
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), "simple", 1000
        )[indices]
        sigmas = loglinear_interp(
            sigmas.tolist(), steps + 1 if not force_sigma_min else steps
        )
        sigmas = torch.FloatTensor(sigmas)
        sigmas = torch.cat(
            [sigmas[:-1] if not force_sigma_min else sigmas, torch.FloatTensor([0.0])]
        )
        return (sigmas.cpu(),)


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
    RETURN_TYPES = (
        "FLOAT",
        "FLOAT",
    )
    RETURN_NAMES = (
        "Sigmas_max",
        "Sigmas_min",
    )
    CATEGORY = "sampling/custom_sampling/sigmas"

    def simple_output(self, model):
        s = model.get_model_object("model_sampling")
        sigmin = s.sigma(s.timestep(s.sigma_min)).item()
        sigmax = s.sigma(s.timestep(s.sigma_max)).item()
        return (
            sigmax,
            sigmin,
        )


class manual_scheduler:
    def __init__(self):
        self.asteval = Interpreter()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "custom_sigmas_manual_schedule": (
                    "STRING",
                    {
                        "default": "((1 - cos(2 * pi * (1-y**0.5) * 0.5)) / 2)*sigmax+((1 - cos(2 * pi * y**0.5 * 0.5)) / 2)*sigmin"
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 0, "max": 100000, "step": 1}),
                "sgm": ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    def simple_output(self, model, custom_sigmas_manual_schedule, steps, sgm):
        if sgm:
            steps += 1
        s = model.get_model_object("model_sampling")
        sigmin = s.sigma(s.timestep(s.sigma_min))
        sigmax = s.sigma(s.timestep(s.sigma_max))
        phi = (1 + 5**0.5) / 2
        sigmas = []
        s = steps
        fibo = fibonacci_normalized_descending(s)
        # Prepare asteval with math functions and variables
        aeval = Interpreter()
        aeval.symtable.update(
            {
                "sigmin": sigmin,
                "sigmax": sigmax,
                "phi": phi,
                "pi": pi,
                "cos": cos,
                "sin": sin,
                "sqrt": sqrt,
                # add other math functions as needed
            }
        )
        for j in range(steps):
            y = j / (s - 1)
            x = 1 - y
            f = fibo[j]
            aeval.symtable.update({"y": y, "x": x, "f": f, "j": j, "s": s})
            try:
                f = aeval(custom_sigmas_manual_schedule)
            except Exception as e:
                print(f"Could not evaluate '{custom_sigmas_manual_schedule}': {e}")
                f = 0
            sigmas.append(f)
        if sgm:
            sigmas = sigmas[:-1]
        sigmas = torch.tensor(sigmas + [0])
        return (sigmas,)


def remap_range_no_clamp(value, minIn, MaxIn, minOut, maxOut):
    finalValue = ((value - minIn) / (MaxIn - minIn)) * (maxOut - minOut) + minOut
    return finalValue


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

    def simple_output(self, sigmas, model):
        scale_factor = getattr(model.model.latent_format, "scale_factor", None)
        if scale_factor is None or scale_factor == 0:
            print("Warning: scale_factor is zero or not set. Returning 0.")
            sigfloat = 0.0
        else:
            sigfloat = float((sigmas[0] - sigmas[-1]) / scale_factor)
        return (sigfloat,)


class sigmas_gradual_merge:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"forceInput": True}),
                "sigmas_2": ("SIGMAS", {"forceInput": True}),
                "proportion_1": (
                    "FLOAT",
                    {"default": 0.5, "min": 0, "max": 1, "step": 0.01},
                ),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    def simple_output(self, sigmas_1, sigmas_2, proportion_1):
        result_sigmas = deepcopy(sigmas_1)
        for idx, s in enumerate(result_sigmas):
            current_factor = remap_range_no_clamp(
                idx, 0, len(result_sigmas) - 1, proportion_1, 1 - proportion_1
            )
            result_sigmas[idx] = sigmas_1[idx] * current_factor + sigmas_2[idx] * (
                1 - current_factor
            )
        return (result_sigmas,)


class multi_sigmas_average:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        sigmas_inputs = {
            f"sigmas_{X + 2}": ("SIGMAS", {"forceInput": True}) for X in range(24)
        }
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"forceInput": True}),
            },
            "optional": sigmas_inputs,
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    def simple_output(self, sigmas_1, **kwargs):
        tensors = [sigmas_1] + [v for k, v in kwargs.items() if k.startswith("sigmas_")]
        result_sigmas = torch.mean(torch.stack(tensors), dim=0)
        return (result_sigmas,)


NODE_CLASS_MAPPINGS = {
    "Merge sigmas by average": sigmas_merge,
    "Merge sigmas gradually": sigmas_gradual_merge,
    "Merge many sigmas by average": multi_sigmas_average,
    "Multiply sigmas": sigmas_mult,
    "Split and concatenate sigmas": sigmas_concat,
    "The Golden Scheduler": the_golden_scheduler,
    "Gaussian Tail Scheduler": GaussianTailScheduler,
    "Aligned Scheduler": aligned_scheduler,
    "Manual scheduler": manual_scheduler,
    "Get sigmas as float": get_sigma_float,
    "Graph sigmas": sigmas_to_graph,
    "Output min/max sigmas": sigmas_min_max_out_node,
}
