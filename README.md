# sigmas_tools_and_the_golden_scheduler
A few nodes to mix sigmas and a custom scheduler that uses phi, then one using eval() to be able to schedule with custom formulas.

# Nodes

**Merge sigmas by average**: takes sigmas_1 and sigmas_2 as an input and merge them with a custom weight.

**Merge sigmas gradually** : takes sigmas_1 and sigmas_2 as an input and merge them by starting with sigmas_1 times the weight and sigmas_2 times 1-the weight, like if you want to start with karras and end with simple.

**Multiply sigmas**: simply multiply the sigmas by what you want.

**Split and concatenate sigmas**: takes sigmas_1 and sigmas_2 as an input and merge them by starting with sigmas_1 until the chosen step, then the rest with sigmas_2

**Get sigmas as float**: Just get first - last step to be able to inject noise inside a latent with noise injection nodes.

**Graph sigmas**: make a graph of the sigmas.

**Manual scheduler**: uses eval() to create a custom schedule. The math module is fully imported. Available variables are:
- sigmin: sigma min
- sigmax: sigma max
- phi
- pi comes from math
- x equals 1 for the first step and 0 for the last step.
- y equals 0 for the first step and 1 for the last step.
- s or steps: total amount of steps.
- j from 0 to total steps -1.

And this one makes the max sigma proportional to the amount of steps, it is pretty good with dpmpp2m:

    max([x**phi*s/phi,sigmin])


This one works nicely with lms, euler and dpmpp2m:

    x**((x+1)*phi)*sigmax+y**((x+1)*phi)*sigmin


Here is how the graphs look like:

![image](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler/assets/15731540/b1d622b8-d3c1-4823-8c1b-73216fc0ce66)


**The Golden Scheduler**: Uses phi as the exponent. Hence the name 😊. The formula is pretty simple:

    (1-x/(steps-1))**phi*sigmax+(x/(steps-1))**phi*sigmin for x in range(steps)

or if you want to use it in the manual node:

    x**phi*sigmax+(1-x)**phi*sigmin

**It works pretty well with dpmpp2m, euler and lms!**

Here is a comparison, side by side with karras. Karras being on the right (or below depending on your screen):

![Golden Scheduler](golden_scheduler.png) ![With Karras](with_karras.png)

Using pi as the exponent is nice too (default formula in the manual node):
![01393UI_00001_](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler/assets/15731540/e15b29b2-9c6c-43a4-b976-e46c6b86003e)

