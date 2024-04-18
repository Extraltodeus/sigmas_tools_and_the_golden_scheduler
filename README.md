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
- f gives a normalized from 1 to 0 curve based on a reversed Fibonacci sequence

And this one makes the max sigma proportional to the amount of steps, it is pretty good with dpmpp2m:

    max([x**phi*s/phi,sigmin])


This one works nicely with lms, euler and dpmpp2m NOW ALSO WITH dpmpp2m_sde if you toggle the sgm button:

    x**((x+1)*phi)*sigmax+y**((x+1)*phi)*sigmin


Here is how the graphs look like:

![image](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler/assets/15731540/b1d622b8-d3c1-4823-8c1b-73216fc0ce66)


**The Golden Scheduler**: Uses phi as the exponent. Hence the name 😊. The formula is pretty simple:

    (1-x/(steps-1))**phi*sigmax+(x/(steps-1))**phi*sigmin for x in range(steps)

Where x it the iteration variable for the steps.

Or if you want to use it in the manual node:

    x**phi*sigmax+y**phi*sigmin

**It works pretty well with dpmpp2m, euler and lms!**

The karras formula can be written like this:

    (sigmax ** (1 / 7) + y * (sigmin ** (1 / 7) - sigmax ** (1 / 7))) ** 7

Using tau:

    (sigmax ** (1 / tau) + y * (sigmin ** (1 / tau) - sigmax ** (1 / tau))) ** tau

Here is a comparison, the golden scheduler,  using my model [Iris Lux](https://civitai.com/models/201287?modelVersionId=234300) :

![Golden Scheduler](golden_scheduler.png)

Karras:

![With Karras](with_karras.png)

With a formula based on the fibonacci sequence:

    (sigmax-sigmin)*f**(1/2)+sigmin

![00048UI_00001_](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler/assets/15731540/0a68f046-3261-433e-abf2-44501674838d)


Here is a mix using dpmpp3m_sde with 50% exponential, 25% simple and 25% sgm uniform:

![00958UI_00001_](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler/assets/15731540/51c65822-12b8-4ef2-980c-2df792838d17)


![456546456465](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler/assets/15731540/f0ea29f5-f92b-4cf4-9040-0117a635df9d)

![image_grid](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler/assets/15731540/d5205921-2b24-4a5f-8f4a-32d6aa7f7430)

![2342434](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler/assets/15731540/425f4684-ea54-4dce-b5c2-19b93afb6233)




