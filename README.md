# sigmas_tools_and_the_golden_scheduler
A few nodes to mix sigmas and a custom scheduler that uses phi

# Nodes

Merge sigmas by average: takes sigmas_1 and sigmas_2 as an input and merge them with a custom weight.

Merge sigmas gradually : takes sigmas_1 and sigmas_2 as an input and merge them by starting with sigmas_1 times the weight and sigmas_2 times 1-the weight, like if you want to start with karras and end with simple.

Multiply sigmas: simply multiply the sigmas by what you want.

Split and concatenate sigmas: takes sigmas_1 and sigmas_2 as an input and merge them by starting with sigmas_1 until the chosen step, then the rest with sigmas_2

The Golden Scheduler: Uses phi as the exponent. The formula is pretty simple:

    (1-x/(steps-1))**phi*sigmax+(x/(steps-1))**phi*sigmin for x in range(steps)

It works pretty well with dpmpp2m, euler and lms!

Here is a comparison, side by side with karras. Karras being on the right (or below depending on your screen):

![Golden Scheduler](golden_scheduler.png) ![With Karras](with_karras.png)
