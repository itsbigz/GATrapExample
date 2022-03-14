# GATrapExample
Using the previously implemented genetic algorithm, I am trying to solve a 3-bit hierarchical trap for 27 and 81 bit strings.
How do I count the fitness in hierarchical trap?
Contribution function: what each block contributes to the total fitness In low levels: 
flow = fhigh = 1 multiplied by 3^height(x) In high level: flow = 0.8 and fhigh = 1 multiplied by 3^height(x)
Interpretation function:  111 -> 1    000->0    others-> -1

part two:
we use a method of diversity in evolution. I used deterministic crowding, each child is compared to its parents and replaces the parent if better than the parent.

References:
David. E. Goldberg. and Martin Pelikan. 2001. Escaping Hierarchical Traps
with Competent Genetic Algorithms. Proceedings of the Genetic and Evolutionary Computation Conference (GECCO2001(

