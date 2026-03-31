from numpy import arange,asarray
from numpy.random import rand
import matplotlib.pyplot as plt
def objective(x):
    return x**2
def derivative(x):
    return x*2
def gradient_descend(objectives,derivatives,bounds,n_iter,step_size):
    solutions=list()
    scores=list()
    solution=bounds[:,0]+rand(len(bounds))*bounds[:,1]-bounds[:,0]
    for i in range(n_iter):
        gradient=derivatives(solution)
        solution=solution-step_size*gradient
        solution_eval=objectives(gradient)
        solutions.append(solution)
        scores.append(solution_eval)
        print("%d f(%s) = %.5f"%(i,solution,solution_eval))
    return [solutions,scores]
bounds=asarray([[-1.0,1.0]])
n_iter=30
step_size=0.1
solutions,scores=gradient_descend(objective,derivative,bounds,n_iter,step_size)
inputs=arange(bounds[0,0],bounds[0,1]+0.1,0.1)
results=objective(inputs)
plt.plot(inputs,results)
plt.plot(solutions,scores,'.-',color='red')
plt.show()