import numpy as np
import matplotlib.pyplot as plt

from wideMinima import f, fprime

def step_filter(val, criterion : bool):
    if criterion: return val
    return np.nan
filter_vec = np.vectorize(step_filter)


def gradient_descent(x, alpha, iteration = 1000, std_cut = 10e-3):
    index = iteration//10
    trace = [x]
    for _ in range(iteration):
        x = x - alpha*fprime(x)
        trace.append(x)
    trace = trace[-index:]
    
    diverge_mask = np.std(trace,axis=0) < std_cut
    return filter_vec(np.mean(trace,axis=0), diverge_mask)




if __name__ == "__main__":
    alpha_set = [0.01, 0.3, 4]
    
    for alpha in alpha_set:
        plt.clf()
        x = np.random.uniform(-5,20,(200,)); x.sort()    # np.linspace(-5,20,200)
        fx = gradient_descent(x, alpha)
        plt.plot(x, fx, 'k')
        plt.savefig(f"C:\\Users\\cjhy29\\Desktop\\mathDNN\\homework\\hw1\\Problem_4_lr_{alpha}.png", dpi=200)
