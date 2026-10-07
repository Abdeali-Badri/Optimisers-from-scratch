import numpy as np
import torch
import torch.nn as nn
import time

torch.manual_seed(42)
np.random.seed(42)

def exponential_decay(initial_lr, step,decay_rate=0.99):
    """
    Calculate the learning rate using exponential decay.

    Mathematical formula:

        eta_t = eta_0 * gamma^t

    where:

        eta_t     = learning rate at step t
        eta_0     = initial learning rate
        gamma     = decay rate
        t         = current step

    Example:

        eta_0 = 0.01
        gamma = 0.99

        eta_1 = 0.01 * 0.99
        eta_2 = 0.01 * 0.99^2

    The learning rate therefore gradually decreases.
    """
    
    learning_rate = ( initial_lr * (decay_rate ** step))
    return learning_rate

def cosine_annealing( initial_lr,step,total_steps, min_lr=0.0):
    """
    Calculate the learning rate using cosine annealing.

    Mathematical formula:

        eta_t =
            eta_min
            +
            1/2 * (eta_0 - eta_min)
            * [1 + cos(pi*t/T)]

    where:

        eta_0 = initial learning rate
        eta_min = minimum learning rate
        t = current step
        T = total number of steps

    Cosine annealing starts with a relatively high
    learning rate and gradually decreases it.
    """
    step = min(step, total_steps)
    cosine_value = np.cos(np.pi * step / total_steps )
    learning_rate = (min_lr + 0.5 * (initial_lr - min_lr) * (1 + cosine_value))
    return learning_rate


class OptimizerFromScratch:
    """
    Base class used by all our custom optimizers.

    Every optimizer receives:

        parameters
        learning rate
        learning-rate schedule
        total number of steps

    Child classes implement their own update equations.
    """

    def __init__( self, parameters, lr=0.01, schedule=None, total_steps=1000):
        """
        Initialize the optimizer.
        """

        # Convert the model parameters into a list.
        self.parameters = list(parameters)
        self.initial_lr = lr
        self.lr = lr
        self.schedule = schedule
        self.total_steps = total_steps
        self.step_number = 0

    def get_learning_rate(self):
        """
        Return the learning rate for the current step.
        """
        if self.schedule is None:
            return self.initial_lr
        if self.schedule == "exponential":
            return exponential_decay( initial_lr=self.initial_lr, step=self.step_number, decay_rate=0.99)
        if self.schedule == "cosine":
            return cosine_annealing( initial_lr=self.initial_lr, step=self.step_number, total_steps=self.total_steps, min_lr=0.00001 )
        raise ValueError("Invalid learning-rate schedule.")

    def zero_grad(self):
        """
        Clear all parameter gradients.
        """
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad = None

    def step(self):
        raise NotImplementedError


class SGDFromScratch(OptimizerFromScratch):
    """
    Vanilla SGD.

    Mathematical update:

        theta_(t+1)
        =
        theta_t
        -
        eta_t * g_t

    where:

        theta_t = current parameter
        g_t     = gradient
        eta_t   = learning rate
    """

    def step(self):
        learning_rate = self.get_learning_rate()
        with torch.no_grad():
            for parameter in self.parameters:
                if parameter.grad is None:
                    continue
                # theta = theta - learning_rate * gradient
                parameter -= ( learning_rate * parameter.grad)
        self.step_number += 1
        self.lr = learning_rate
        

class MomentumFromScratch(OptimizerFromScratch):
    """
    SGD with Momentum.

    First calculate velocity:

        v_t =
            beta * v_(t-1)
            +
            g_t

    Then update parameters:

        theta_(t+1)
        =
        theta_t
        -
        eta_t * v_t

    beta controls how much previous gradients
    influence the current update.
    """

    def __init__( self, parameters, lr=0.01, beta=0.9, schedule=None, total_steps=1000):
        super().__init__(parameters, lr, schedule, total_steps)
        self.beta = beta
        self.velocity = []
        for parameter in self.parameters:
            self.velocity.append( torch.zeros_like(parameter))
    def step(self):
        learning_rate = self.get_learning_rate()
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters):
                if parameter.grad is None:
                    continue
                velocity = self.velocity[index]
                # v = beta*v + gradient
                velocity.mul_(self.beta)
                velocity.add_(parameter.grad)
                parameter -= ( learning_rate * velocity)
        self.step_number += 1
        self.lr = learning_rate
        


class AdaGradFromScratch(OptimizerFromScratch):
    """
    AdaGrad optimizer.

    Accumulated squared gradients:

        G_t =
            G_(t-1)
            +
            g_t^2

    Parameter update:

        theta_(t+1)
        =
        theta_t
        -
        eta_t
        *
        g_t
        /
        (sqrt(G_t) + epsilon)

    AdaGrad gives smaller updates to parameters
    that repeatedly receive large gradients.
    """

    def __init__( self, parameters, lr=0.01, epsilon=1e-8, schedule=None, total_steps=1000):
        super().__init__( parameters, lr, schedule, total_steps)
        self.epsilon = epsilon
        self.accumulated = []
        for parameter in self.parameters:
            self.accumulated.append( torch.zeros_like(parameter))
    def step(self):
        learning_rate = self.get_learning_rate()
        with torch.no_grad():
            for index, parameter in enumerate( self.parameters):
                if parameter.grad is None:
                    continue
                accumulated = self.accumulated[index]
                # G = G + gradient^2
                accumulated.addcmul_( parameter.grad, parameter.grad, value=1.0)
                # sqrt(G) + epsilon
                denominator = ( torch.sqrt(accumulated) + self.epsilon)
                parameter -= (learning_rate * parameter.grad/ denominator)
        self.step_number += 1
        self.lr = learning_rate


class RMSPropFromScratch(OptimizerFromScratch):
    """
    RMSProp optimizer.

    RMSProp maintains an exponentially weighted
    average of squared gradients.

    Formula:

        s_t =
            beta*s_(t-1)
            +
            (1-beta)*g_t^2

    Update:

        theta_(t+1)
        =
        theta_t
        -
        eta_t*g_t
        /
        (sqrt(s_t) + epsilon)
    """

    def __init__(self, parameters, lr=0.001, beta=0.9, epsilon=1e-8, schedule=None, total_steps=1000):
        super().__init__( parameters, lr, schedule, total_steps)
        self.beta = beta
        self.epsilon = epsilon
        self.square_average = []
        for parameter in self.parameters:
            self.square_average.append( torch.zeros_like(parameter))
    def step(self):
        learning_rate = self.get_learning_rate()
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters):
                if parameter.grad is None:
                    continue
                square_average = ( self.square_average[index])
                # beta * previous_average
                square_average.mul_(self.beta)
                # (1-beta) * gradient^2
                square_average.addcmul_( parameter.grad, parameter.grad, value=1 - self.beta)

                denominator = ( torch.sqrt(square_average) + self.epsilon)
                parameter -= (learning_rate * parameter.grad/ denominator)
        self.step_number += 1
        self.lr = learning_rate



class AdamFromScratch(OptimizerFromScratch):
    """
    Adam optimizer.

    Adam combines:
        - Momentum
        - Adaptive learning rates

    First moment:

        m_t =
            beta1*m_(t-1)
            +
            (1-beta1)*g_t

    Second moment:

        v_t =
            beta2*v_(t-1)
            +
            (1-beta2)*g_t^2

    Bias correction:

        m_hat_t =
            m_t / (1-beta1^t)

        v_hat_t =
            v_t / (1-beta2^t)

    Final update:

        theta_(t+1)
        =
        theta_t
        -
        eta_t
        *
        m_hat_t
        /
        (sqrt(v_hat_t) + epsilon)
    """

    def __init__(self,parameters, lr=0.001, beta1=0.9,beta2=0.999, epsilon=1e-8,schedule=None,total_steps=1000):
        super().__init__( parameters,lr,schedule,total_steps)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []
        for parameter in self.parameters:
            self.m.append(torch.zeros_like(parameter))
            self.v.append(torch.zeros_like(parameter))

    def step(self):
        learning_rate = self.get_learning_rate()
        # This is required for Adam bias correction.
        self.step_number += 1
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters):
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                m = self.m[index]
                v = self.v[index]               
                # m_t = beta1*m_(t-1) + (1-beta1)*g_t

                m.mul_(self.beta1)
                m.add_( gradient,alpha=1 - self.beta1)

                
                # v_t = beta2*v_(t-1) + (1-beta2)*g_t^2

                v.mul_(self.beta2)

                v.addcmul_(gradient,gradient,value = 1 - self.beta2)

                m_hat = (m/(1 - self.beta1 ** self.step_number))
                v_hat = (v / (1 - self.beta2 ** self.step_number))

                denominator = ( torch.sqrt(v_hat) + self.epsilon)
                parameter -= ( learning_rate * m_hat / denominator)
        self.lr = learning_rate


def rosenbrock(x, y):
    """
    Calculate the two-dimensional Rosenbrock function.

    Formula:

        f(x,y)
        =
        (1-x)^2
        +
        100(y-x^2)^2

    The global minimum is:

        (x,y) = (1,1)

    and:

        f(1,1) = 0
    """
    first_term = ( 1 - x) ** 2
    second_term = (100* (y - x ** 2) ** 2)
    return ( first_term + second_term)

def optimize_rosenbrock( optimizer_class, learning_rate, iterations=3000, schedule=None, optimizer_kwargs=None):
    """
    Optimize Rosenbrock using a custom optimizer.

    No plots are generated.

    The function returns:
        final position
        final loss
        loss history
    """
    # The global minimum is [1, 1].
    parameters = torch.tensor( [-1.5, 2.0], dtype=torch.float32, device="cpu", requires_grad=True)
    parameter_list = [parameters]
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    optimizer = optimizer_class(
        parameter_list,
        lr=learning_rate,
        schedule=schedule,
        total_steps=iterations,
        **optimizer_kwargs
    )
    loss_history = []
    for iteration in range(iterations):
        optimizer.zero_grad()
        x = parameters[0]
        y = parameters[1]
        loss = rosenbrock(x,y)

        # dL/dx
        # dL/dy
        
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()
    return ( parameters.detach().cpu().numpy(), np.array(loss_history))



def run_rosenbrock_experiment():
    """
    Compare all five optimizers on Rosenbrock.
    """
    print("PART 1: ROSENBROCK OPTIMIZATION")
    configurations = {

        "SGD": {
            "class": SGDFromScratch,
            "lr": 0.0005,
            "kwargs": {}
        },

        "Momentum": {
            "class": MomentumFromScratch,
            "lr": 0.0005,
            "kwargs": {
                "beta": 0.9
            }
        },

        "AdaGrad": {
            "class": AdaGradFromScratch,
            "lr": 0.5,
            "kwargs": {}
        },

        "RMSProp": {
            "class": RMSPropFromScratch,
            "lr": 0.01,
            "kwargs": {
                "beta": 0.9
            }
        },

        "Adam": {
            "class": AdamFromScratch,
            "lr": 0.01,
            "kwargs": {
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8
            }
        }
    }
    results = {}
    for name, configuration in configurations.items():
        print()
        print(f"Running {name}")
        final_position, losses = (
            optimize_rosenbrock(
                optimizer_class=configuration["class"],
                learning_rate=configuration["lr"],
                iterations=3000,
                schedule=None,
                optimizer_kwargs=configuration["kwargs"]
            )
        )
        results[name] = {
            "position": final_position,
            "loss": losses
        }
        print(f"Final x = {final_position[0]:.6f}")
        print(f"Final y = {final_position[1]:.6f}")
        print(f"Final loss = {losses[-1]:.10f}")
    return results



def create_classification_dataset(
    samples_per_class=250
):
    """
    Create a simple two-class classification dataset.

    Class 0:
        centered approximately around [-1.5, -1.0]

    Class 1:
        centered approximately around [1.5, 1.0]

    The neural network will learn to distinguish the two classes.
    """
    class_0 = (
        torch.randn(samples_per_class,2) * 0.8)
    class_0 += torch.tensor( [-1.5, -1.0])
    class_1 = (
        torch.randn(samples_per_class, 2)* 0.8)
    class_1 += torch.tensor( [1.5, 1.0])
    X = torch.cat([ class_0, class_1],dim=0)

    y = torch.cat(
        [
            torch.zeros(
                samples_per_class
            ),

            torch.ones(
                samples_per_class
            )
        ]
    ).long()
    permutation = torch.randperm(len(X))
    X = X[permutation]
    y = y[permutation]
    X = X.to("CPU")
    y = y.to("CPU")
    return X, y



class SmallNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,32 )
        self.layer2 = nn.Linear( 32,16)
        self.output = nn.Linear( 16,2)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.output(x)
        return x
def train_model(
    optimizer_name,
    optimizer_class,
    learning_rate,
    epochs=200,
    schedule=None,
    optimizer_kwargs=None
):
    X, y = create_classification_dataset()
    model = SmallNeuralNetwork().to("CPU")
    criterion = nn.CrossEntropyLoss()
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = optimizer_class(
        model.parameters(),
        lr=learning_rate,
        schedule=schedule,
        total_steps=epochs,
        **optimizer_kwargs
    )
    loss_history = []
    accuracy_history = []
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion( outputs,y)
        loss.backward()
        optimizer.step()
        predictions = torch.argmax(outputs,dim=1)
        accuracy = ( predictions == y).float().mean()
        loss_history.append( loss.item())
        accuracy_history.append(accuracy.item())
    training_time = (time.time() - start_time)

    convergence_threshold = 0.05
    convergence_epoch = None
    for epoch, loss_value in enumerate(loss_history):
        if loss_value < convergence_threshold:
            convergence_epoch = epoch + 1
            break
    return {
        "model": model,
        "loss": loss_history,
        "accuracy": accuracy_history,
        "final_loss": loss_history[-1],
        "final_accuracy": accuracy_history[-1],
        "convergence_epoch": convergence_epoch,
        "time": training_time
    }

def compare_neural_network_optimizers():
    print("PART 2: NEURAL NETWORK TRAINING")

    # Optimizer configurations.
    configurations = {

        "SGD": {
            "class": SGDFromScratch,
            "lr": 0.05,
            "kwargs": {}
        },

        "Momentum": {
            "class": MomentumFromScratch,
            "lr": 0.03,
            "kwargs": {
                "beta": 0.9
            }
        },

        "AdaGrad": {
            "class": AdaGradFromScratch,
            "lr": 0.1,
            "kwargs": {}
        },

        "RMSProp": {
            "class": RMSPropFromScratch,
            "lr": 0.01,
            "kwargs": {
                "beta": 0.9
            }
        },

        "Adam": {
            "class": AdamFromScratch,
            "lr": 0.01,
            "kwargs": {
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8
            }
        }
    }
    results = {}
    for name, configuration in configurations.items():

        print()
        print(f"Training using {name}")
        result = train_model(
            optimizer_name=name,
            optimizer_class=configuration["class"],
            learning_rate=configuration["lr"],
            epochs=200,
            schedule=None,
            optimizer_kwargs=configuration["kwargs"]
        )
        results[name] = result

    
    print("OPTIMIZER COMPARISON")
    print(
        f"{'Optimizer':<15}"
        f"{'Final Loss':<20}"
        f"{'Final Accuracy':<20}"
        f"{'Convergence':<20}"
        f"{'Time (sec)':<15}"
    )
    for name, result in results.items():
        if result["convergence_epoch"] is None:
            convergence = "Not reached"
        else:
            convergence = (
                f"Epoch "
                f"{result['convergence_epoch']}"
            )

        print(
            f"{name:<15}"
            f"{result['final_loss']:<20.6f}"
            f"{result['final_accuracy'] * 100:<19.2f}%"
            f"{convergence:<20}"
            f"{result['time']:<15.4f}"
        )
    return results


def experiment_momentum_coefficients():
    """
    Test different values of beta in Momentum.

    Values:

        beta = 0.0
        beta = 0.5
        beta = 0.9
        beta = 0.99

    The experiment prints the final loss and accuracy.
    """
    print("PART 3: MOMENTUM HYPERPARAMETER EXPERIMENT")

    beta_values = [0.0, 0.5, 0.9,0.99]
    for beta in beta_values:
        print(
            f"Testing momentum beta = {beta}"
        )
        X, y = create_classification_dataset()
        model = SmallNeuralNetwork().to("CPU")
        criterion = nn.CrossEntropyLoss()
        optimizer = MomentumFromScratch(
            model.parameters(),
            lr=0.03,
            beta=beta,
            total_steps=150
        )
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion( outputs, y)
            loss.backward()
            optimizer.step()

        # Calculate predictions.
        predictions = torch.argmax( outputs, dim=1)
        accuracy = (predictions == y).float().mean()
        print(
            f"Final loss     : "
            f"{loss.item():.6f}"
        )

        print(
            f"Final accuracy : "
            f"{accuracy.item() * 100:.2f}%"
        )

def experiment_epsilon():
    """
    Test different epsilon values in Adam.

    Epsilon appears in:

        sqrt(v_hat) + epsilon

    Its purpose is to prevent division by zero
    and improve numerical stability.
    """
    print("PART 4: ADAM EPSILON EXPERIMENT")
    epsilon_values = [
        1e-2,
        1e-4,
        1e-6,
        1e-8
    ]
    for epsilon in epsilon_values:

        print()
        print(f"Testing epsilon = {epsilon}")
        X, y = create_classification_dataset()
        model = SmallNeuralNetwork().to("CPU")
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamFromScratch(
            model.parameters(),
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=epsilon,
            total_steps=150
        )
        for epoch in range(150):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y).float().mean()
        print(
            f"Final loss     : "
            f"{loss.item():.6f}"
        )

        print(
            f"Final accuracy : "
            f"{accuracy.item() * 100:.2f}%"
        )

def compare_learning_rate_schedules():
    print("PART 5: LEARNING RATE SCHEDULE EXPERIMENT")
    schedules = [
        None,
        "exponential",
        "cosine"
    ]
    for schedule in schedules:
        if schedule is None:
            schedule_name = "Constant"
        else:
            schedule_name = schedule.title()
        print()
        print(
            f"Testing schedule: {schedule_name}"
        )
        X, y = create_classification_dataset()
        model = SmallNeuralNetwork().to("CPU")
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamFromScratch(
            model.parameters(),
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            schedule=schedule,
            total_steps=200
        )
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs,y)
            loss.backward()
            optimizer.step()
        predictions = torch.argmax(outputs, dim=1)
        accuracy = ( predictions == y).float().mean()
        print(
            f"Final loss     : "
            f"{loss.item():.6f}"
        )

        print(
            f"Final accuracy : "
            f"{accuracy.item() * 100:.2f}%"
        )

        print(
            f"Final LR       : "
            f"{optimizer.lr:.10f}"
        )