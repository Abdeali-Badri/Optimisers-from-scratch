from optimizers_from_scratch import (
    run_rosenbrock_experiment,
    compare_neural_network_optimizers,
    experiment_momentum_coefficients,
    experiment_epsilon,
    compare_learning_rate_schedules
)
def main():
    rosenbrock_results = (run_rosenbrock_experiment())
    neural_results = (compare_neural_network_optimizers())
    experiment_momentum_coefficients()
    experiment_epsilon()
    compare_learning_rate_schedules()
if __name__ == "__main__":
    main()