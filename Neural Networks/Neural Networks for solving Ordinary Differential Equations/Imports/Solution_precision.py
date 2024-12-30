import tensorflow as tf


def compute_atol(t_eval, model, analytic_solution, verbose=0):
    # Compute the exact solution at the given evaluation points
    y_exact = analytic_solution(t_eval)

    # Compute the model's predictions at the given evaluation points
    y_model = model(t_eval)

    # Calculate absolute error
    absolute_errors = tf.abs(y_model - y_exact)

    # Calculate overall tolerance
    atol = tf.reduce_max(absolute_errors).numpy()

    if verbose:
        print(f"Absolute tolerance: {atol}")

    return atol
