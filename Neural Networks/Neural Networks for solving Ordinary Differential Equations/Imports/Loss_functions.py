import tensorflow as tf


# Function to compute derivatives of a model with respect to inputs up to a specified order
@tf.function(jit_compile=True)
def compute_derivative(model, t_x, order=1):
    """
    Computes derivatives of a model's output with respect to its input up to a specified order.

    Parameters:
    model (tf.keras.Model or callable): The TensorFlow model whose derivatives are computed.
    t_x (tf.Tensor): The input tensor with respect to which derivatives are computed.
    order (int): The order of derivatives to compute (default is 1).

    Returns:
    list[tf.Tensor]: A list of tensors representing derivatives from 1st order up to the specified order.
    """
    derivatives = []

    # Use a persistent GradientTape if higher-order derivatives are needed
    with tf.GradientTape(persistent=(order > 1)) as tape:
        tape.watch(t_x)  # Track the input tensor for differentiation
        y = model(t_x)  # Evaluate the model output

        # Compute successive derivatives iteratively
        for i in range(order):
            y = tape.gradient(y, t_x)  # Compute the gradient
            derivatives.append(y)

    # Free up memory by deleting the persistent tape if it was used
    if order > 1:
        del tape

    return derivatives


# Custom loss function for an ordinary differential equation dy/dt + y = 0, y(0) = 1
@tf.function
def loss_fn_exp(model, t, initial_condition=1.0, initial_coordinate=0.0):
    """
    Defines a custom loss function for solving the differential equation dy/dt + y = 0, y(0) = 1.

    Parameters:
    model (tf.keras.Model or callable): The TensorFlow model representing the solution to the differential equation.
    t (tf.Tensor): Input tensor representing time or independent variable values.
    initial_condition (float): The initial value of the solution y(0) (default is 1.0).
    initial_coordinate (float): The coordinate value at which the initial condition is applied (default is 0.0).

    Returns:
    tf.Tensor: The computed loss value combining the differential equation residual and the initial condition error.
    """
    y_pred = model(t)  # Compute the predicted solution values

    # Compute the first-order derivative of the model output
    dy_dt = compute_derivative(model, t)[0]

    # Calculate the residual of the differential equation
    equation_residual = dy_dt + y_pred  # Knowing the equation is dy/dt + y = 0

    # Compute the loss for the initial condition: (y(initial_coordinate) - initial_condition)^2
    initial_loss = tf.square(model(tf.constant([[initial_coordinate]])) - initial_condition)

    # Combine the residual loss and initial condition loss
    return tf.reduce_mean(tf.square(equation_residual)) + initial_loss
