import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from time import time
sns.set_theme()


class ModelTrainerCosine:
    """
    A class for MLP model trainer with default Cosine Decay learning rate scheduler.
    """
    def __init__(self, params, activation_function, loss_function, decay_steps=1000, data_aug=False, lr_scheduler=None):
        """
        Initializes the ModelTrainerCosine class with specified parameters and loss function.

        Parameters:
        params (dict): Dictionary containing hyperparameters (n_layers, units, learning_rate).
        activation_function (str): Name of the activation function to use in the model layers (e.g., 'relu', 'tanh').
        loss_function (callable): Loss function for training.
        decay_steps (int): Number of steps for learning rate decay (default is 1000).
        data_aug (bool): Whether to apply data augmentation during training (default is False).
        lr_scheduler (callable): Custom learning rate scheduler (optional).
        """
        self.params = params
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.data_aug = data_aug
        self.lr_scheduler = lr_scheduler
        self.decay_steps = decay_steps
        self.model = self._build_final_model()
        self.loss_history = []
        self.lr_history = []
        self.time_to_train = ValueError("Training not yet launched")

    def _build_final_model(self):
        """
        Builds the final TensorFlow model based on the specified parameters.

        Returns:
        tf.keras.Model: Compiled Keras model.
        """
        model = tf.keras.Sequential()
        for _ in range(self.params["n_layers"]):
            model.add(tf.keras.layers.Dense(self.params["units"], activation=self.activation_function))
        model.add(tf.keras.layers.Dense(1))  # Output layer

        # Use default Cosine Decay scheduler if no different is provided
        if self.lr_scheduler is None:
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.params["learning_rate"],
                decay_steps=self.decay_steps,
                alpha=0.1
            )
        else:
            lr_schedule = self.lr_scheduler

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
        return model

    @tf.function
    def _train_step(self, t):
        """
        Performs a single training step by computing the loss and applying gradients.

        Parameters:
        t (tf.Tensor): Input tensor representing the training data.

        Returns:
        tf.Tensor: Computed loss value for the training step.
        """
        with tf.GradientTape() as tape:
            loss = self.loss_function(self.model, t)
        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return loss

    def _train_model(self, t, epochs):
        """
        Trains the model using a loop over the specified number of epochs.

        Parameters:
        t (tf.Tensor): Input tensor representing the training data.
        epochs (int): Number of training epochs.
        """
        optimizer = self.model.optimizer
        dt = t[1] - t[0]  # Compute time step size
        original_time = tf.identity(t)

        for epoch in range(epochs):
            if self.data_aug:
                t = self.data_augmentation(t, dt)

            loss = self._train_step(t)

            # Log training progress
            if epoch % 100 == 0 or epoch == epochs - 1:
                if self.data_aug:
                    val_loss = self.loss_function(self.model, original_time)
                    train_loss = self.loss_function(self.model, t)
                    print(f'Epoch {epoch}, Loss: {train_loss.numpy()}, Val Loss: {val_loss.numpy()}')
                else:
                    print(f'Epoch {epoch}, Loss: {loss.numpy()}')

            self.loss_history.append(loss.numpy().item())

            # Log the learning rate
            current_lr = optimizer.learning_rate
            if isinstance(current_lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr_value = current_lr(optimizer.iterations).numpy()
            else:
                lr_value = current_lr.numpy()
            self.lr_history.append(lr_value)

    def __call__(self, t):
        """
        Predicts the output for the given input time values using the trained model.

        Parameters:
        t (tf.Tensor): Input tensor representing time values.

        Returns:
        tf.Tensor: Model predictions for the input time values.
        """
        return self.model(t)

    def train(self, t, epochs):
        """
        Starts the model training process.

        Parameters:
        t (tf.Tensor): Input tensor representing the training data.
        epochs (int): Number of training epochs.

        Returns:
        tf.keras.Model: The trained model.
        """
        print("Starting model training...")
        t1 = time()
        self._train_model(t, epochs)
        t2 = time()
        print(f"Final model training completed with loss: {self.loss_history[-1]}")
        print(f'Time taken: {t2 - t1} seconds')
        self.time_to_train = t2 - t1
        return self.model

    def plot_loss(self, y_log=True):
        """
        Plots the training loss history over epochs.

        Parameters:
        y_log (bool): Whether to use a logarithmic scale for the y-axis (default is True).
        """
        plt.plot(self.loss_history, label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        if y_log:
            plt.yscale('log')
        plt.show()

    def plot_lr_history(self):
        """
        Plots the learning rate history over epochs.
        """
        plt.plot(self.lr_history, label="Learning Rate")
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Over Time")
        plt.legend()
        plt.show()

    @staticmethod
    def data_augmentation(self, input, dt, noise_factor=6):
        """
        Applies data augmentation by adding Gaussian noise to the input data.

        Parameters:
        input (tf.Tensor): Input tensor to augment.
        dt (float): Time step size.
        noise_factor (float): Factor controlling the standard deviation of the noise (default is 6).

        Returns:
        tf.Tensor: Augmented input tensor.
        """
        noise = tf.random.normal(shape=tf.shape(input), mean=0.0, stddev=dt / noise_factor)
        augmented_input = input + noise
        augmented_input = tf.clip_by_value(augmented_input, tf.reduce_min(input), tf.reduce_max(input))
        return augmented_input


class ModelTrainerOnPlateau:
    """
    A class for MLP model trainer with custom ReduceLROnPlateau scheduler.
    """
    def __init__(self, params, activation_function, loss_function, plateau_params: dict, data_aug=False):
        """
        Initializes the ModelTrainerOnPlateau class with specified parameters and loss function.

        Parameters:
        params (dict): Dictionary containing hyperparameters (n_layers, units, learning_rate).
        activation_function (str): Name of the activation function to use in the model layers (e.g., 'relu', 'tanh').
        loss_function (callable): Loss function for training.
        plateau_params (dict): Dictionary containing parameters for learning rate adjustment (patience, reduce_factor, min_lr, verbose).
        data_aug (bool): Whether to apply data augmentation during training (default is False).
        """
        self.params = params
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.data_aug = data_aug
        self.patience = plateau_params["patience"]
        self.reduce_factor = plateau_params["reduce_factor"]
        self.min_lr = plateau_params["min_lr"]
        self.verbose = plateau_params.get("verbose", True)
        self.model = self._build_final_model()
        self.loss_history = []
        self.lr_history = []
        self.tuning_time = ValueError("Training not yet launched")

    def _build_final_model(self):
        """
        Builds the final TensorFlow model based on the specified parameters.

        Returns:
        tf.keras.Model: Compiled Keras model.
        """
        model = tf.keras.Sequential()

        for _ in range(self.params["n_layers"]):
            model.add(tf.keras.layers.Dense(self.params["units"], activation=self.activation_function))
        model.add(tf.keras.layers.Dense(1))  # Output layer

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.params["learning_rate"]))
        return model

    def _train_model(self, t, epochs):
        """
        Trains the model using a loop over the specified number of epochs with a custom learning rate scheduler.

        Parameters:
        t (tf.Tensor): Input tensor representing the training data.
        epochs (int): Number of training epochs.
        """
        optimizer = self.model.optimizer
        dt = t[1] - t[0]  # Compute time step size
        original_time = tf.identity(t)

        best_loss = float('inf')
        epochs_without_improvement = 0
        lr_change_indicator = True

        for epoch in range(epochs):
            if self.data_aug:
                t = self.data_augmentation(t, dt)

            with tf.GradientTape() as tape:
                loss = self.loss_function(self.model, t)
            gradients = tape.gradient(loss, self.model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

            # Log training progress
            if epoch % 100 == 0 or epoch == epochs - 1:
                if self.data_aug:
                    val_loss = self.loss_function(self.model, original_time)
                    train_loss = self.loss_function(self.model, t)
                    print(f'Epoch {epoch}, Loss: {train_loss.numpy()}, Val Loss: {val_loss.numpy()}')
                else:
                    print(f'Epoch {epoch}, Loss: {loss.numpy()}')

            self.loss_history.append(loss.numpy().item())

            # Log the learning rate
            current_lr = optimizer.learning_rate
            if isinstance(current_lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr_value = current_lr(optimizer.iterations).numpy()
            else:
                lr_value = current_lr.numpy()
            self.lr_history.append(lr_value)

            if lr_change_indicator:
                # Custom Reduce on Plateau logic
                if loss < best_loss:
                    best_loss = loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    current_lr_value = optimizer.learning_rate.numpy()
                    new_lr = max(current_lr_value * self.reduce_factor, self.min_lr)
                    optimizer.learning_rate.assign(new_lr)
                    epochs_without_improvement = 0
                    if self.verbose:
                        print(f"Epoch {epoch}: Reducing learning rate to {new_lr}")
                    if new_lr == self.min_lr:
                        print(f"Epoch {epoch}: Min learning rate reached")
                        lr_change_indicator = False

    def __call__(self, time):
        """
        Predicts the output for the given input time values using the trained model.

        Parameters:
        time (tf.Tensor): Input tensor representing time values.

        Returns:
        tf.Tensor: Model predictions for the input time values.
        """
        return self.model(time)

    def train(self, t, epochs):
        """
        Starts the model training process.

        Parameters:
        t (tf.Tensor): Input tensor representing the training data.
        epochs (int): Number of training epochs.

        Returns:
        tf.keras.Model: The trained model.
        """
        print("Starting model training...")
        t1 = time()
        self._train_model(t, epochs)
        t2 = time()
        print(f"Final model training completed with loss: {self.loss_history[-1]}")
        print(f'Time taken: {t2 - t1} seconds')
        self.tuning_time = t2 - t1
        return self.model

    def plot_loss(self, y_log=True):
        """
        Plots the training loss history over epochs.

        Parameters:
        y_log (bool): Whether to use a logarithmic scale for the y-axis (default is True).
        """
        plt.plot(self.loss_history, label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        if y_log:
            plt.yscale('log')
        plt.show()

    def plot_lr_history(self, y_log=False):
        """
        Plots the learning rate history over epochs.

        Parameters:
        y_log (bool): Whether to use a logarithmic scale for the y-axis (default is False).
        """
        plt.plot(self.lr_history, label="Learning Rate")
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Over Time")
        plt.legend()
        if y_log:
            plt.yscale('log')
        plt.show()

    @staticmethod
    def data_augmentation(self, input, dt, noise_factor=6):
        """
        Applies data augmentation by adding Gaussian noise to the input data.

        Parameters:
        input (tf.Tensor): Input tensor to augment.
        dt (float): Time step size.
        noise_factor (float): Factor controlling the standard deviation of the noise (default is 6).

        Returns:
        tf.Tensor: Augmented input tensor.
        """
        noise = tf.random.normal(shape=tf.shape(input), mean=0.0, stddev=dt / noise_factor)
        augmented_input = input + noise
        augmented_input = tf.clip_by_value(augmented_input, tf.reduce_min(input), tf.reduce_max(input))
        return augmented_input
