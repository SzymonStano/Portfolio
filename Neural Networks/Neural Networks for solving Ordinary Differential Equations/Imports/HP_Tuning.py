import tensorflow as tf
import optuna
from time import time


class HpTuningCosine:
    """
    A class for hyperparameter tuning for MLP model with Cosine Decay scheduler.
    """
    def __init__(self, t, activation_function, loss_function, epochs=50,
                 decay_steps=500, n_trials=50, n_layers=(1, 15), n_units=(32, 256)):
        """
        Initializes the HpTuningCosine class for hyperparameter tuning and training.

        Parameters:
        t (tf.Tensor): Time-series data used for training.
        activation_function (str): Activation function for the model (e.g., 'relu', 'tanh').
        loss_function (callable): Loss function to minimize during training.
        epochs (int): Number of training epochs. Default is 50.
        decay_steps (int): Steps for cosine decay of learning rate. Default is 500.
        n_trials (int): Number of trials for hyperparameter tuning. Default is 50.
        n_layers (tuple): Range for the number of layers in the model (min, max). Default is (1, 15).
        n_units (tuple): Range for the number of units per layer (min, max). Default is (32, 256).
        """
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.epochs = epochs
        self.t = t
        self.n_trials = n_trials
        self.decay_steps = decay_steps
        self.n_layers = n_layers
        self.n_units = n_units
        self.tuning_time = ValueError("Tuning not yet launched")

    def hp_tuning(self):
        """
        Conducts hyperparameter tuning using Optuna.

        The method evaluates multiple combinations of hyperparameters (number of layers,
        number of units, learning rate) to find the best configuration for the model.

        Returns:
        dict: Best hyperparameter configuration found during tuning.
        """
        t1 = time()

        def objective(trial):
            """
            Defines the objective function for Optuna's optimization.

            Parameters:
            trial (optuna.trial.Trial): A trial object provided by Optuna.

            Returns:
            float: Final loss value of the model with current hyperparameters.
            """
            # Define hyperparameters
            n_layers = trial.suggest_int("n_layers", self.n_layers[0], self.n_layers[1])
            units = trial.suggest_int("units", self.n_units[0], self.n_units[1])
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

            # Build the model
            model = tf.keras.Sequential()
            for _ in range(n_layers):
                model.add(tf.keras.layers.Dense(units, activation=self.activation_function))
            model.add(tf.keras.layers.Dense(1))

            model.build(input_shape=(None, self.t.shape[1]))  # Assuming ts has shape (batch_size, feature_dim)

            # Learning rate schedule
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=learning_rate,
                decay_steps=self.decay_steps,
                alpha=0.1
            )

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))

            # Train the model
            final_loss = self.train_model(model, self.t, hp_tuning=True)

            return final_loss

        # Create and optimize study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)

        # Record results
        t2 = time()
        print(f"Tuning time: {t2 - t1}s")
        print("Best params:", study.best_params)
        self.tuning_time = t2 - t1
        return study.best_params

    def train_model(self, model, ts, data_aug=False, hp_tuning=False):
        """
        Trains the TensorFlow model using a custom training loop.

        Parameters:
        model (tf.keras.Model): TensorFlow model to be trained.
        ts (tf.Tensor): Time-series data for training.
        data_aug (bool): Whether to apply data augmentation. Default is False.
        hp_tuning (bool): Whether this training is part of hyperparameter tuning. Default is False.

        Returns:
        float: Final loss value after training.
        """
        optimizer = model.optimizer

        for epoch in range(self.epochs):
            with tf.GradientTape() as tape:
                loss = self.loss_function(model, ts)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        return loss.numpy()

    @staticmethod
    def data_augmentation(input, dt, noise_factor=6):
        """
        Applies data augmentation by adding noise to the input data.

        Parameters:
        input (tf.Tensor): Input tensor (e.g., time-series data).
        dt (float): Time step between consecutive data points.
        noise_factor (int): Controls the noise level. Higher values result in less noise. Default is 6.

        Returns:
        tf.Tensor: Augmented input data with added noise.
        """
        noise = tf.random.normal(shape=tf.shape(input), mean=0.0, stddev=dt / noise_factor)
        augmented_input = input + noise
        augmented_input = tf.clip_by_value(augmented_input, tf.reduce_min(input), tf.reduce_max(input))
        return augmented_input


class HpTuningOnPlateau:
    """
    A class for hyperparameter tuning for MLP with custom ReduceLROnPlateau scheduler.
    """
    def __init__(self, t, activation_function, loss_function, plateau_params: dict, epochs=50, n_trials=50, n_layers=(1, 15), n_units=(32, 256)):
        """
        Initializes the HpTuningOnPlateau class for hyperparameter tuning and training.

        Parameters:
        t (tf.Tensor): Time-series data used for training.
        activation_function (str): Activation function for the model (e.g., 'relu', 'tanh').
        loss_function (callable): Loss function to minimize during training.
        plateau_params (dict): Parameters for learning rate adjustment on plateau.
            - patience (int): Number of epochs without improvement before reducing the learning rate.
            - reduce_factor (float): Factor by which to reduce the learning rate.
            - min_lr (float): Minimum allowed learning rate.
            - verbose (bool): Whether to print updates during training.
        epochs (int): Number of training epochs. Default is 50.
        n_trials (int): Number of trials for hyperparameter tuning. Default is 50.
        n_layers (tuple): Range for the number of layers in the model (min, max). Default is (1, 15).
        n_units (tuple): Range for the number of units per layer (min, max). Default is (32, 256).
        """
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.epochs = epochs
        self.t = t
        self.n_trials = n_trials
        self.n_layers = n_layers
        self.n_units = n_units
        self.patience = plateau_params["patience"]
        self.reduce_factor = plateau_params["reduce_factor"]
        self.min_lr = plateau_params["min_lr"]
        self.verbose = plateau_params["verbose"]
        self.tuning_time = ValueError("Tuning not yet launched")

    def hp_tuning(self):
        """
        Conducts hyperparameter tuning using Optuna.

        The method evaluates multiple combinations of hyperparameters (number of layers,
        number of units, learning rate) to find the best configuration for the model.

        Returns:
        dict: Best hyperparameter configuration found during tuning.
        """
        t1 = time()

        def objective(trial):
            """
            Defines the objective function for Optuna's optimization.

            Parameters:
            trial (optuna.trial.Trial): A trial object provided by Optuna.

            Returns:
            float: Final loss value of the model with current hyperparameters.
            """
            # Define hyperparameters
            n_layers = trial.suggest_int("n_layers", self.n_layers[0], self.n_layers[1])
            units = trial.suggest_int("units", self.n_units[0], self.n_units[1])
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

            # Build the model
            model = tf.keras.Sequential()
            for _ in range(n_layers):
                model.add(tf.keras.layers.Dense(units, activation=self.activation_function))
            model.add(tf.keras.layers.Dense(1))

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

            # Train the model
            final_loss = self.train_model(model, self.t, hp_tuning=True)

            return final_loss

        # Create and optimize study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)

        # Record results
        t2 = time()
        print(f"Tuning time: {t2 - t1}s")
        print("Best params:", study.best_params)
        self.tuning_time = t2 - t1
        return study.best_params

    def train_model(self, model, ts, data_aug=False, hp_tuning=False):
        """
        Trains the TensorFlow model using a custom training loop.

        Parameters:
        model (tf.keras.Model): TensorFlow model to be trained.
        ts (tf.Tensor): Time-series data for training.
        data_aug (bool): Whether to apply data augmentation. Default is False.
        hp_tuning (bool): Whether this training is part of hyperparameter tuning. Default is False.

        Returns:
        float: Final loss value after training.
        """
        optimizer = model.optimizer

        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            with tf.GradientTape() as tape:
                loss = self.loss_function(model, ts)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            if loss < best_loss:
                best_loss = loss
                epochs_without_improvement = 0  # Reset patience counter
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                current_lr_value = optimizer.learning_rate.numpy()
                new_lr = max(current_lr_value * self.reduce_factor, self.min_lr)
                optimizer.learning_rate.assign(new_lr)  # Update LR
                epochs_without_improvement = 0  # Reset patience counter
                if self.verbose:
                    print(f"Epoch {epoch}: Reducing learning rate to {new_lr}")

        return loss.numpy()

    @staticmethod
    def data_augmentation(input, dt, noise_factor=6):
        """
        Applies data augmentation by adding noise to the input data.

        Parameters:
        input (tf.Tensor): Input tensor (e.g., time-series data).
        dt (float): Time step between consecutive data points.
        noise_factor (int): Controls the noise level. Higher values result in less noise. Default is 6.

        Returns:
        tf.Tensor: Augmented input data with added noise.
        """
        noise = tf.random.normal(shape=tf.shape(input), mean=0.0, stddev=dt / noise_factor)
        augmented_input = input + noise
        augmented_input = tf.clip_by_value(augmented_input, tf.reduce_min(input), tf.reduce_max(input))
        return augmented_input
