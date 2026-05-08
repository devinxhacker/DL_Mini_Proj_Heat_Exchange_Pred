"""
Physics-Informed LSTM Model for Heat Exchanger Temperature Prediction
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# reduce verbose TensorFlow logs (must be set before importing TF)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    # quiet retracing/info logs
    try:
        tf.get_logger().setLevel('ERROR')
    except Exception:
        pass
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def physics_loss(y_true, y_pred):
    """Custom physics-informed loss function"""
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    hot_outlet_pred = y_pred[:, 0]
    cold_outlet_pred = y_pred[:, 1]
    physics_penalty = tf.reduce_mean(
        tf.maximum(0.0, -hot_outlet_pred) +
        tf.maximum(0.0, -cold_outlet_pred)
    )
    total_loss = mse_loss + 0.1 * physics_penalty
    return total_loss


class PhysicsInformedLSTM:
    def __init__(
        self,
        sequence_length=10,
        lstm_units=128,
        learning_rate=0.001,
        physics_weight: float = 1.0,
        cp_hot_kj_kgk: float = 4.18,
        cp_cold_kj_kgk: float = 4.18,
        cold_inlet_temperature_k: float = 293.15,
        use_hard_energy_balance: bool = False,
        residual_learning: bool = True,
        use_bidirectional: bool = True,
    ):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.physics_weight = float(physics_weight)
        self.cp_hot = float(cp_hot_kj_kgk)
        self.cp_cold = float(cp_cold_kj_kgk)
        self.cold_inlet_temperature_k = float(cold_inlet_temperature_k)
        self.use_hard_energy_balance = bool(use_hard_energy_balance)
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        # scaler for residual target (hot residual when residual_learning=True)
        self.scaler_residual = StandardScaler()
        self.history = None
        self.residual_learning = bool(residual_learning)
        self.use_bidirectional = bool(use_bidirectional)
        
    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        # stack of LSTM layers (optionally bidirectional)
        if self.use_bidirectional:
            x = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))(inputs)
            x = layers.Dropout(0.2)(x)
            x = layers.Bidirectional(layers.LSTM(self.lstm_units // 2, return_sequences=True))(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Bidirectional(layers.LSTM(self.lstm_units // 4, return_sequences=False))(x)
            x = layers.Dropout(0.2)(x)
        else:
            x = layers.LSTM(self.lstm_units, return_sequences=True)(inputs)
            x = layers.Dropout(0.2)(x)
            x = layers.LSTM(self.lstm_units // 2, return_sequences=True)(x)
            x = layers.Dropout(0.2)(x)
            x = layers.LSTM(self.lstm_units // 4, return_sequences=False)(x)
            x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(16, activation='relu')(x)
        # if using hard energy-balance, predict only hot outlet and compute cold outlet from energy balance
        out_dims = 1 if self.use_hard_energy_balance or self.residual_learning else 2
        outputs = layers.Dense(out_dims, activation='linear')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        # compile with a placeholder loss (we use custom training loop when training)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        self.model = model
        return model
    
    def prepare_data(self, df_in):
        df = df_in.copy()
        if 'cold_inlet_temperature_k' not in df.columns:
            df['cold_inlet_temperature_k'] = self.cold_inlet_temperature_k
        input_features = [
            'hot_inlet_temperature_k',
            'cold_inlet_temperature_k',
            'cold_inlet_mass_flow_kg_s',
            'hx_1_heat_load_kw',
            'hot_outlet_pressure_pa',
            'cold_outlet_pressure_pa',
            'hot_outlet_mass_flow_kg_s',
            'cold_outlet_mass_flow_kg_s',
            'hx_1_logarithmic_mean_temperature_difference_lmtd_k'
        ]
        output_features = [
            'hot_outlet_temperature_k',
            'cold_outlet_temperature_k'
        ]
        X = df[input_features].values
        y = df[output_features].values
        return X, y
    
    def train(self, X_train, y_train, X_val, y_val, epochs=1000, batch_size=32, verbose=1):
        # Prepare scaled and raw copies so we can compute physics loss on physical units
        X_train_orig = np.asarray(X_train, dtype=float)
        X_val_orig = np.asarray(X_val, dtype=float)

        X_train_scaled = self.scaler_X.fit_transform(X_train_orig.reshape(-1, X_train_orig.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(X_train_orig.shape)
        X_val_scaled = self.scaler_X.transform(X_val_orig.reshape(-1, X_val_orig.shape[-1]))
        X_val_scaled = X_val_scaled.reshape(X_val_orig.shape)

        y_train_orig = np.asarray(y_train, dtype=float)
        y_val_orig = np.asarray(y_val, dtype=float)
        # If residual learning, compute residual targets for hot outlet: residual = true_hot - baseline_hot
        # Indices in the input feature vector per prepare_data()
        # 0: hot_inlet_temperature_k
        # 1: cold_inlet_temperature_k
        # 2: cold_inlet_mass_flow_kg_s
        # 6: hot_outlet_mass_flow_kg_s or assumed hot flow
        hot_inlet_idx = 0
        cold_inlet_idx = 1
        cold_flow_idx = 2
        hot_flow_idx = 6

        if self.residual_learning:
            # compute baseline hot from last-step inputs for each sequence
            def compute_baseline(X_orig):
                hot_inlet = X_orig[:, -1, hot_inlet_idx]
                hot_flow = X_orig[:, -1, hot_flow_idx]
                heat_load = X_orig[:, -1, 3]
                # baseline_hot = hot_inlet - heat_load / (hot_flow * cp_hot)
                denom = (hot_flow * float(self.cp_hot)) + 1e-12
                baseline_hot = hot_inlet - (heat_load / denom)
                return baseline_hot.reshape(-1, 1)

            # compute baselines
            baseline_train = compute_baseline(X_train_orig)
            baseline_val = compute_baseline(X_val_orig)

            y_train_residual = (y_train_orig[:, 0].reshape(-1, 1) - baseline_train)
            y_val_residual = (y_val_orig[:, 0].reshape(-1, 1) - baseline_val)

            y_train_scaled = self.scaler_residual.fit_transform(y_train_residual)
            y_val_scaled = self.scaler_residual.transform(y_val_residual)
        else:
            y_train_scaled = self.scaler_y.fit_transform(y_train_orig)
            y_val_scaled = self.scaler_y.transform(y_val_orig)

        # Create datasets that yield both scaled inputs (for the model) and original inputs (for physics)
        # Use drop_remainder so batch shapes are consistent (avoids retracing/OutOfRange warnings)
        # keep partial batches for validation so small val sets are not dropped
        train_ds = tf.data.Dataset.from_tensor_slices(
            (X_train_scaled, X_train_orig, y_train_scaled, y_train_orig)
        ).shuffle(1024).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices(
            (X_val_scaled, X_val_orig, y_val_scaled, y_val_orig)
        ).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        train_history = {"loss": [], "val_loss": []}

        best_val = np.inf
        patience = 15
        wait = 0

        # Indices in the input feature vector per prepare_data()
        # 0: hot_inlet_temperature_k
        # 1: cold_inlet_temperature_k
        # 2: cold_inlet_mass_flow_kg_s
        # 6: hot_outlet_mass_flow_kg_s or assumed hot flow
        hot_inlet_idx = 0
        cold_inlet_idx = 1
        cold_flow_idx = 2
        hot_flow_idx = 6

        for epoch in range(epochs):
            # Training
            epoch_losses = []
            for X_scaled_batch, X_orig_batch, y_scaled_batch, y_orig_batch in train_ds:
                with tf.GradientTape() as tape:
                    preds_scaled = self.model(X_scaled_batch, training=True)

                    # Build predicted physical outputs depending on mode
                    y_true_phys = tf.cast(y_orig_batch, tf.float32)
                    if self.residual_learning:
                        # preds_scaled are residual predictions for hot outlet
                        res_scale = tf.constant(self.scaler_residual.scale_[0], dtype=tf.float32)
                        res_mean = tf.constant(self.scaler_residual.mean_[0], dtype=tf.float32)
                        hot_residual = tf.reshape(preds_scaled[:, 0] * res_scale + res_mean, [-1])

                        # compute baseline from last timestep inputs
                        hot_inlet = tf.cast(X_orig_batch[:, -1, hot_inlet_idx], tf.float32)
                        hot_flow = tf.cast(X_orig_batch[:, -1, hot_flow_idx], tf.float32)
                        heat_load = tf.cast(X_orig_batch[:, -1, 3], tf.float32)
                        denom = (hot_flow * float(self.cp_hot)) + 1e-12
                        baseline_hot = hot_inlet - (heat_load / denom)

                        hot_pred_phys = baseline_hot + hot_residual

                        # derive cold outlet via energy balance
                        cold_inlet = tf.cast(X_orig_batch[:, -1, cold_inlet_idx], tf.float32)
                        cold_flow = tf.cast(X_orig_batch[:, -1, cold_flow_idx], tf.float32)
                        Q_hot = hot_flow * float(self.cp_hot) * (hot_inlet - hot_pred_phys)
                        cold_pred_phys = Q_hot / (cold_flow * float(self.cp_cold) + 1e-12) + cold_inlet
                        preds_phys = tf.stack([hot_pred_phys, cold_pred_phys], axis=1)
                    elif self.use_hard_energy_balance:
                        # preds_scaled: (batch,1) => hot outlet only (absolute)
                        hot_scale = float(self.scaler_y.scale_[0])
                        hot_mean = float(self.scaler_y.mean_[0])
                        hot_pred_phys = tf.reshape(preds_scaled[:, 0] * hot_scale + hot_mean, [-1])

                        # Use last-timestep inputs to compute cold outlet from energy balance
                        hot_inlet = tf.cast(X_orig_batch[:, -1, hot_inlet_idx], tf.float32)
                        cold_inlet = tf.cast(X_orig_batch[:, -1, cold_inlet_idx], tf.float32)
                        cold_flow = tf.cast(X_orig_batch[:, -1, cold_flow_idx], tf.float32)
                        hot_flow = tf.cast(X_orig_batch[:, -1, hot_flow_idx], tf.float32)

                        Q_hot = hot_flow * float(self.cp_hot) * (hot_inlet - hot_pred_phys)
                        cold_pred_phys = Q_hot / (cold_flow * float(self.cp_cold) + 1e-12) + cold_inlet
                        preds_phys = tf.stack([hot_pred_phys, cold_pred_phys], axis=1)
                    else:
                        # Unscale predictions to physical units: y = y_scaled * scale + mean
                        y_scale = tf.constant(self.scaler_y.scale_, dtype=tf.float32)
                        y_mean = tf.constant(self.scaler_y.mean_, dtype=tf.float32)
                        preds_phys = preds_scaled * y_scale + y_mean

                    # Compute MSE in physical units
                    mse_loss = tf.reduce_mean(tf.square(y_true_phys - preds_phys))

                    # Compute energy balance gap per sample (compare Q_hot and Q_cold)
                    hot_inlet = tf.cast(X_orig_batch[:, -1, hot_inlet_idx], tf.float32)
                    cold_inlet = tf.cast(X_orig_batch[:, -1, cold_inlet_idx], tf.float32)
                    cold_flow = tf.cast(X_orig_batch[:, -1, cold_flow_idx], tf.float32)
                    hot_flow = tf.cast(X_orig_batch[:, -1, hot_flow_idx], tf.float32)
                    heat_load = tf.cast(X_orig_batch[:, -1, 3], tf.float32)

                    hot_outlet_pred = preds_phys[:, 0]
                    cold_outlet_pred = preds_phys[:, 1]

                    Q_hot = hot_flow * float(self.cp_hot) * (hot_inlet - hot_outlet_pred)
                    Q_cold = cold_flow * float(self.cp_cold) * (cold_outlet_pred - cold_inlet)

                    # normalize energy gap by heat load magnitude to keep units comparable to temperature MSE
                    denom = tf.abs(heat_load) + 1e-6
                    rel_gap = tf.abs(Q_hot - Q_cold) / denom
                    physics_penalty = tf.reduce_mean(rel_gap)

                    total_loss = mse_loss + float(self.physics_weight) * physics_penalty

                grads = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                epoch_losses.append(float(total_loss.numpy()))

            # Validation
            val_losses = []
            for X_scaled_batch, X_orig_batch, y_scaled_batch, y_orig_batch in val_ds:
                preds_scaled = self.model(X_scaled_batch, training=False)
                y_true_phys = tf.cast(y_orig_batch, tf.float32)
                if self.residual_learning:
                    res_scale = tf.constant(self.scaler_residual.scale_[0], dtype=tf.float32)
                    res_mean = tf.constant(self.scaler_residual.mean_[0], dtype=tf.float32)
                    hot_residual = tf.reshape(preds_scaled[:, 0] * res_scale + res_mean, [-1])
                    hot_inlet = tf.cast(X_orig_batch[:, -1, hot_inlet_idx], tf.float32)
                    hot_flow = tf.cast(X_orig_batch[:, -1, hot_flow_idx], tf.float32)
                    heat_load = tf.cast(X_orig_batch[:, -1, 3], tf.float32)
                    denom = (hot_flow * float(self.cp_hot)) + 1e-12
                    baseline_hot = hot_inlet - (heat_load / denom)
                    hot_pred_phys = baseline_hot + hot_residual
                    cold_inlet = tf.cast(X_orig_batch[:, -1, cold_inlet_idx], tf.float32)
                    cold_flow = tf.cast(X_orig_batch[:, -1, cold_flow_idx], tf.float32)
                    Q_hot = hot_flow * float(self.cp_hot) * (hot_inlet - hot_pred_phys)
                    cold_pred_phys = Q_hot / (cold_flow * float(self.cp_cold) + 1e-12) + cold_inlet
                    preds_phys = tf.stack([hot_pred_phys, cold_pred_phys], axis=1)
                elif self.use_hard_energy_balance:
                    hot_scale = float(self.scaler_y.scale_[0])
                    hot_mean = float(self.scaler_y.mean_[0])
                    hot_pred_phys = tf.reshape(preds_scaled[:, 0] * hot_scale + hot_mean, [-1])
                    hot_inlet = tf.cast(X_orig_batch[:, -1, hot_inlet_idx], tf.float32)
                    cold_inlet = tf.cast(X_orig_batch[:, -1, cold_inlet_idx], tf.float32)
                    cold_flow = tf.cast(X_orig_batch[:, -1, cold_flow_idx], tf.float32)
                    hot_flow = tf.cast(X_orig_batch[:, -1, hot_flow_idx], tf.float32)
                    Q_hot = hot_flow * float(self.cp_hot) * (hot_inlet - hot_pred_phys)
                    cold_pred_phys = Q_hot / (cold_flow * float(self.cp_cold) + 1e-12) + cold_inlet
                    preds_phys = tf.stack([hot_pred_phys, cold_pred_phys], axis=1)
                else:
                    y_scale = tf.constant(self.scaler_y.scale_, dtype=tf.float32)
                    y_mean = tf.constant(self.scaler_y.mean_, dtype=tf.float32)
                    preds_phys = preds_scaled * y_scale + y_mean

                mse_loss = tf.reduce_mean(tf.square(y_true_phys - preds_phys))
                hot_inlet = tf.cast(X_orig_batch[:, -1, hot_inlet_idx], tf.float32)
                cold_inlet = tf.cast(X_orig_batch[:, -1, cold_inlet_idx], tf.float32)
                cold_flow = tf.cast(X_orig_batch[:, -1, cold_flow_idx], tf.float32)
                hot_flow = tf.cast(X_orig_batch[:, -1, hot_flow_idx], tf.float32)
                heat_load = tf.cast(X_orig_batch[:, -1, 3], tf.float32)
                hot_outlet_pred = preds_phys[:, 0]
                cold_outlet_pred = preds_phys[:, 1]
                Q_hot = hot_flow * float(self.cp_hot) * (hot_inlet - hot_outlet_pred)
                Q_cold = cold_flow * float(self.cp_cold) * (cold_outlet_pred - cold_inlet)
                denom = tf.abs(heat_load) + 1e-6
                rel_gap = tf.abs(Q_hot - Q_cold) / denom
                physics_penalty = tf.reduce_mean(rel_gap)
                val_loss = mse_loss + float(self.physics_weight) * physics_penalty
                val_losses.append(float(val_loss.numpy()))

            avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            train_history["loss"].append(avg_train_loss)
            train_history["val_loss"].append(avg_val_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss={avg_train_loss:.4f} - val_loss={avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val - 1e-6:
                best_val = avg_val_loss
                wait = 0
                # Save best weights
                best_weights = self.model.get_weights()
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print("Early stopping: restoring best weights")
                    self.model.set_weights(best_weights)
                    break

        # Store history in same shape as Keras
        class H:
            history = train_history

        self.history = H.history
        return self.history
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1]))
        X_scaled = X_scaled.reshape(X.shape)
        # Use direct forward pass instead of model.predict to avoid tf.function retracing noise
        y_pred_scaled = self.model(X_scaled, training=False).numpy()
        if self.residual_learning:
            # model outputs scaled residual for hot outlet
            hot_residual = (y_pred_scaled[:, 0] * self.scaler_residual.scale_[0]) + self.scaler_residual.mean_[0]
            # compute baseline from last-step inputs
            hot_inlet = X[:, -1, 0]
            hot_flow = X[:, -1, 6]
            heat_load = X[:, -1, 3]
            denom = (hot_flow * float(self.cp_hot)) + 1e-12
            baseline_hot = hot_inlet - (heat_load / denom)
            hot_pred = baseline_hot + hot_residual
            # derive cold outlet
            cold_inlet = X[:, -1, 1]
            cold_flow = X[:, -1, 2]
            q_hot = hot_flow * float(self.cp_hot) * (hot_inlet - hot_pred)
            cold_pred = q_hot / (cold_flow * float(self.cp_cold) + 1e-12) + cold_inlet
            return np.stack([hot_pred, cold_pred], axis=1)

        if self.use_hard_energy_balance or y_pred_scaled.shape[1] == 1:
            # Hard-balance mode predicts hot outlet only; callers derive cold outlet separately.
            hot_scale = float(self.scaler_y.scale_[0])
            hot_mean = float(self.scaler_y.mean_[0])
            hot_pred = (y_pred_scaled[:, 0] * hot_scale) + hot_mean
            return hot_pred.reshape(-1, 1)

        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred
