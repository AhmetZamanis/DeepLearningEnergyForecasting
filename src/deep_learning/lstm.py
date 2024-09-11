import torch
import lightning as L

from src.deep_learning.quantile_loss import QuantileLoss


class StatefulQuantileLSTM(L.LightningModule):
    """
    Stateful LSTM forecasting model, returns quantile predictions.
    Input & output sequences are 3D tensors of shape (batch_size, timesteps, features).
    Hidden & cell states are retained & passed forward across training & inference batches.
    """

    # Initialize model
    def __init__(self, hyperparams_dict):
    """
    hyperparams_dict: Dictionary of hyperparameter values. See below for required hyperparameters.
    """
        
         # Delegate function to parent class
        super().__init__() 
        
        # Save external hyperparameters so they are available when loading saved models
        self.save_hyperparameters(logger = False) 

        # Define hyperparameters
        self.output_length = hyperparams_dict["output_length"]  # Length of output sequence
        self.input_size = hyperparams_dict["input_size"]  # Number of features (network inputs)
        self.horizon_start = hyperparams_dict["horizon_start"]  # Start of the forecast horizon relevant for loss computing
        self.quantiles = hyperparams_dict["quantiles"]  # Provide as list of floats: [0.025, 0.5, 0.975]
        self.learning_rate = hyperparams_dict["learning_rate"]
        self.lr_decay = hyperparams_dict["lr_decay"]  # Learning rate decay, 0.9 to 1 in practice, 1 = no decay
        self.num_layers = hyperparams_dict["num_layers"]  # Number of layers in the LSTM block
        self.hidden_size = hyperparams_dict["hidden_size"]  # Number of units in each LSTM block = LSTM block output size
        self.dropout_rate = hyperparams_dict["dropout_rate"]

        # Define architecture
        
        # LSTM input: input, (prev_hidden_states, prev_cell_states)
        # Shapes: (N, input_length, input_size), ((num_layers, N, hidden_size), (num_layers, N, hidden_size))
        self.lstm = torch.nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True
        )
        # LSTM output: output, (last_hidden_states, last_cell_states)
        # Shapes: (N, input_length, hidden_size), ((num_layers, N, hidden_size), (num_layers, N, hidden_size))
        # "output" has the last layer's output / final hidden state for each timestep in the input sequence.
        # The tuple of hidden & cell states have the last timestep's hidden & cell states for each LSTM layer.

        # Output layer input: LSTM output, shape (N, 1, hidden_size)
        # The final hidden state output for the last timestep in the input sequence.
        self.output_layer = torch.nn.Linear(
            in_features = self.hidden_size,
            out_features = len(self.quantiles)
        )
        # Output layer output: Quantile predictions, shape (N, n_quantiles)

        # Loss function: Quantile loss
        self.loss = QuantileLoss(quantiles = self.quantiles)
        self._median_quantile = np.median(self.quantiles)
        self._median_quantile_idx = self.quantiles.index(self._median_quantile)

        # Initialize hidden & cell state containers for statefulness
        self._last_hiddens_train = None
        self._last_cells_train = None
        self._final_hiddens_train = None
        self._final_cells_train = None

    # Define forward propagation
    # Pass prev_states as tuple (prev_hidden_states, prev_cell_states)
    def forward(self, input_chunk, prev_states = None): 

        # Pass inputs through LSTMs
        # If prev_states is not passed, they are automatically initialized as zeroes
        if prev_states == None:
            lstm_output, (last_hidden_states, last_cell_states) = self.lstm(input_chunk)
        else: 
            lstm_output, (last_hidden_states, last_cell_states) = self.lstm(input_chunk, prev_states)

        # Pass final LSTM output through output layer. Keep in mind this is the last layer's hidden state
        # output for the last timestep in the input sequence.
        preds = self.output_layer(lstm_output[:, -1, :])

        return last_hidden_states, last_cell_states, preds

    # Retain the computational graphs across backprop steps, so backprop across time can be performed
    # across batches. Demands more GPU memory.
    def backward(self, loss):

        # Free the computational graph on the last step of an epoch.
        # If this is not done, GPU memory will fill up, even after the model itself is deleted.
        if self.trainer.is_last_batch:
            loss.backward(retain_graph = False)

        # Retain the computational graph on all steps except last in an epoch.
        else:
            loss.backward(retain_graph = True)

    # Define training step
    def training_step(self, batch, batch_idx):

        # Initialize variables to record horizon, hidden & cell states, predictions
        h = 0
        prev_hiddens = []
        prev_cells = []
        batch_preds = []

        # Get inputs & outputs for first forecast step
        input_sequences, output_sequences = batch
        input_seq = input_sequences # Inputs of the forecast step 0. (N, input_length, input_size) 
        output_seq = output_sequences[:, 0, :] # Target & future covars of forecast step 0. Needed for later forecast steps. (N, input_size)

        # Perform training & recording for first forecast step
        # If a hidden & cell state is retained from the previous batch, use it. This will be the case for all batches except the first in an epoch.
        if self._last_hiddens_train == None:
            last_hidden_states, last_cell_states, preds = self.forward(input_seq)
        else:
            last_hidden_states, last_cell_states, preds = self.forward(
                input_seq, 
                prev_states = (self._last_hiddens_train, self._last_cells_train)
            )

        prev_hiddens.append(last_hidden_states) # 1-dimensional list. Each element has shape (num_layers, N, hidden_size)
        prev_cells.append(last_cell_states) # 1-dimensional list. Each element has shape (num_layers, N, hidden_size)
        batch_preds.append(preds) # 1-dimensional list. Each element has shape (N, n_quantiles)
        h += 1

        # Perform training & recording for remaining forecast steps
        while h < self.output_length:

            # Get inputs & outputs for forecast step h: 
            input_seq = torch.cat((
                input_seq[:, 1:, :], # Inputs of the previous forecast step, with the first row dropped. (N, input_length - 1, input_size)
                output_seq.unsqueeze(1) # Target & future covars of previous forecast step, the last row of the new input. (N, 1, input_size)
            ), dim = 1)
            
            output_seq = output_sequences[:, h, :] # Target & covars. of forecast step h. Needed for later forecast steps. (N, input_size)

            # Perform training & recording for forecast step h:
            last_hidden_states, last_cell_states, preds = self.forward(
                input_seq, 
                prev_states = (prev_hiddens[h-1], prev_cells[h-1])
            )
            prev_hiddens.append(last_hidden_states)
            prev_cells.append(last_cell_states)
            batch_preds.append(preds)
            h += 1

        # Calculate quantile loss for all forecast steps starting from the horizon
        # We're only interested in predicting from T+8 to T+32, but predictions at T+1 will depend on predictions at T,
        # so calculating loss over all h steps is probably ideal (horizon_start = 0). The code below supports either method.
        preds_horizon = batch_preds[self.horizon_start:] # List length (output_length - horizon_start). Each element has shape (N, n_quantiles).
        preds_horizon = torch.stack(preds_horizon, dim = 1) # Shape (N, output_length - horizon_start, n_quantiles)
        targets_horizon = output_sequences[:, self.horizon_start:, 0] # Target values from horizon to end of sequence. Shape(N, output_length - horizon_start)
        loss = self.loss.loss(preds_horizon, targets_horizon) # Quantile losses for each batch & timestep. Shape (N, output_length - horizon_start, n_quantiles)

        # Reduce the quantile loss
        # A lot of room for experimentation here. Below is the typical application for RNNs. See: 
        # (https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks#overview)
        loss_reduced = loss.mean(dim = 2) # Average over quantiles. Yields (N, output_length - horizon_start)
        loss_reduced = loss_reduced.sum(dim = 1) # Sum over forecast steps. Yields (N)
        loss_reduced = loss_reduced.mean() # Average over batches. Yields scalar loss for backpropagation.
        
        # Log the training loss
        self.log("train_loss", loss_reduced, on_step = True, on_epoch = True, prog_bar = True, logger = False)

        # Update last hidden & cell states from training (for within-epoch use)
        self._last_hiddens_train = prev_hiddens[-1]
        self._last_cells_train = prev_cells[-1]

        # Update final hidden & cell states from training (for inference)
        self._final_hiddens_train = prev_hiddens[-1]
        self._final_cells_train = prev_cells[-1]

        return loss_reduced

    # When a training epoch ends, flush the last hidden & cell states.
    # Final hidden & cell states remain for inference.
    def on_train_epoch_end(self):
        self._last_hiddens_train = None
        self._last_cells_train = None

    # Method to flush the final hidden & cell states left from training, if desired
    def reset_states(self):
        self._final_hiddens_train = None
        self._final_cells_train = None

    # Define validation_step
    def validation_step(self, batch, batch_idx):

        # Initialize variables to record horizon, hidden & cell states, predictions
        h = 0
        prev_hiddens = []
        prev_cells = []
        batch_preds = []

        # Get inputs & outputs for first forecast step
        input_sequences, output_sequences = batch
        input_seq = input_sequences # Inputs of the forecast step 0. (N, input_length, input_size) 
        output_seq = output_sequences[:, 0, 1:] # Future covars of forecast step 0. Needed for later forecast steps. (N, input_size - 1)

        # Perform validation & recording for first forecast step
        # If a hidden & cell state is retained from training, use it.
        if self._final_hiddens_train == None:
            last_hidden_states, last_cell_states, preds = self.forward(input_seq)
        else:
            last_hidden_states, last_cell_states, preds = self.forward(
                input_seq, 
                prev_states = (self._final_hiddens_train, self._final_cells_train)
            )

        prev_hiddens.append(last_hidden_states) # 1-dimensional list. Each element has shape (num_layers, N, hidden_size)
        prev_cells.append(last_cell_states) # 1-dimensional list. Each element has shape (num_layers, N, hidden_size)
        batch_preds.append(preds) # 1-dimensional list. Each element has shape (N, n_quantiles)
        h += 1

        # Perform validation & recording for remaining forecast steps
        while h < self.output_length:

            # Get inputs & outputs for forecast step h:
            output_seq = torch.cat((
                batch_preds[h-1][:, self._median_quantile_idx].unsqueeze(1), # Point prediction of forecast step 0. (N, 1)
                output_seq # Future covars of forecast step 0. (N, input_size - 1)
            ), dim = 1)
            
            input_seq = torch.cat((
                input_seq[:, 1:, :], # Inputs of the previous forecast step, with the first row dropped. (N, input_length - 1, input_size)
                output_seq.unsqueeze(1) # Prediction & future covars of previous forecast step, the last row of the new input. (N, 1, input_size)
            ), dim = 1)
            
            output_seq = output_sequences[:, h, 1:] # Future covars. of forecast step h. Needed for later forecast steps. (N, input_size-1)

            # Perform training & recording for forecast step h:
            last_hidden_states, last_cell_states, preds = self.forward(
                input_seq, 
                prev_states = (prev_hiddens[h-1], prev_cells[h-1])
            )
            prev_hiddens.append(last_hidden_states)
            prev_cells.append(last_cell_states)
            batch_preds.append(preds)
            h += 1

        # Calculate loss for forecast steps starting from horizon
        preds_horizon = batch_preds[self.horizon_start:] # List length (output_length - horizon_start). Each element has shape (N, n_quantiles).
        preds_horizon = torch.stack(preds_horizon, dim = 1) # Shape (N, output_length - horizon_start, n_quantiles)
        targets_horizon = output_sequences[:, self.horizon_start:, 0] # Target values from horizon to end of sequence. Shape(N, output_length - horizon_start)
        loss = self.loss.loss(preds_horizon, targets_horizon) # Quantile losses for each batch & timestep. Shape (N, output_length - horizon_start, n_quantiles)

        # Reduce the quantile loss
        loss_reduced = loss.mean(dim = 2) # Average over quantiles. Yields (N, output_length - horizon_start)
        loss_reduced = loss_reduced.sum(dim = 1) # Sum over forecast steps. Yields (N)
        loss_reduced = loss_reduced.mean() # Average over batches. Yields scalar loss for backpropagation.

        # Log the val. loss
        self.log("val_loss", loss_reduced, on_step = True, on_epoch = True, prog_bar = True, logger = False)

        return loss_reduced

    # Define prediction_step
    def predict_step(self, batch, batch_idx):

        # Initialize variables to record horizon, hidden & cell states, predictions
        h = 0
        prev_hiddens = []
        prev_cells = []
        batch_preds = []

        # Get inputs & outputs for first forecast step
        input_sequences, output_sequences = batch
        input_seq = input_sequences # Inputs of the forecast step 0. (N, input_length, input_size) 
        output_seq = output_sequences[:, 0, 1:] # Future covars of forecast step 0. Needed for later forecast steps. (N, input_size - 1)

        # Perform prediction & recording for first forecast step
        # If a hidden & cell state is retained from training, use it.
        if self._final_hiddens_train == None:
            last_hidden_states, last_cell_states, preds = self.forward(input_seq)
        else:
            last_hidden_states, last_cell_states, preds = self.forward(
                input_seq, 
                prev_states = (self._final_hiddens_train, self._final_cells_train)
            )

        prev_hiddens.append(last_hidden_states) # 1-dimensional list. Each element has shape (num_layers, N, hidden_size)
        prev_cells.append(last_cell_states) # 1-dimensional list. Each element has shape (num_layers, N, hidden_size)
        batch_preds.append(preds) # 1-dimensional list. Each element has shape (N, n_quantiles)
        h += 1

        # Perform prediction & recording for remaining forecast steps
        while h < self.output_length:

            # Get inputs & outputs for forecast step h:
            output_seq = torch.cat((
                batch_preds[h-1][:, self._median_quantile_idx].unsqueeze(1), # Point prediction of forecast step 0. (N, 1)
                output_seq # Future covars of forecast step 0. (N, input_size - 1)
            ), dim = 1)
            
            input_seq = torch.cat((
                input_seq[:, 1:, :], # Inputs of the previous forecast step, with the first row dropped. (N, input_length - 1, input_size)
                output_seq.unsqueeze(1) # Prediction & future covars of previous forecast step, the last row of the new input. (N, 1, input_size)
            ), dim = 1)
            
            output_seq = output_sequences[:, h, 1:] # Future covars. of forecast step h. Needed for later forecast steps. (N, input_size-1)

            # Perform training & recording for forecast step h:
            last_hidden_states, last_cell_states, preds = self.forward(
                input_seq, 
                prev_states = (prev_hiddens[h-1], prev_cells[h-1])
            )
            prev_hiddens.append(last_hidden_states)
            prev_cells.append(last_cell_states)
            batch_preds.append(preds)
            h += 1

        # Reshape predictions
        preds = torch.stack(batch_preds, dim = 0) # Yields shape (output_length, N, n_quantiles)
        preds = torch.movedim(preds, 1, 0) # Yields shape (N, output_length, n_quantiles)

        return preds

    # Define optimizer & learning rate scheduler
    def configure_optimizers(self):

        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        # Exponential LR scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
          optimizer, gamma = self.lr_decay) 
        
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
          "scheduler": lr_scheduler
          }
        }
