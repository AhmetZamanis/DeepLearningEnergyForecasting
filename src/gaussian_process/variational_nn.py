import gpytorch
from gpytorch.kernels import ScaleKernel, LinearKernel, PeriodicKernel, AdditiveKernel
from gpytorch.priors import NormalPrior
from gpytorch.variational import NNVariationalStrategy, MeanFieldVariationalDistribution


# Variational nearest neightbors GP model class (inducing points = X_train, unbatched)
class VNNGPModel(gpytorch.models.ApproximateGP):

    def __init__(self, inducing_points, batch_size, k = 256, cuda = True):

        if cuda:
            inducing_points = inducing_points.cuda()

        # Initialize parameters
        self.n_inducing_points = inducing_points.size(0)
        self.k = k
        self.batch_size = batch_size

        # Initialize variational parameters
        variational_distribution = MeanFieldVariationalDistribution(num_inducing_points = self.n_inducing_points)
        
        variational_strategy = NNVariationalStrategy(
            self, inducing_points, variational_distribution, k = k, training_batch_size = batch_size)
        
        super(VNNGPModel, self).__init__(variational_strategy)

        # Initialize mean function
        self.mean_module = gpytorch.means.ConstantMean()
        #self.mean_module = gpytorch.means.ZeroMean()

        # Initialize covariance kernel
        # Naming this module differently fails training!
        self.covar_module = AdditiveKernel(
            LinearKernel(active_dims = 0),
            ScaleKernel(PeriodicKernel(
                active_dims = (1),
                period_length_prior = NormalPrior(24, 1) # Applied to hour feature
            )),
            ScaleKernel(PeriodicKernel(
                active_dims = (2),
                period_length_prior = NormalPrior(7, 1) # Applied to day of week feature
            )),
            ScaleKernel(PeriodicKernel(
                active_dims = (3),
                period_length_prior = NormalPrior(12, 1) # Applied to month feature
            )),
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    # Two options when calling the model:
    # Option 1: x = None, which will sample a batch of inducing points as x. Faster, will break time series order.
    # Option 2: x = X_batched, which will retrieve the matching inducing points for x. Slower, will maintain time series order.
    def __call__(self, x, prior = False, **kwargs):
        return self.variational_strategy(x = x, prior = prior, **kwargs)


# Variational nearest neighbors GP wrapper class (minibatch training for both X_train and inducing_points, minibatch predictions)
class VNNGP:
    
    def __init__(self, model, likelihood, cuda = True):
        self.model = model
        self.likelihood = likelihood
        self.cuda = cuda

    # Training method
    def train(self, train_loader, num_data, max_epochs, learning_rate = 0.001, early_stop = 5, early_stop_tol = 1e-3):

        # Put model & likelihood on GPU if cuda is enabled
        if self.cuda:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

        # Put model in training mode
        self.model.train()
        self.likelihood.train()

        # Create hyperparameter optimizer with model parameters
        # IMPORTANT: If self.model is not created with the likelihood as an input parameter,
        # the likelihood parameters likely must be passed to the optimizer here!
        optimizer = torch.optim.Adam([
            {"params": self.model.parameters()},
            {"params": self.likelihood.parameters()},
        ], lr = learning_rate)

        # Create loss (returns unreduced loss tensor, batch_size elements)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data = num_data)

        # N. of batches for epoch loss calculation
        num_batches = len(train_loader)

        # Training loop
        for epoch in range(max_epochs):

            # Initialize loss tracking for epoch
            total_loss_epoch = 0

            # Iterate over batches
            minibatch_iter = tqdm(train_loader, desc="Epoch progress", leave=False)
            for X, y in minibatch_iter:

                # Put tensors on cuda if enabled
                if self.cuda:
                    X = X.cuda()
                    y = y.cuda()

                # Reset gradients
                optimizer.zero_grad()
    
                # Get outputs from model
                output = self.model(X)
    
                # Calculate loss perform backpropagation, adjust weights
                batch_loss = -mll(output, y)
                batch_loss.backward()
                optimizer.step()

                # Print batch loss
                batch_loss_scalar = batch_loss.item()
                minibatch_iter.set_postfix(loss = batch_loss_scalar)
    
                # Save batch loss
                total_loss_epoch += batch_loss_scalar

            # Calculate epoch loss
            epoch_loss = total_loss_epoch / num_batches
        
            # Initialize best loss & rounds with no improvement if first epoch
            if epoch == 0:
                self._best_epoch = epoch
                self._best_loss = epoch_loss
                self._epochs_no_improvement = 0
                self._best_state_dict = self.model.state_dict()
            
            # Record an epoch with no improvement
            if self._best_loss < epoch_loss - early_stop_tol:
                self._epochs_no_improvement += 1

            # Record an improvement in the loss
            if self._best_loss > epoch_loss + early_stop_tol:
                self.best_epoch = epoch
                self._best_loss = epoch_loss
                self._epochs_no_improvement = 0
                self._best_state_dict = self.model.state_dict()

            # Print epoch info
            print(f"Epoch complete: {epoch+1}/{max_epochs}, Loss: {epoch_loss}, Best loss: {self._best_loss}")

            # Early stop if necessary
            if self._epochs_no_improvement >= early_stop:
                print(f"Early stopping after epoch {epoch+1}")
                break

        # Load best checkpoint after training ends 
        self.model.load_state_dict(self._best_state_dict)

    # Method to update model training data (kernel hyperparameters unchanged, no additional training)
    # Returns an ExactGP model
    def update_train(self, X_update, y_update):
        
        # Put tensors on GPU if cuda is enabled
        if self.cuda:
            X_update = X_update.cuda()
            y_update = y_update.cuda()

        # Update model training data
        self.model = self.model.get_fantasy_model(X_update, y_update)

    # Predict method (batched)
    def predict(self, test_loader, cpu = True, fast_preds = False):

        # Activate eval mode
        self.model.eval()
        self.likelihood.eval()

        # Initialize tensors to store batch predictions
        preds, lower, upper = torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])
    
        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var(state = fast_preds):
            for X_test, y_test in test_loader:

                # Tensors to GPU, if cuda enabled
                if self.cuda:
                    X_test = X_test.cuda()
                    preds = preds.cuda()
                    lower = lower.cuda()
                    upper = upper.cuda()
    
                # Returns the approximate model posterior distribution over functions p(f*|x*, X, y)
                # Noise is not yet added to the functions
                f_posterior = self.model(X_test)
    
                # Returns the approximate predictive posterior distribution p(y*|x*, X, y)
                # Noise is added to the functions
                y_posterior = self.likelihood(f_posterior)
    
                # Get posterior predictive mean & prediction intervals
                # By default, 2 standard deviations around the mean
                y_mean = y_posterior.mean
                y_lower, y_upper = y_posterior.confidence_region()

                # Concatenate batch predictions to prediction tensors
                preds = torch.cat([preds, y_mean])
                lower = torch.cat([lower, y_lower])
                upper = torch.cat([upper, y_upper])

        # Return data to CPU
        if cpu:
            preds = preds.cpu()
            lower = lower.cpu()
            upper = upper.cpu()
    
        return preds[1:], lower[1:], upper[1:]
