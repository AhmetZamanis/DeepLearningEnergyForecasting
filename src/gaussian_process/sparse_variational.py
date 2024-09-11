import gpytorch
from gpytorch.kernels import ScaleKernel, LinearKernel, PeriodicKernel, AdditiveKernel
from gpytorch.priors import NormalPrior
from gpytorch.variational import VariationalStrategy, NaturalVariationalDistribution


# Sparse Variational GP model class
class SVGPModel(gpytorch.models.ApproximateGP):

    def __init__(self, inducing_points, learn_inducting_locations = False):

        # Initialize variational parameters
        variational_distribution = NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations = learn_inducting_locations)
        super(GPModel, self).__init__(variational_strategy)

        # Initialize mean function
        self.mean_module = gpytorch.means.ConstantMean()
        #self.mean_module = gpytorch.means.ZeroMean()

        # Initialize covariance kernel
        self.covariance_module = AdditiveKernel(
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
        covar = self.covariance_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# Sparse variational GP wrapper class, minibatch training except for inducing points
class SVMinibatchGP:
    
    def __init__(self, model, likelihood, cuda = True):
        self.model = model
        self.likelihood = likelihood
        self.cuda = cuda

    # Training method
    def train(self, train_loader, num_data, max_epochs, learning_rate = 0.1, early_stop = 10, early_stop_tol = 1e-4):

        # Put model & likelihood on GPU if cuda is enabled
        if self.cuda:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

        # Find optimal kernel hyperparameters
        self.model.train()
        self.likelihood.train()

        # Create variational optimizer for natural gradient descent
        var_optimizer = gpytorch.optim.NGD(
            self.model.variational_parameters(),
            num_data = num_data,
            lr = learning_rate
        )

        # Create hyperparameter optimizer with model parameters
        optimizer = torch.optim.Adam([
            {"params": self.model.parameters()},
            {"params": self.likelihood.parameters()}
        ], lr = learning_rate)

        # Create loss
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data = num_data)

        # N. of batches for epoch loss calculation
        num_batches = len(train_loader)

        # Training loop
        for epoch in range(max_epochs):

            # Initialize loss tracking for epoch
            total_loss_epoch = 0

            # Iterate over batches
            minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
            for X, y in minibatch_iter:

                # Put tensors on cuda if enabled
                if self.cuda:
                    X = X.cuda()
                    y = y.cuda()

                # Reset gradients
                optimizer.zero_grad()
                var_optimizer.zero_grad()
    
                # Get outputs from model
                output = self.model(X)
    
                # Calculate loss perform backpropagation, adjust weights
                batch_loss = -mll(output, y)
                batch_loss.backward()
                optimizer.step()
                var_optimizer.step()

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
            
            # Record an epoch with no improvement
            if self._best_loss < epoch_loss - early_stop_tol:
                self._epochs_no_improvement += 1

            # Record an improvement in the loss
            if self._best_loss > epoch_loss:
                self.best_epoch = epoch
                self._best_loss = epoch_loss
                self._epochs_no_improvement = 0

            # Print epoch info
            print(f"Epoch complete: {epoch+1}/{max_epochs}, Loss: {epoch_loss}, Best loss: {self._best_loss}")

            # Early stop if necessary
            if self._epochs_no_improvement >= early_stop:
                print(f"Early stopping after epoch {epoch+1}")
                break

    # Method to update model training data (kernel hyperparameters unchanged, no additional training)
    # Returns an ExactGP model
    def update_train(self, X_update, y_update):
        
        # Put tensors on GPU if cuda is enabled
        if self.cuda:
            X_update = X_update.cuda()
            y_update = y_update.cuda()

        # Update model training data
        self.model = self.model.get_fantasy_model(X_update, y_update)

    # Predict method (unbatched)
    def predict(self, X_test, cpu = True, fast_preds = False):

        # Test data to GPU, if cuda enabled
        if self.cuda:
            X_test = X_test.cuda()

        # Activate eval mode
        self.model.eval()
        self.likelihood.eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var(state = fast_preds):

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

        # Return data to CPU if desired
        if cpu:
            y_mean = y_mean.cpu()
            y_lower = y_lower.cpu()
            y_upper = y_upper.cpu()

        return y_mean, y_lower, y_upper
