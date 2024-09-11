# The ExactGP classes are the most "complete" ones, the others could be updated similar to them
import gpytorch
from gpytorch.kernels import ScaleKernel, LinearKernel, PeriodicKernel, AdditiveKernel
from gpytorch.priors import NormalPrior


# ExactGP model class
class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, X_train, y_train, likelihood):
        super().__init__(X_train, y_train, likelihood)

        # Create mean module
        self.mean_module = gpytorch.means.ConstantMean()

        # Create covariance module
        self.covar_module = AdditiveKernel(
            LinearKernel(active_dims = 0), # Linear trend
            ScaleKernel(PeriodicKernel(
                active_dims = (1),
                period_length_prior = NormalPrior(24, 1) # Hourly seasonality
            )),
            ScaleKernel(PeriodicKernel(
                active_dims = (2),
                period_length_prior = NormalPrior(7, 1) # Day of week seasonality
            )),
            ScaleKernel(PeriodicKernel(
                active_dims = (3),
                period_length_prior = NormalPrior(12, 1) # Year of month seasonality
            )),
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ExactGP wrapper class (unbatched gradient descent)
class ExactGP:
    
    def __init__(self, model, likelihood, cuda = True):
        self.cuda = cuda

        # Put model & likelihood on GPU if cuda is enabled
        if cuda:
            self.model = model.cuda()
            self.likelihood = likelihood.cuda()
        else:
            self.model = model
            self.likelihood = likelihood
            
    # Training method
    def train(self, X_train, y_train, max_epochs, learning_rate = 0.01, early_stop = 5, early_stop_tol = 1e-3):

        # Put data on GPU if cuda is enabled
        if self.cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()

        # Put models into training mode
        self.model.train()
        self.likelihood.train()

        # Create Adam optimizer with model parameters
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)

        # Create marginal log likelihood loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Training loop
        for epoch in range(max_epochs):

            # Reset gradients
            optimizer.zero_grad()

            # Get outputs from model
            output = self.model(X_train)

            # Calculate loss and perform backpropagation
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

            # Get loss & noise values to be printed
            loss_scalar = loss.item()
            noise = self.model.likelihood.noise.item()
            
            # Initialize best loss & rounds with no improvement if first epoch
            if epoch == 0:
                self._best_epoch = epoch
                self._best_loss = loss_scalar
                self._epochs_no_improvement = 0
                self._best_state_dict = self.model.state_dict()

            # Record an epoch with no improvement
            if self._best_loss < loss_scalar - early_stop_tol:
                self._epochs_no_improvement += 1

            # Record an improvement in the loss
            if self._best_loss > loss_scalar + early_stop_tol:
                self.best_epoch = epoch
                self._best_loss = loss_scalar
                self._epochs_no_improvement = 0
                self._best_state_dict = self.model.state_dict()

            # Print epoch summary
            print(f"Epoch: {epoch+1}/{max_epochs}, Loss: {loss_scalar}, Noise: {noise}, Best loss: {self._best_loss}")

            # Early stop if necessary
            if self._epochs_no_improvement >= early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best checkpoint after training 
        self.model.load_state_dict(self._best_state_dict)

        # Delete unneeded tensors
        del X_train, y_train, optimizer, mll, output, loss, loss_scalar, noise

        # Clear GPU memory 
        torch.cuda.empty_cache()
            
    # Predict method
    def predict(self, X_test, cpu = True, fast_preds = False):

        # Test data to GPU, if cuda enabled
        if self.cuda:
            X_test = X_test.cuda()

        # Activate eval mode
        self.model.eval()
        self.likelihood.eval()

        # Make predictions without gradient calculation
        with torch.no_grad(), gpytorch.settings.fast_pred_var(state = fast_preds):

            # Returns the model posterior distribution over functions: p(f*|X_test, X_train, y_train)
            # Noise is not yet added to the functions
            f_posterior = self.model(X_test)

            # Returns the predictive posterior distribution: p(y*|X_test, X_train, y_train)
            # Noise is added to the functions
            y_posterior = self.likelihood(f_posterior)

            # Get posterior predictive mean & prediction intervals
            # By default, 2 standard deviations around the mean
            y_mean = y_posterior.mean
            y_lower, y_upper = y_posterior.confidence_region()

        # Return predictions to CPU if desired
        # Could be advantageous to stack the predictions into one array
        if cpu:
            y_mean = y_mean.cpu()
            y_lower = y_lower.cpu()
            y_upper = y_upper.cpu()

        # Delete unneeded tensors
        del X_test, f_posterior, y_posterior

        # Clear GPU memory
        torch.cuda.empty_cache()

        return y_mean, y_lower, y_upper

    # Method to update model training data (kernel hyperparameters unchanged, no additional training performed)
    # When done repeatedly, GPU memory fills up, couldn't determine exact cause or find a fix.
    def update_train(self, X_update, y_update):
        
        # Put tensors on GPU if cuda is enabled
        if self.cuda:
            X_update = X_update.cuda()
            y_update = y_update.cuda()

        # Update model training data
        self.model = self.model.get_fantasy_model(X_update, y_update)

        # Delete unneeded tensors
        del X_update, y_update

        # Clear GPU memory
        torch.cuda.empty_cache()

    # Method to save model state dict
    def save(self, save_dir):
        torch.save(self.model.state_dict(), save_dir)

    # Method to load model parameters from saved state dict
    def load(self, state_dict):
        self.model.load_state_dict(state_dict)
