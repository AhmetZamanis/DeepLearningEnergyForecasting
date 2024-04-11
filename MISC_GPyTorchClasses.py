
# Code for applying various Gaussian Process regression methods with GPyTorch


# ExactGP model class
class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, X_train, y_train, likelihood):
        super().__init__(X_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        #self.mean_module = gpytorch.means.ZeroMean()
        self.covariance_module = AdditiveKernel(
            LinearKernel(active_dims = 0),
            #ScaleKernel(LinearKernel(active_dims = 0)),
            ScaleKernel(PeriodicKernel(
                active_dims = (1),
                period_length_prior = NormalPrior(24, 1)
                #period_length_constraint = Interval(23, 25) 
            )),
            ScaleKernel(PeriodicKernel(
                active_dims = (2),
                period_length_prior = NormalPrior(7, 1),
                #period_length_constraint = Interval(6, 8) 
            )),
            ScaleKernel(PeriodicKernel(
                active_dims = (3),
                period_length_prior = NormalPrior(12, 1)
                #period_length_constraint = Interval(11, 13) 
            )),
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covariance_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ExactGP wrapper class (unbatched gradient descent)
class ExactGP:
    
    def __init__(self, model, likelihood, cuda = True):
        self.model = model
        self.likelihood = likelihood
        self.cuda = cuda

    # Training method
    def train(self, X_train, y_train, max_epochs, learning_rate = 0.1, early_stop = 10, early_stop_tol = 1e-4):

        # Put tensors on GPU if cuda is enabled
        if self.cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

        # Find optimal kernel hyperparameters
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

            # Print epoch info & update model parameters
            loss_scalar = loss.item()
            noise = self.model.likelihood.noise.item()
            print(f"Epoch: {epoch+1}/{max_epochs}, Loss: {loss_scalar}, Noise: {noise}")
            optimizer.step()

            # Initialize best loss & rounds with no improvement if first epoch
            if epoch == 0:
                self._best_epoch = epoch
                self._best_loss = loss_scalar
                self._epochs_no_improvement = 0

            # Record an epoch with no improvement
            if self._best_loss < loss_scalar - early_stop_tol:
                self._epochs_no_improvement += 1

            # Early stop if necessary
            if self._epochs_no_improvement >= early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

            # Record an improvement in the loss
            if self._best_loss > loss_scalar:
                self.best_epoch = epoch
                self._best_loss = loss_scalar
                self._epochs_no_improvement = 0
                
    # Method to update model training data (kernel hyperparameters unchanged, no additional training)
    def update_train(self, X_update, y_update):
        
        # Put tensors on GPU if cuda is enabled
        if self.cuda:
            X_update = X_update.cuda()
            y_update = y_update.cuda()

        # Update model training data
        self.model = self.model.get_fantasy_model(X_update, y_update)

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

            # Returns the model posterior distribution over functions p(f*|x*, X, y)
            # Noise is not yet added to the functions
            f_posterior = self.model(X_test)

            # Returns the predictive posterior distribution p(y*|x*, X, y)
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

