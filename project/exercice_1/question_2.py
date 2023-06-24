import numpy as np

def ridge_regression_estimator(
    X: np.ndarray, y: np.ndarray, lambda_: float
) -> np.ndarray:
    """
    Compute the Ridge regression estimator

    We use numpy broadcasting to accelerate computations
    and obtain several OLS estimators.

    Parameters:
        X: (n, d) matrix
        y: (n, n_tests) matrix
        lambda: regularization parameter

    Returns:
        theta_hat: (d, n_tests) matrix
    """
    n, d = X.shape
    covariance_matrix = X.T @ X
    Sigma_matrix = covariance_matrix / n
    theta_hat = 1 / n * np.linalg.inv(Sigma_matrix + lambda_ * np.identity(d)) @ X.T @ y
    return theta_hat


def main() -> None:
    """
    Simulate problem 1
    """
    lambda_ = 10 ** -1
    n_tests = int(1e4)
    # instantiate a Pseudo-random number generator (PRNG)
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Set the parameters for the distribution
    n_samples = 1000
    mean = [0, 0]  # Mean of X
    cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix of X
    X = np.random.multivariate_normal(mean, cov, size=n_samples)

    # Step 2: Define the conditional distribution of Y given X
    def joint_distribution(X):
        # Generate Y based on a conditional distribution given X
        n = len(X)
        mean_y = np.zeros((n, 2))  # Mean of Y based on X
        cov_y = np.zeros((n, 2, 2))  # Covariance matrix of Y based on X

        # Define the mean and covariance for each pair (x1, x2) in X
        for i, (x1, x2) in enumerate(X):
            mean_y[i] = [x1 + 2 * x2, x1 - x2]  # Example transformation
            cov_y[i] = [[1.5 * x1 + 0.5 * x2, 0.5 * x1 + 2 * x2], [0.5 * x1 + 2 * x2, x1 + 3 * x2]]  # Example transformation

        # Generate samples of Y given the mean and covariance
        Y = np.zeros((n, 2))
        for i in range(n):
            Y[i] = np.random.multivariate_normal(mean_y[i], cov_y[i])

        return Y

    # Step 3: Generate Y based on X using the joint distribution
    y = joint_distribution(X)


    # generate predictions with the Bayes estimator
    # When doing classification with the "0-1" loss,
    # the Bayes estimator predict the most probable output
    # for each input. (we will show this during the class)
    # In that case, it turns out that it corresponds exactly to
    # predicting X, but note that this will not always be the case.
    y_pred_bayes = X

    # compute the empirical risk for the Bayes estimator
    

    

    # compute the Ridge regression estimator
    theta_hat = ridge_regression_estimator(X, y, lambda_)

    y_test = joint_distribution(X)

    empirical_risk_bayes = len(np.where(y_test - y_pred_bayes)[0]) / n_samples

    y_pred_ridge = X @ theta_hat

    empirical_risk_ridge = np.linalg.norm(y_pred_ridge - y_test) ** 2 / (n_samples * n_tests)

    print("\nX")
    print(X)
    print("\ny")
    print(y)
    print("\ny pred squared loss")
    print(y_pred_bayes.astype(int))
    print("\nempirical risk for bayes predictor squared loss")
    print(empirical_risk_bayes)
    print("\nempirical risk for ridge predictor squared loss")
    print(empirical_risk_ridge)



if __name__ == "__main__":
    main()