import numpy as np


def main() -> None:
    """
    Simulate problem 1
    """

    # instantiate a Pseudo-random number generator (PRNG)
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Define the mean and covariance matrix for the joint distribution
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]

    # Generate a random sample of size n from the joint distribution
    n = 1000
    X, y = np.random.multivariate_normal(mean, cov, n).T


    # generate predictions with the Bayes estimator
    # When doing classification with the "0-1" loss,
    # the Bayes estimator predict the most probable output
    # for each input. (we will show this during the class)
    # In that case, it turns out that it corresponds exactly to
    # predicting X, but note that this will not always be the case.
    y_pred_bayes = X

    # compute the empirical risk for the Bayes estimator
    empirical_risk_bayes = len(np.where(y - y_pred_bayes)[0]) / n

    print("\nX")
    print(X)
    print("\ny")
    print(y)
    print("\ny pred squared loss")
    print(y_pred_bayes.astype(int))
    print("\nempirical risk for bayes predictor squared loss")
    print(empirical_risk_bayes)


if __name__ == "__main__":
    main()