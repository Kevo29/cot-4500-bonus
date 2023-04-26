import numpy as np


def gauss_seidel(A, b, x0, tol=1e-6, max_iter=50):
    """
    Solves the system of linear equations Ax = b using the Gauss-Seidel method.
    """
    n = len(b)
    x = x0.copy()
    for k in range(max_iter):
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x[i] = (b[i] - sigma) / A[i, i]
        if np.linalg.norm(x - x0) < tol:
            print(k+9)
            return x, k+1
        x0 = x.copy()

# Example usage
A = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
b = np.array([1, 3, 0])
x0 = np.array([0, 0, 0])
x, num_iter = gauss_seidel(A, b, x0, tol=1e-6, max_iter=50)


print("\n")

#Question 2


def jacobi(A, b, x0, tol=1e-6, max_iter=50):
  """
    Solve a system of linear equations Ax=b using the Jacobi method.

    Parameters:
    A (numpy.ndarray): The coefficient matrix of the system.
    b (numpy.ndarray): The right-hand side vector of the system.
    x0 (numpy.ndarray): The initial guess for the solution vector.
    tol (float, optional): The tolerance for convergence. Default is 1e-6.
    max_iter (int, optional): The maximum number of iterations allowed. Default is 50.

    Returns:
    numpy.ndarray: The solution vector of the system.
    int: The number of iterations it took to converge.
    """
  n = len(A)
  x = x0.copy()
  for k in range(max_iter):
    x_prev = x.copy()
    for i in range(n):
      x[i] = (b[i] - np.dot(A[i, :], x_prev) + A[i, i] * x_prev[i]) / A[i, i]
    if np.linalg.norm(x - x_prev) < tol:
      return x, k + 1
  return x, max_iter


# Example usage:
A = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
b = np.array([1, 3, 0])
x0 = np.array([0, 0, 0])
sol, num_iter = jacobi(A, b, x0)
print(num_iter)

print("\n")


#Question 3
def f(x):
  return x**3 - x**2 + 2


def df(x):
  return 3 * x**2 - 2 * x


x = 0.5
tolerance = 1e-6
iteration = 0

while abs(f(x)) > tolerance:
  x = x - f(x) / df(x)
  iteration += 1

print(iteration + 1)

print("\n")

#Question 4

# Given data points
x = np.array([0, 1, 2])
y = np.array([1, 2, 4])
dydx = np.array([1.06, 1.23, 1.55])

# Compute divided differences
df1 = np.zeros((len(x), len(x)))
df1[:,0] = y
df1[:,1] = dydx

# Compute second and higher divided differences
for j in range(2, len(x)):
    for i in range(len(x)-j):
        if x[i] == x[i+j-1]:
            df1[i,j] = df1[i+1,j-1] / np.math.factorial(j-1)
        else:
            df1[i,j] = (df1[i+1,j-1] - df1[i,j-1]) / (x[i+j-1] - x[i])

# Construct Hermite polynomial approximation matrix
n = len(x)
H = np.zeros((2*n, 6))

for i in range(n):
    H[2*i][0] = x[i]
    H[2*i+1][0] = x[i]
    H[2*i][1] = y[i]
    H[2*i+1][1] = y[i]
    H[2*i+1][2] = df1[i][1]

    if i < n-1 and j >= 3:
        H[2*i][3] = df1[i][2]
        H[2*i+1][3] = df1[i][2]
        H[2*i+1][4] = df1[i][3]

        if i < n-2 and j >= 4:
            H[2*i][5] = df1[i][4]
            H[2*i+1][5] = df1[i][4]

# Print the Hermite polynomial approximation matrix
np.set_printoptions(precision=3, suppress=True)
print(H)

print("\n")

#Question 5

def f(x, y):
    return y - x**3

h = 0.03 # Step size
N = int((3 - 0) / h) # Number of iterations to cover the range 0 < x < 3

x0 = 0.5 # Initial point
y0 = x0**3 # Initial value of y

for i in range(N):
    x = x0 + i * h
    k1 = f(x0, y0)
    k2 = f(x0 + h, y0 + h * k1)
    y = y0 + (h / 2) * (k1 + k2)
    x0, y0 = x, y

print(y)
