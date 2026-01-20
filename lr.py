# Linear Regression WITHOUT any ML libraries
# Using pure Python (no numpy, no sklearn)

# Sample dataset (Hours studied vs Marks obtained)
x = [-2, -1, 0, 1, 2]
y = [65, 95, 80, 115, 105]

# Step 1: Calculate mean of X and Y
def mean(values):
    return sum(values) / len(values)

x_mean = mean(x)
y_mean = mean(y)

# Step 2: Calculate slope (m)
numerator = 0
denominator = 0

for i in range(len(x)):
    numerator += (x[i] - x_mean) * (y[i] - y_mean)
    denominator += (x[i] - x_mean) ** 2

m = numerator / denominator

# Step 3: Calculate intercept (c)
c = y_mean - (m * x_mean)

print("Slope (m):", m)
print("Intercept (c):", c)

# Step 4: Predict function
def predict(x_value):
    return m * x_value + c

# Test prediction
test_x = 6
print("Prediction for x =", test_x, "is:", predict(test_x))