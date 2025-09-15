import random

# Predict sales based on inputs, weights, and bias
def predict(inputs, weights, bias):
    return sum(x * w for x, w in zip(inputs, weights)) + bias

# Train weights using a simple rule (no if-else)
def train(data, weights, bias):
    def update(item):
        actual, *features = item
        predicted = predict(features, weights, bias)
        error = actual - predicted
        return (
            [w + (error * x) // 10000 for w, x in zip(weights, features)],
            bias + error // 10000
        )
    for item in data:
        weights, bias = update(item)
    return weights, bias

# Recursive sum of all data columns (no if-else)
def get_totals(data, index=0):
    return (
        (0, 0, 0, 0)
        if index == len(data)
        else tuple(map(sum, zip(data[index], get_totals(data, index + 1))))
    )

# Dataset: (sales, shop_size, employees, salary)
shop_data = [
    (1500, 100, 5, 2000),
    (2300, 150, 7, 2800),
    (900, 80, 3, 1200),
    (3500, 200, 10, 4000),
    (1300, 90, 4, 1800),
    (1900, 120, 6, 2400),
]

# Initial random weights and bias
weights = [random.randint(1, 3) for _ in range(3)]
bias = random.randint(1, 3)

print("Initial weights:", weights, "Bias:", bias)

# Train once
weights, bias = train(shop_data, weights, bias)

print("Trained weights:", weights, "Bias:", bias)

# Show total values
total_sales, total_size, total_employees, total_salary = get_totals(shop_data)
print("\nTotals:")
print("Sales:", total_sales)
print("Shop Size:", total_size)
print("Employees:", total_employees)
print("Salary:", total_salary)
