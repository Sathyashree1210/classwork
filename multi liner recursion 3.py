import random

# Predict total sales using weighted inputs and bias
def predict(shop_size, employees, salary, weights, bias):
    return shop_size * weights[0] + employees * weights[1] + salary * weights[2] + bias

# Recursive function to train weights and bias (1 round only, simple)
def train(data, weights, bias, index=0):
    if index == len(data):
        return weights, bias
    elif index < len(data):
        actual_sales, shop_size, employees, salary = data[index]
        predicted = predict(shop_size, employees, salary, weights, bias)
        error = actual_sales - predicted
        
# Basic weight/bias update rule
        weights[0] += (error * shop_size) // 10000
        weights[1] += (error * employees) // 10000
        weights[2] += (error * salary) // 10000
        bias += error // 10000
        
        return train(data, weights, bias, index + 1)
    else:
        return weights, bias

# Dataset: (sales, shop_size, employees, salary)
shop_data = [
    (1500, 100, 5, 2000),
    (2300, 150, 7, 2800),
    (900, 80, 3, 1200),
    (3500, 200, 10, 4000),
    (1300, 90, 4, 1800),
    (1900, 120, 6, 2400),
]

# Initialize small weights and bias
weights = [random.randint(1, 3) for _ in range(3)]
bias = random.randint(1, 3)

print("Initial Weights:", weights, "Bias:", bias)

# Train the model once
weights, bias = train(shop_data, weights, bias)

print("Trained Weights:", weights, "Bias:", bias)

# Sum attributes using recursion (same as before)
def total_attributes(data, index=0):
    if index == len(data):
        return 0, 0, 0, 0
    elif index < len(data):
        sales, shop_size, employees, salary = data[index]
        s_sales, s_size, s_emp, s_salary = total_attributes(data, index + 1)
        return (
            sales + s_sales,
            shop_size + s_size,
            employees + s_emp,
            salary + s_salary
        )
    else:
        return 0, 0, 0, 0

# Get total values
total_sales, total_size, total_employees, total_salary = total_attributes(shop_data)

print("\nTotals:")
print("Total sales:", total_sales)
print("Total shop size:", total_size)
print("Total employees:", total_employees)
print("Total salary:", total_salary)
