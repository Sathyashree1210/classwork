import random

# Sample dataset: (size in sq. ft., price in $1000s)
data = [
    (600, 150),
    (800, 200),
    (1000, 250),
    (1200, 300),
    (1400, 350)
]

# Initialize weights and bias randomly
w = random.uniform(-1, 1)
b = random.uniform(-1, 1)

# Learning rate
lr = 0.0001

# Mean Squared Error Loss
def compute_loss(w, b, data):
    total_loss = 0
    for x, y in data:
        y_pred = w * x + b
        total_loss += (y - y_pred) ** 2
    return total_loss / len(data)

# Recursive training function
def train(w, b, data, lr, epochs, current_epoch=0):
    if current_epoch >= epochs:
        return w, b  # Stop condition

    dw, db = 0, 0
    for x, y in data:
        y_pred = w * x + b
        dw += -2 * x * (y - y_pred)
        db += -2 * (y - y_pred)

    # Average gradients
    dw /= len(data)
    db /= len(data)

    # Update weights and bias
    w = w - lr * dw
    b = b - lr * db

    if current_epoch % 100 == 0:
        loss = compute_loss(w, b, data)
        print(f"Epoch {current_epoch} - Loss: {loss:.4f}, Weight: {w:.4f}, Bias: {b:.4f}")

    # Recursive call
    return train(w, b, data, lr, epochs, current_epoch + 1)

# Train the model
final_w, final_b = train(w, b, data, lr, epochs=1000)

print(f"\nFinal model: price = {final_w:.4f} * size + {final_b:.4f}")
