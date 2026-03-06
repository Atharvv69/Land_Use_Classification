from Patch import labels
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

esa_labels = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    80: "Water",
    90: "Wetland"
}

x = [img for img, label in labels]
y = [label for img, label in labels]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.4,
    stratify=y,
    random_state=42
)

train_counts = Counter(y_train)
test_counts = Counter(y_test)

# classes = sorted(set(y_train) | set(y_test))
# print("\nClass  Train%   Test%")
# for c in classes:
#     train_pct = 100 * train_counts.get(c, 0) / len(x_train)
#     test_pct  = 100 * test_counts.get(c, 0) / len(x_test)
#     print(f"{c:>5}  {train_pct:6.2f}%  {test_pct:6.2f}%")

classes = sorted(set(y))
class_names = [esa_labels[c] for c in classes]

train_values = [train_counts.get(c, 0) for c in classes]
test_values = [test_counts.get(c, 0) for c in classes]

x = range(len(classes))

plt.figure(figsize=(10,5))
plt.bar(x, train_values, width=0.4, label = "Train")
plt.bar([i + 0.4 for i in x], test_values, width=0.4, label = "Test")
plt.xticks([i + 0.2 for i in x], class_names, rotation = 45)
plt.xlabel("Land Cover Class")
plt.ylabel("Number of Samples")
plt.title("Class Distribution in Train and Test Sets")
plt.legend()



if __name__ == "__main__":
    print("Train_size: ", len(x_train))
    print("Test_size: ", len(x_test))
    print("Train Distribution: ", train_counts)
    print("Test Distribution: ", test_counts)
    plt.show()

