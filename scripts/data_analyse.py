import pickle

with open("data/train.pkl", "rb") as file:
    data = pickle.load(file)

print(type(data))
print(f"Liczba elementów: {len(data)}")
counts = {}
for i in range(len(data)):
    sequence, label = data[i]
    print(f"Element {i}:")
    print(f"  Sekwencja (długość {len(sequence)}): {sequence[:10]}...")
    print(f"  Klasa: {label}")
    counts[label] = counts.get(label, 0) + 1

for label, count in counts.items():
    print(f"{label}: {count} examples")
