#!/usr/bin/env python3
"""
Create small test datasets for workflow agent training.
"""

import pandas as pd
import os

# Create data directory
os.makedirs("data", exist_ok=True)

# Training data - 5 simple programming tasks
train_tasks = [
    "Write a Python function that calculates the factorial of a number using recursion.",
    "Create a function that finds the maximum element in a list without using built-in max() function.",
    "Implement a binary search algorithm to find a target value in a sorted array.",
    "Write a function that checks if a string is a palindrome.",
    "Create a function that generates the first n Fibonacci numbers.",
]

# Validation data - 2 simple tasks
val_tasks = [
    "Write a function that sorts a list of integers using bubble sort.",
    "Create a function that counts the number of vowels in a string.",
]

# Create DataFrames
train_df = pd.DataFrame({"task": train_tasks})
val_df = pd.DataFrame({"task": val_tasks})

# Save as parquet files
train_df.to_parquet("data/train.parquet", index=False)
val_df.to_parquet("data/val.parquet", index=False)

print("Created test datasets:")
print(f"Training: {len(train_tasks)} tasks")
print(f"Validation: {len(val_tasks)} tasks")
print("\nTraining tasks:")
for i, task in enumerate(train_tasks, 1):
    print(f"  {i}. {task}")
print("\nValidation tasks:")
for i, task in enumerate(val_tasks, 1):
    print(f"  {i}. {task}")
