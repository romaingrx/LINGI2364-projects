#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 11:54:17 2021

@author: francois
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import statistics
from operator import add

train02 = {}
test02 = {}
train03 = {}
test03 = {}
train04 = {}
test04 = {}

with open("results/benchamrk_1000_4folds.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        file = row["file"][:2]
        k = row["k"]
        fold = row["fold"]
        train = float(row["train_acc"]) * 100
        test = float(row["test_acc"]) * 100

        if file == "02":
            if train02.get(k) == None:
                train02[k] = [train]
                test02[k] = [test]
            else:
                train02[k].append(train)
                test02[k].append(test)

        elif file == "03":
            if train03.get(k) == None:
                train03[k] = [train]
                test03[k] = [test]
            else:
                train03[k].append(train)
                test03[k].append(test)

        elif file == "04":
            if train04.get(k) == None:
                train04[k] = [train]
                test04[k] = [test]
            else:
                train04[k].append(train)
                test04[k].append(test)

        else:
            print("PROBLEM")

# print(train02)
# print(test02)
# print(train03)
# print(test03)
# print(train04)
# print(test04)

a2_m = []
a2_s = []
a3_m = []
a3_s = []
a4_m = []
a4_s = []
b2_m = []
b2_s = []
b3_m = []
b3_s = []
b4_m = []
b4_s = []

abscisse = []
for i in train02:
    abscisse.append(int(i))
    m = statistics.mean(train02[i])
    a2_m.append(m)
    s = statistics.stdev(train02[i])
    a2_s.append(s)

for i in test02:
    m = statistics.mean(test02[i])
    b2_m.append(m)
    s = statistics.stdev(test02[i])
    b2_s.append(s)

for i in train03:
    m = statistics.mean(train03[i])
    a3_m.append(m)
    s = statistics.stdev(train03[i])
    a3_s.append(s)

for i in test03:
    m = statistics.mean(test03[i])
    b3_m.append(m)
    s = statistics.stdev(test03[i])
    b3_s.append(s)

for i in train04:
    m = statistics.mean(train04[i])
    a4_m.append(m)
    s = statistics.stdev(train04[i])
    a4_s.append(s)

for i in test04:
    m = statistics.mean(test04[i])
    b4_m.append(m)
    s = statistics.stdev(test04[i])
    b4_s.append(s)
#    print(x)
#    print(s)

abscisse.sort()

### TRAIN ###
s02p = list(map(add, a2_m, a2_s))
s02m = list(map(add, a2_m, [-1 * i for i in a2_s]))
s03p = list(map(add, a3_m, a3_s))
s03m = list(map(add, a3_m, [-1 * i for i in a3_s]))
s04p = list(map(add, a4_m, a4_s))
s04m = list(map(add, a4_m, [-1 * i for i in a4_s]))

### TEST ###
t02p = list(map(add, b2_m, b2_s))
t02m = list(map(add, b2_m, [-1 * i for i in b2_s]))
t03p = list(map(add, b3_m, b3_s))
t03m = list(map(add, b3_m, [-1 * i for i in b3_s]))
t04p = list(map(add, b4_m, b4_s))
t04m = list(map(add, b4_m, [-1 * i for i in b4_s]))

plt.figure(1)
plt.plot(abscisse, a2_m, color="crimson", label="Training set")
# plt.scatter(abscisse, a2_m, s=15)
# plt.plot(abscisse, s02p, label='Std+ TRAIN')
# plt.plot(abscisse, s02m, label='Std- TRAIN')
plt.fill_between(abscisse, s02p, s02m, color="pink")
print("Decision Tree Train: ", a2_m)
print("Decision Tree Test: ", b2_m)

plt.plot(abscisse, b2_m, color="green", label="Test set")
# plt.scatter(abscisse, b2_m, s=15)
# plt.plot(abscisse, t02p, label='Std+ TEST')
# plt.plot(abscisse, t02m, label='Std- TEST')
plt.fill_between(abscisse, t02p, t02m, color="palegreen")
# plt.yscale("log")
plt.title("Influence of parameter k on prediction accuracy of the Decision Tree model")
plt.xlabel("k")
plt.ylabel("Accuracy [%]")
plt.legend()
# plt.savefig('each_matrix.jpg')
plt.show()

plt.figure(2)
plt.plot(abscisse, b3_m, color="green", label="Test set")
# plt.scatter(abscisse, b3_m, s=15)
# plt.plot(abscisse, t03p, label='Std+ TEST')
# plt.plot(abscisse, t03m, label='Std- TEST')
plt.fill_between(abscisse, t03p, t03m, color="palegreen")

plt.plot(abscisse, a3_m, color="crimson", label="Training set")
# plt.scatter(abscisse, a3_m, s=15)
# plt.plot(abscisse, s03p, label='Std+ TRAIN')
# plt.plot(abscisse, s03m, label='Std- TRAIN')
plt.fill_between(abscisse, s03p, s03m, color="pink")
print("Sequential Train: ", a3_m)
print("Sequential Test: ", b3_m)

# plt.yscale("log")
plt.title("Influence of parameter k on prediction accuracy of the Sequential model")
plt.xlabel("k")
plt.ylabel("Accuracy [%]")
plt.legend()
# plt.savefig('each_matrix.jpg')
plt.show()

plt.figure(3)
plt.plot(abscisse, a4_m, color="crimson", label="Training set")
# plt.scatter(abscisse, a4_m, s=15)
# plt.plot(abscisse, s04p, label='Std+ TRAIN')
# plt.plot(abscisse, s04m, label='Std- TRAIN')
plt.fill_between(abscisse, s04p, s04m, color="pink")
print("Random Train: ", a4_m)
print("Random Test: ", b4_m)
plt.plot(abscisse, b4_m, color="green", label="Test set")
# plt.scatter(abscisse, b4_m, s=15)
# plt.plot(abscisse, t04p, label='Std+ TEST')
# plt.plot(abscisse, t04m, label='Std- TEST')
plt.fill_between(abscisse, t04p, t04m, color="palegreen")
# plt.yscale("log")
plt.title("Influence of parameter k on prediction accuracy of the Random Forest model")
plt.xlabel("k")
plt.ylabel("Accuracy [%]")
plt.legend()
# plt.savefig('each_matrix.jpg')
plt.show()
