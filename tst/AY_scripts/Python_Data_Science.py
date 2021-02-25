import pandas as pd
import numpy as np
from pathlib import Path

# path = Path("Z:\\comma_group1\\User\\fcp18asy\\Python_Classes\\Uni_Python_Classes\\Data_Science")
# name = "airline-safety.csv"
# df = pd.read_csv(path/name)
# dfnan = df.mask(df["incidents_85_99"] < 70)
# df_filtered = dfnan.dropna()
# print(df_filtered)
# df.plot.scatter('avail_seat_km_per_week', 'incidents_85_99')
# # Check the correlation: Output: 0.36
# df['incidents_85_99'].corr(df['incidents_00_14'])
#
#
# def scope_test():
#     def do_local():
#         spam = "local spam" # Local namespace
#
#     def do_nonlocal():
#         nonlocal spam # To One level higher namespace
#         spam = "nonlocal spam"
#
#     def do_global():
#         global spam # To the GLOBAL namespace
#         spam = "global spam"
#
#     spam = "test spam"
#     do_local()
#     print("After local assignment:", spam)
#     do_nonlocal()
#     print("After nonlocal assignment:", spam)
#     do_global()
#     print("After global assignment:", spam)
#
#
# scope_test()
# print("In global scope:", spam)


# class Person:
#     name = "Mike"
#     age = 22
#
#
# a = Person()
# b = Person()
#
# print(a.name)
# print(a.age)
# print(b.name)
# print(b.age)

# class MyClass:
#     class_var = 1
#
#     def __init__(self, instance_var):
#         self.instance_var = instance_var
#
#
# a = MyClass(2)
# b = MyClass(1)
# print(a.class_var, a.instance_var)
# print(b.class_var, b.instance_var)

# class Circle:
#     def __init__(self, r):
#         self.radius = r
#
#     def area(self):
#         a = np.pi * self.radius**2
#         return a
#
#     def perimeter(self):
#         p = 2 * np.pi * self.radius
#         return p
#
#
# New_Circle = Circle(5)
# print(New_Circle.area())
# print(New_Circle.perimeter())


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def show(self):
        return self.x, self.y

    def distance(self):
        dis = np.sqrt(self.x**2 + self.y**2)
        return dis


class Point2(Point):
    def move(self, dx, dy):
        self.x = self.x + dx
        self.y = self.y + dy


a = Point2(0, 0)
print(a)
print(a.show())
a.move(2, 4)
print(a.show())
print(a.distance())
