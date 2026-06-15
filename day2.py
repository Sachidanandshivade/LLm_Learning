a = 10
b = 3
print(a+b)
print(a-b)
print(a * b)
print(a / b)
print(a ** 2)

#function with input
def neuron(input,weight,bias):
    return input*weight+bias

print(neuron(2,3,1))
print(neuron(5,0.5,2))

numbers = [1,2,3,4,5]

#sum function
total = sum(numbers)
print("Sum:", total)

#this is called list comprehension
doubled = [x*2 for x in numbers]
print("Doubled:",doubled)


#Lists as vectors

word_cat = [0.2,0.9,0.1]
word_dog=[0.3,0.8,0.2]
word_car=[0.9,0.1,0.8]

def dot_product(a,b):
    total = 0
    for i in range(len(a)):
        total += a[i]*b[i]
    return total

print("cat vs dog:", dot_product(word_cat,word_dog))
print("cat vs car:",dot_product(word_cat,word_car))