name = "Sachi"
age = 21

print("Hello",name)
print("Age",age)

skills = ["Java","React","SpringBoot"]
print("Skills",skills)

for skill in skills:
    print("I know:", skill)

def greet(name):
    return "Hello " + name

print(greet("sachi"))
print(greet("world"))

age = 20
if age >= 18:
    print("You are old enough")
else:
    print("You are Minor")

person = {
    "name": "Sachi",
    "age": 21,
    "skills": ["Java","React","SpringBoot"]
}

print(person["name"])
print(person["age"])
print(person["skills"])

