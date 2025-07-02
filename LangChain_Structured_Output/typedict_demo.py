from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int


new_person:Person={'name': 'Atharva', 'age': 20}


print(new_person)