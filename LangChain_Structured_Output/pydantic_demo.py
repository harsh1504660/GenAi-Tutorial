from pydantic import BaseModel,EmailStr,Field
from typing import Optional
class Student(BaseModel):
    name:str
    age:Optional[int] = None
    email:EmailStr
    cgpa:float = Field(gt=0,lt=10,description="CGPA should be between 0 and 10")


new_student = {'name': '2', 'age': '20','email':'abc','cgpa':5}
## will thow an error as email should be a valid email string
## will throiw error as name should be a string
# new_student = {'name': 'Atharva'}

student = Student(**new_student)

print(student.model_dump_json())