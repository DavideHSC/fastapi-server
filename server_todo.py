from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
app = FastAPI()

class Task(BaseModel):
    title: str
    description: str = ""

tasks = []
@app.post("/tasks/")
def create_task(task: Task):
    tasks.append(task)
    return {"msg": "task added succesfully"}
@app.get("/tasks/", response_model=List[Task])
def get_tasks():
    return tasks
@app.get("/tasks/{task_id}", response_model=Task)
def get_task(task_id: int):
    if 0 <= task_id < len(tasks):
        return tasks[task_id]
    else:
        raise HTTPException(status_code=404, detail="Task not found")
@app.put("/tasks/{task_id}", response_model=Task)
def update_task(task_id: int, updated_task: Task):
    if 0 <= task_id < len(tasks):
        tasks[task_id] = updated_task
        return updated_task
    else:
        raise HTTPException(status_code=404, detail="Task not found")
@app.delete("/tasks/{task_id}", response_model=Task)
def delete_task(task_id: int):
    if 0 <= task_id < len(tasks):
        deleted_task = tasks.pop(task_id)
        return deleted_task
    else:
        raise HTTPException(status_code=404, detail="Task not found")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)