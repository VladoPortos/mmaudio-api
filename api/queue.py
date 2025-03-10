import logging
import threading
from typing import Dict, Any, List, Callable, Optional
import time

logger = logging.getLogger(__name__)

class TaskQueue:
    def __init__(self):
        self.queue = []  # List of task_ids in order
        self.tasks = {}  # Dict of task_id -> task info
        self.current_task = None
        self.lock = threading.Lock()
    
    def add_task(self, task_id: str, func: Callable, params: Dict[str, Any]) -> int:
        """
        Add a task to the queue and return its position in the queue.
        """
        with self.lock:
            position = len(self.queue)
            self.queue.append(task_id)
            self.tasks[task_id] = {
                "id": task_id,
                "func": func,
                "params": params,
                "status": "queued",
                "position": position,
                "result": None,
                "error": None,
                "created_at": time.time()
            }
            logger.info(f"Task {task_id} added to queue at position {position}")
            return position
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            # Update position for queued tasks
            if task["status"] == "queued":
                try:
                    task["position"] = self.queue.index(task_id)
                except ValueError:
                    # Task is in tasks dict but not in queue (shouldn't happen)
                    task["position"] = None
            
            return task
    
    def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from the queue.
        """
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            if task_id in self.queue:
                self.queue.remove(task_id)
                
                # Update positions for remaining tasks
                for i, tid in enumerate(self.queue):
                    self.tasks[tid]["position"] = i
            
            del self.tasks[task_id]
            return True
    
    def get_all_task_ids(self) -> List[str]:
        """
        Get all task IDs.
        """
        with self.lock:
            return list(self.tasks.keys())
    
    def process_next_task(self):
        """
        Process the next task in the queue.
        """
        task_id = None
        
        with self.lock:
            if not self.queue:
                logger.info("No tasks in queue")
                return
            
            task_id = self.queue[0]
            self.current_task = task_id
            task = self.tasks[task_id]
            task["status"] = "processing"
            
            # Remove from queue but keep in tasks
            self.queue.pop(0)
            
            # Update positions for remaining tasks
            for i, tid in enumerate(self.queue):
                self.tasks[tid]["position"] = i
        
        # Process the task outside the lock
        try:
            logger.info(f"Processing task {task_id}")
            result = task["func"](**task["params"])
            
            with self.lock:
                if task_id in self.tasks:  # Task might have been deleted
                    task = self.tasks[task_id]
                    task["status"] = "completed"
                    task["result"] = result
                    task["completed_at"] = time.time()
                    logger.info(f"Task {task_id} completed")
        except Exception as e:
            logger.exception(f"Error processing task {task_id}: {str(e)}")
            
            with self.lock:
                if task_id in self.tasks:  # Task might have been deleted
                    task = self.tasks[task_id]
                    task["status"] = "failed"
                    task["error"] = str(e)
        finally:
            with self.lock:
                if self.current_task == task_id:
                    self.current_task = None
                
                # Process next task if any
                if self.queue:
                    # Create a new thread to process the next task
                    threading.Thread(target=self.process_next_task).start()
