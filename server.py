import json
import uuid
from typing import List
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path

from swebench.harness.run_evaluation import main as run_evaluation

app = FastAPI()

class PullRequestSolution(BaseModel):
    model_name_or_path: str
    instance_id: str 
    model_patch: str

class EvaluationRequest(BaseModel):
    solutions: List[PullRequestSolution]
    dataset_name: str = "princeton-nlp/SWE-bench_Lite"
    split: str = "test"
    max_workers: int = 4
    timeout: int = 1800

@app.post("/evaluate")
async def evaluate_solutions(request: EvaluationRequest):
    # Generate unique run ID
    run_id = str(uuid.uuid4())
    
    # Create predictions file
    predictions = [
        {
            "model_name_or_path": sol.model_name_or_path,
            "instance_id": sol.instance_id,
            "prediction": sol.model_patch
        }
        for sol in request.solutions
    ]
    
    # Save predictions to temporary file
    predictions_path = f"predictions_{run_id}.json"
    with open(predictions_path, "w") as f:
        json.dump(predictions, f)

    try:
        # Run evaluation
        run_evaluation(
            dataset_name=request.dataset_name,
            split=request.split,
            instance_ids=None,  # Run all provided instances
            predictions_path=predictions_path,
            max_workers=request.max_workers,
            force_rebuild=False,
            cache_level="env",
            clean=False,
            open_file_limit=4096,
            run_id=run_id,
            timeout=request.timeout,
            exclude_completed=True
        )

        # Read results
        report_file = Path(f"{predictions[0]['model_name_or_path'].replace('/', '__')}.{run_id}.json")
        if not report_file.exists():
            raise HTTPException(status_code=500, detail="Evaluation failed - no report generated")
            
        with open(report_file) as f:
            results = json.load(f)
            
        return JSONResponse(content=results)

    finally:
        # Cleanup
        try:
            Path(predictions_path).unlink()
            if report_file.exists():
                report_file.unlink()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
