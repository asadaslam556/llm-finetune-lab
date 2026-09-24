"""Pipeline control: start a run, poll status.

Start returns 202 immediately with the run id. The frontend then polls
/api/pipeline/status, which just reads the atomic status file, every couple
of seconds to animate the rail. 409 when a run is already going.
"""

from fastapi import APIRouter, HTTPException

from ...core.errors import PipelineBusy
from ...pipeline.runner import get_store, is_running, start_run
from ..schemas import PipelineRunAccepted, PipelineRunRequest

router = APIRouter(tags=["pipeline"])


@router.post("/api/pipeline/run", response_model=PipelineRunAccepted, status_code=202)
def run_pipeline(req: PipelineRunRequest):
    try:
        return start_run(dry_run=req.dry_run, stages=req.stages)
    except PipelineBusy as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/api/pipeline/status")
def pipeline_status():
    # `busy` comes from the lock, not the file. The runner writes its final
    # status a moment before it releases the lock, so a UI that trusted the
    # file alone would re-enable the start button a hair early and hand the
    # user a 409 for their trouble.
    return {**get_store().read(), "busy": is_running()}
