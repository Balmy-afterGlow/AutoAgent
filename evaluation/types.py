from pydantic import BaseModel
from typing import Any, List
import json
import logging

logger = logging.getLogger(__name__)


class EvalMetadata(BaseModel):
    agent_func: str
    model: str
    eval_output_dir: str
    start_time: str
    dataset: str | None = None
    data_split: str | None = None
    details: dict[str, Any] | None = None
    container_name: str | None = None
    port: int | None = None
    git_clone: bool | None = None
    test_pull_name: str | None = None

    def model_dump(self, *args, **kwargs):
        dumped_dict = super().model_dump(*args, **kwargs)
        # avoid leaking sensitive information
        return dumped_dict

    def model_dump_json(self, *args, **kwargs):
        dumped = super().model_dump_json(*args, **kwargs)
        dumped_dict = json.loads(dumped)
        logger.debug(f"Dumped metadata: {dumped_dict}")
        return json.dumps(dumped_dict)


class EvalOutput(BaseModel):
    # NOTE: User-specified
    instance_id: str
    # output of the evaluation
    # store anything that is needed for the score calculation
    test_result: dict[str, Any]

    instruction: str | None = None

    # Interaction info
    metadata: EvalMetadata | None = None
    # list[tuple[dict[str, Any], dict[str, Any]]] - for compatibility with the old format
    messages: List | None = None
    error: str | None = None

    # Optionally save the input test instance
    instance: dict[str, Any] | None = None

    def model_dump(self, *args, **kwargs):
        dumped_dict = super().model_dump(*args, **kwargs)
        # Remove None values
        dumped_dict = {k: v for k, v in dumped_dict.items() if v is not None}
        # Apply custom serialization for metadata (to avoid leaking sensitive information)
        if self.metadata is not None:
            dumped_dict["metadata"] = self.metadata.model_dump()
        return dumped_dict

    def model_dump_json(self, *args, **kwargs):
        dumped = super().model_dump_json(*args, **kwargs)
        dumped_dict = json.loads(dumped)
        # Apply custom serialization for metadata (to avoid leaking sensitive information)
        if "metadata" in dumped_dict:
            dumped_dict["metadata"] = json.loads(self.metadata.model_dump_json())
        return json.dumps(dumped_dict)
