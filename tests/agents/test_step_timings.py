import json

import pytest

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import LimitsExceeded
from minisweagent.models.test_models import DeterministicModel, make_output


def test_step_timings_include_multiple_actions_and_submission(tmp_path):
    agent = DefaultAgent(
        DeterministicModel(outputs=[
            make_output("", [{"command": "/sleep 0.02"}]),
            make_output("", [{"command": "sleep 0.02"}, {"command": "sleep 0.02"}]),
            make_output("", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo patch"}]),
        ]),
        LocalEnvironment(), system_template="", instance_template="{{task}}", cost_limit=0,
        output_path=tmp_path / "trajectory.json",
    )
    assert agent.run("test")["exit_status"] == "Submitted"
    saved = json.loads((tmp_path / "trajectory.json").read_text())
    assert saved["info"]["model_stats"]["api_calls"] == len(saved["step_timings"]) == 2
    first, last = saved["step_timings"]
    assert first["inference"]["elapsed_s"] >= 0.02
    assert len(first["tools"]) == 2
    assert sum(tool["elapsed_s"] for tool in first["tools"]) >= 0.04
    assert first["inference"]["end_mono_ns"] <= first["tools"][0]["start_mono_ns"]
    assert first["tools"][0]["end_mono_ns"] <= first["tools"][1]["start_mono_ns"]
    assert last["tools"][0]["outcome"] == "Submitted"
    assert last["tools"][0]["elapsed_s"] > 0


def test_failed_query_is_timed_but_limit_check_is_not_a_step():
    agent = DefaultAgent(
        DeterministicModel(outputs=[make_output("", [{"raise": ValueError("bad response")}])]),
        LocalEnvironment(), system_template="", instance_template="", step_limit=1,
    )
    with pytest.raises(ValueError, match="bad response"):
        agent.query()
    with pytest.raises(LimitsExceeded):
        agent.query()
    assert len(agent.step_timings) == agent.n_calls == 1
    assert agent.step_timings[0]["inference"]["outcome"] == "ValueError"
    assert agent.step_timings[0]["inference"]["elapsed_s"] >= 0
    assert agent.step_timings[0]["tools"] == []
