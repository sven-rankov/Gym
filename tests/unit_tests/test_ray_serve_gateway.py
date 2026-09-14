# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import socket
import urllib.error
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from nemo_gym.orchestration import ray_serve_gateway
from nemo_gym.orchestration.ray_serve_gateway import (
    HEALTH_PATH,
    VLLMInstance,
    build_instance_command,
    free_local_port,
    main,
    max_replicas_per_node,
    parse_args,
)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_required_fields():
    args = parse_args(["--model", "org/model", "--port", "8000"])
    assert args.model == "org/model"
    assert args.port == 8000
    assert args.tensor_parallel_size == 1
    assert args.pipeline_parallel_size == 1
    assert args.number_of_instances == 1
    assert args.trust_remote_code is False


def test_parse_args_all_fields():
    args = parse_args(
        [
            "--model",
            "org/model",
            "--port",
            "9000",
            "--tensor-parallel-size",
            "8",
            "--pipeline-parallel-size",
            "2",
            "--number-of-instances",
            "4",
            "--trust-remote-code",
        ]
    )
    assert args.tensor_parallel_size == 8
    assert args.pipeline_parallel_size == 2
    assert args.number_of_instances == 4
    assert args.trust_remote_code is True


def test_parse_args_served_model_name_and_extra_args():
    args = parse_args(
        ["--model", "org/model", "--port", "8000", "--served-model-name", "my-model", "--extra-args", "--foo bar"]
    )
    assert args.served_model_name == "my-model"
    assert args.extra_args == "--foo bar"


def test_parse_args_served_model_name_and_extra_args_default_to_none_and_empty():
    args = parse_args(["--model", "org/model", "--port", "8000"])
    assert args.served_model_name is None
    assert args.extra_args == ""


def test_parse_args_missing_required_raises():
    with pytest.raises(SystemExit):
        parse_args(["--port", "8000"])


def test_parse_args_accepts_gpus_per_node_for_caller_compatibility():
    args = parse_args(["--model", "org/model", "--port", "8000", "--gpus-per-node", "8"])
    assert args.gpus_per_node == 8


# ---------------------------------------------------------------------------
# free_local_port
# ---------------------------------------------------------------------------


def test_free_local_port_returns_a_usable_port():
    port = free_local_port()
    assert 0 < port < 65536
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", port))


def test_free_local_port_returns_distinct_ports_across_calls():
    ports = {free_local_port() for _ in range(20)}
    assert len(ports) == 20


# ---------------------------------------------------------------------------
# max_replicas_per_node
# ---------------------------------------------------------------------------


def test_max_replicas_per_node_none_without_gpus_per_node_info():
    assert max_replicas_per_node(tensor_parallel_size=1, pipeline_parallel_size=1, gpus_per_node=None) is None


def test_max_replicas_per_node_allows_multiple_instances_to_share_a_node():
    # TP2 instances comfortably share one 8-GPU node - up to 4 of them.
    assert max_replicas_per_node(tensor_parallel_size=2, pipeline_parallel_size=1, gpus_per_node=8) == 4


def test_max_replicas_per_node_one_when_footprint_exactly_fills_a_node():
    # TP8 fills the whole 8-GPU node - no room for a second instance's driver there.
    assert max_replicas_per_node(tensor_parallel_size=8, pipeline_parallel_size=1, gpus_per_node=8) == 1


def test_max_replicas_per_node_one_when_footprint_exceeds_a_node():
    # TP8 x PP2 = 16 GPUs/instance, spans 2 nodes - no other instance's driver may share either node.
    assert max_replicas_per_node(tensor_parallel_size=8, pipeline_parallel_size=2, gpus_per_node=8) == 1


# ---------------------------------------------------------------------------
# build_instance_command
# ---------------------------------------------------------------------------


def test_build_instance_command_basic():
    cmd = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=False, port=8001
    )
    assert cmd[:3] == ["vllm", "serve", "org/model"]
    assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "8001"
    assert "--tensor-parallel-size" in cmd
    assert "--distributed-executor-backend" in cmd
    assert cmd[cmd.index("--distributed-executor-backend") + 1] == "ray"


def test_build_instance_command_uses_given_port():
    cmd = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=False, port=9001
    )
    assert cmd[cmd.index("--port") + 1] == "9001"


def test_build_instance_command_pipeline_parallel_flag_only_when_gt_1():
    cmd = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=False, port=8001
    )
    assert "--pipeline-parallel-size" not in cmd

    cmd2 = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=2, trust_remote_code=False, port=8001
    )
    assert "--pipeline-parallel-size" in cmd2
    assert cmd2[cmd2.index("--pipeline-parallel-size") + 1] == "2"


def test_build_instance_command_trust_remote_code():
    cmd = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=True, port=8001
    )
    assert "--trust-remote-code" in cmd


def test_build_instance_command_no_trust_remote_code_by_default():
    cmd = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=False, port=8001
    )
    assert "--trust-remote-code" not in cmd


def test_build_instance_command_served_model_name():
    cmd = build_instance_command(
        model="org/model",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        trust_remote_code=False,
        port=8001,
        served_model_name="my-model",
    )
    assert cmd[cmd.index("--served-model-name") + 1] == "my-model"


def test_build_instance_command_no_served_model_name_by_default():
    cmd = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=False, port=8001
    )
    assert "--served-model-name" not in cmd


def test_build_instance_command_extra_args_split_into_separate_tokens():
    cmd = build_instance_command(
        model="org/model",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        trust_remote_code=False,
        port=8001,
        extra_args="--max-model-len 8192",
    )
    assert "--max-model-len" in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "8192"


# ---------------------------------------------------------------------------
# VLLMInstance
# ---------------------------------------------------------------------------


def _vllm_instance_impl() -> type:
    """The plain class under Ray Serve's `@serve.deployment` and `@serve.ingress(app)` wrappers.

    `func_or_class` is the ingress wrapper, whose `__init__` is async and expects a Serve replica
    context. The class defined in the module sits in its MRO with the plain `__init__`; instantiating
    that directly exercises the real replica logic without a Ray cluster.
    """
    for klass in VLLMInstance.func_or_class.__mro__:
        if klass.__module__ == ray_serve_gateway.__name__ and not inspect.iscoroutinefunction(klass.__init__):
            return klass
    raise AssertionError("Ray Serve changed how it wraps deployments; update _vllm_instance_impl")


_VLLMInstanceImpl = _vllm_instance_impl()

# Ray copies the class when it builds the deployment, so its methods see a copy of the module
# globals: names such as `free_local_port` or `HEALTH_TIMEOUT_S` cannot be patched on the module.
# Attributes of the shared `subprocess`, `urllib`, `time`, `ray` and `aiohttp` modules can.


def _health_response(status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.__enter__.return_value = resp
    return resp


def _instance(proc: MagicMock, *, urlopen_side_effect=None, **init_kwargs):
    """Build a replica with the vLLM subprocess, Ray runtime context and HTTP calls stubbed out."""
    kwargs = dict(model="org/model", tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=False)
    kwargs.update(init_kwargs)
    with (
        patch.object(ray_serve_gateway.subprocess, "Popen", return_value=proc) as popen,
        patch.object(
            ray_serve_gateway.ray, "get_runtime_context", return_value=MagicMock(gcs_address="10.0.0.1:6379")
        ),
        patch.object(ray_serve_gateway.aiohttp, "ClientSession", return_value=MagicMock()),
        patch.object(ray_serve_gateway.urllib.request, "urlopen", side_effect=urlopen_side_effect) as urlopen,
        patch.object(ray_serve_gateway.time, "sleep"),
    ):
        instance = _VLLMInstanceImpl(**kwargs)
    return instance, popen, urlopen


def _running_proc() -> MagicMock:
    proc = MagicMock()
    proc.poll.return_value = None
    return proc


def test_vllm_instance_launches_vllm_serve_on_its_own_port_inside_the_ray_cluster():
    instance, popen, urlopen = _instance(
        _running_proc(),
        urlopen_side_effect=[_health_response()],
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        trust_remote_code=True,
        served_model_name="my-model",
        extra_args="--max-model-len 8192",
    )

    cmd = popen.call_args.args[0]
    port = int(cmd[cmd.index("--port") + 1])
    assert cmd == build_instance_command("org/model", 2, 2, True, port, "my-model", "--max-model-len 8192")
    # vLLM's own Ray executor must join the gateway's cluster rather than start a new one.
    assert popen.call_args.kwargs["env"]["RAY_ADDRESS"] == "10.0.0.1:6379"
    assert instance._base_url == f"http://localhost:{port}"
    assert urlopen.call_args.args[0] == f"http://localhost:{port}{HEALTH_PATH}"


def test_vllm_instance_keeps_polling_until_the_health_endpoint_answers_200():
    _, _, urlopen = _instance(
        _running_proc(),
        urlopen_side_effect=[
            urllib.error.URLError("connection refused"),
            TimeoutError(),
            _health_response(status=503),
            _health_response(status=200),
        ],
    )

    assert urlopen.call_count == 4


def test_vllm_instance_raises_when_vllm_exits_before_becoming_healthy():
    proc = MagicMock()
    proc.poll.return_value = 3
    proc.returncode = 3

    with pytest.raises(RuntimeError, match="exited early with code 3"):
        _instance(proc, urlopen_side_effect=[_health_response()])


def test_vllm_instance_raises_when_the_health_deadline_passes():
    # First call sets the deadline; the second is already past it.
    with patch.object(ray_serve_gateway.time, "monotonic", side_effect=[0.0, ray_serve_gateway.HEALTH_TIMEOUT_S + 1]):
        with pytest.raises(TimeoutError, match="did not become healthy in time"):
            _instance(_running_proc(), urlopen_side_effect=urllib.error.URLError("connection refused"))


def test_vllm_instance_check_health_passes_while_vllm_runs_and_fails_once_it_exits():
    proc = _running_proc()
    instance, _, _ = _instance(proc, urlopen_side_effect=[_health_response()])

    instance.check_health()

    proc.poll.return_value = 1
    proc.returncode = 1
    with pytest.raises(RuntimeError, match="exited with code 1"):
        instance.check_health()


async def test_vllm_instance_health_endpoint_returns_200():
    instance, _, _ = _instance(_running_proc(), urlopen_side_effect=[_health_response()])

    response = await instance.health()

    assert response.status_code == 200


def _proxy_instance(upstream_status: int, upstream_headers: dict[str, str], upstream_body: bytes):
    """A healthy replica whose aiohttp session returns a canned vLLM response."""
    instance, _, _ = _instance(_running_proc(), urlopen_side_effect=[_health_response()])
    upstream = MagicMock()
    upstream.status = upstream_status
    upstream.headers = upstream_headers
    upstream.read = AsyncMock(return_value=upstream_body)
    request_cm = MagicMock()
    request_cm.__aenter__ = AsyncMock(return_value=upstream)
    request_cm.__aexit__ = AsyncMock(return_value=False)
    instance._session = MagicMock()
    instance._session.request.return_value = request_cm
    return instance


def _gateway_request(method: str, headers: dict[str, str], query_params: dict[str, str], body: bytes) -> MagicMock:
    request = MagicMock()
    request.method = method
    request.headers = headers
    request.query_params = query_params
    request.body = AsyncMock(return_value=body)
    return request


async def test_vllm_instance_proxy_forwards_the_request_and_relays_the_vllm_response():
    instance = _proxy_instance(201, {"Content-Type": "application/json", "Content-Length": "2"}, b"{}")
    request = _gateway_request(
        "POST",
        {"Host": "gateway:8000", "Content-Length": "7", "Authorization": "Bearer token"},
        {"stream": "false"},
        b'{"a":1}',
    )

    response = await instance.proxy(request, "v1/chat/completions")

    # `Host` and `Content-Length` are dropped: vLLM must see its own host, and aiohttp sizes the body itself.
    assert instance._session.request.call_args == call(
        "POST",
        f"{instance._base_url}/v1/chat/completions",
        params={"stream": "false"},
        data=b'{"a":1}',
        headers={"Authorization": "Bearer token"},
    )
    assert response.status_code == 201
    assert response.body == b"{}"
    assert response.headers["content-type"] == "application/json"
    assert response.headers["content-length"] == "2"


async def test_vllm_instance_proxy_does_not_relay_hop_by_hop_headers_from_a_chunked_vllm_response():
    # A streamed chat completion arrives chunked. The proxy re-sends the collected body with its own
    # Content-Length, so relaying Transfer-Encoding as well yields a response aiohttp clients reject
    # ("Content-Length can't be present with Transfer-Encoding"); `date`/`server` would be duplicated
    # by the gateway's own server.
    body = b"data: {}\n\ndata: [DONE]\n\n"
    instance = _proxy_instance(
        200,
        {
            "Content-Type": "text/event-stream",
            "Transfer-Encoding": "chunked",
            "Connection": "keep-alive",
            "Date": "Mon, 14 Sep 2026 20:54:29 GMT",
            "Server": "uvicorn",
            "X-Request-Id": "abc",
        },
        body,
    )
    request = _gateway_request("POST", {"Host": "gateway:8000"}, {}, b'{"stream": true}')

    response = await instance.proxy(request, "v1/chat/completions")

    assert response.status_code == 200
    assert response.body == body
    assert response.headers["content-type"] == "text/event-stream"
    assert response.headers["x-request-id"] == "abc"
    assert response.headers["content-length"] == str(len(body))
    for header in ("transfer-encoding", "connection", "date", "server"):
        assert header not in response.headers


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class _StopGateway(Exception):
    """Raised from the patched `time.sleep` to leave `main`'s keep-alive loop."""


def _run_main(argv: list[str], *, ray_init_side_effect=None, environ: dict[str, str] | None = None):
    deployment = MagicMock()
    with (
        patch.object(ray_serve_gateway.ray, "init", side_effect=ray_init_side_effect) as ray_init,
        patch.object(ray_serve_gateway, "VLLMInstance", deployment),
        patch.object(ray_serve_gateway.serve, "start") as serve_start,
        patch.object(ray_serve_gateway.serve, "run") as serve_run,
        patch.object(ray_serve_gateway.time, "sleep", side_effect=_StopGateway),
        patch.dict(ray_serve_gateway.os.environ, environ or {}, clear=True),
    ):
        with pytest.raises(_StopGateway):
            main(argv)
        environ = dict(ray_serve_gateway.os.environ)
    return ray_init, deployment, serve_start, serve_run, environ


def test_main_joins_the_existing_ray_cluster_and_serves_the_deployment():
    ray_init, deployment, serve_start, serve_run, environ = _run_main(
        [
            "--model",
            "org/model",
            "--port",
            "8000",
            "--tensor-parallel-size",
            "8",
            "--pipeline-parallel-size",
            "2",
            "--number-of-instances",
            "2",
            "--gpus-per-node",
            "8",
            "--trust-remote-code",
            "--served-model-name",
            "my-model",
            "--extra-args",
            "--max-model-len 8192",
        ]
    )

    assert ray_init.call_args_list == [call(address="auto")]
    # TP8 x PP2 spans two 8-GPU nodes, so no node may host a second instance's driver.
    assert deployment.options.call_args == call(num_replicas=2, max_replicas_per_node=1)
    assert deployment.options.return_value.bind.call_args == call(
        model="org/model",
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        trust_remote_code=True,
        served_model_name="my-model",
        extra_args="--max-model-len 8192",
    )
    assert serve_start.call_args == call(http_options={"host": "0.0.0.0", "port": 8000})
    assert serve_run.call_args == call(deployment.options.return_value.bind.return_value)
    assert environ["RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S"] == "1.0"


def test_main_starts_a_local_ray_cluster_when_there_is_none_to_join():
    ray_init, deployment, _, _, _ = _run_main(
        ["--model", "org/model", "--port", "8000"],
        ray_init_side_effect=[ConnectionError("no cluster"), None],
    )

    assert ray_init.call_args_list == [call(address="auto"), call()]
    # Without --gpus-per-node, Serve packs replicas freely.
    assert deployment.options.call_args == call(num_replicas=1, max_replicas_per_node=None)


def test_main_keeps_an_existing_queue_length_deadline():
    # The multi-node sbatch path exports this before `ray start`; main must not override it.
    _, _, _, _, environ = _run_main(
        ["--model", "org/model", "--port", "8000"],
        environ={"RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S": "5.0"},
    )

    assert environ["RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S"] == "5.0"
