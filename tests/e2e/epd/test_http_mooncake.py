# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
import os

try:
    import pytest
    import copy
except ImportError:
    pass

from ..conftest import RemoteEPDServer
from .conftest import load_config

try:
    from tools.aisbench import run_aisbench_cases
except (ImportError, ModuleNotFoundError):
    pass

model_path = load_config().get("model_path")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
DATASET_PATH = load_config().get("dataset_path")

TENSOR_PARALLELS = [1]
DATASET_NAME = ["simulate_truth"]

MOONCAKE_PRODUCER_CONFIG_PATH = (
    load_config().get("mooncake_config_path") + "producer.json"
)
MOONCAKE_CONSUMER_CONFIG_PATH = (
    load_config().get("mooncake_config_path") + "consumer.json"
)
REQUEST_RATE = [0.28, 0.56, 0.84, 1.12, 1.4, 1.68]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e2pd_001_mooncake(
    model: str, tp_size: int, request_rate: float, dataset_name: str
):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        model,
        "--gpu-memory-utilization",
        "0.0",
        "--port",
        "19535",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--enable-request-id-headers",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "10000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"'
        + MOONCAKE_PRODUCER_CONFIG_PATH
        + '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
    ]

    pd_server_args = [
        model,
        "--gpu-memory-utilization",
        "0.95",
        "--port",
        "19536",
        "--tensor-parallel-size",
        str(tp_size),
        "--enforce-eager",
        "--enable-request-id-headers",
        "--max-model-len",
        "10000",
        "--max-num-batched-tokens",
        "10000",
        "--max-num-seqs",
        "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"'
        + MOONCAKE_CONSUMER_CONFIG_PATH
        + '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]

    mooncake_args = [
        "--rpc_port",
        "50051",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8081",
        "--rpc_thread_num",
        "8",
        "--default_kv_lease_ttl",
        "0",
        "eviction_ratio",
        "0.05",
        "--eviction_high_watermark_ratio",
        "0.9",
    ]

    warmup_cases = [
        {
            "case_type": "performance",
            "dataset_path": os.path.join(DATASET_PATH, dataset_name),
            "request_conf": "vllm_api_stream_chat",
            "dataset_conf": "textvqa/textvqa_gen",
            "num_prompts": 50,
            "max_out_len": 256,
            "batch_size": 16,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "request_rate": 0,
            "seed": 77,
        }
    ]

    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake",
        "threshold": 0.97,
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    api_port = 10001
    async with RemoteEPDServer(
        run_mode="serve",
        store_type="mooncake",
        proxy_type="disagg_proxy",
        api_server_port=api_port,
        pd_num=2,
        e_num=1,
        env_dict=env_dict,
        e_serve_args=e_server_args,
        pd_serve_args=pd_server_args,
        mooncake_args=mooncake_args,
    ) as _:
        # warm up
        run_aisbench_cases(
            model=model,
            port=api_port,
            aisbench_cases=warmup_cases,
            verify=False,
            save=False,
        )
        # aisbench test
        run_aisbench_cases(
            model=model, port=api_port, aisbench_cases=aisbench_cases
        )
