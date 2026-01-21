# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Prism Multi-Model Server.

The entry point for multi-model inference serving.
This server supports multiple LLMs sharing GPUs through flexible scheduling.
"""

import asyncio
import atexit
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import signal
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import orjson
import requests
import torch
import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, ORJSONResponse, Response, StreamingResponse
from uvicorn.config import LOGGING_CONFIG

# SGLang imports
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    add_api_key_middleware,
    configure_logger,
    is_port_available,
    kill_child_process,
    prepare_model_and_tokenizer,
    set_ulimit,
)
from sglang.utils import get_exception_traceback

# Prism imports
from prism.io_struct import (
    ActivateReqInput,
    DeactivateReqInput,
    GetMemPoolSizeReq,
    MemoryUsage,
    ResizeMemPoolReqInput,
)
from prism.multi_model.server_args import MultiModelServerArgs
from prism.utils.redis_utils import RedisClient

# Fix a bug of Python threading
import threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

logger = logging.getLogger(__name__)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


# FastAPI application
app = FastAPI(title="Prism Multi-Model Server")
request_handler = None
model_names = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
async def health() -> Response:
    """Check the health of the HTTP server."""
    return Response(status_code=200)


@app.get("/get_model_names")
async def get_model_names():
    """Get the available model names."""
    return list(model_names) if model_names else []


@app.api_route("/deactivate", methods=["GET", "POST"])
async def deactivate(obj: DeactivateReqInput):
    """Deactivate a model to release GPU memory."""
    tic = time.time()
    try:
        success, memory_usage = await request_handler.deactivate(obj)
        logger.info(f"[Server] Deactivate time cost: {time.time() - tic:.4f}s")
        return ORJSONResponse({
            "success": success,
            "message": "Model deactivated successfully",
            "memory_usage": memory_usage.to_dict() if memory_usage else None,
        })
    except Exception as e:
        logger.error(f"Error: {get_exception_traceback()}")
        return ORJSONResponse(
            {"success": False, "error": {"message": str(e), "type": type(e).__name__}},
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.api_route("/activate", methods=["GET", "POST"])
async def activate(obj: ActivateReqInput):
    """Activate a model on a specific GPU."""
    try:
        success, memory_usage = await request_handler.activate(obj)
        return ORJSONResponse({
            "success": success,
            "message": "Model activated successfully",
            "memory_usage": memory_usage.to_dict() if memory_usage else None,
        })
    except Exception as e:
        return ORJSONResponse(
            {"success": False, "error": {"message": str(e), "type": type(e).__name__}},
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.api_route("/get_memory_pool_size", methods=["GET", "POST"])
async def get_memory_pool_size(obj: GetMemPoolSizeReq):
    """Get the memory pool size in number of tokens."""
    try:
        ret = await request_handler.get_memory_pool_size(obj)
        return ret.size
    except Exception as e:
        logger.error(f"Error: {get_exception_traceback()}")
        return JSONResponse(
            {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
        )


@app.api_route("/resize_mem_pool", methods=["GET", "POST"])
async def resize_mem_pool(obj: ResizeMemPoolReqInput):
    """Resize the memory pool."""
    request_handler.resize_mem_pool(obj)
    return Response(status_code=200)


async def generate_request(obj, request: Request):
    """Handle a generate request."""
    # Import here to avoid circular imports
    from sglang.srt.managers.io_struct import GenerateReqInput
    
    if obj.stream:
        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in request_handler.generate_request(obj, request):
                    yield b"data: " + orjson.dumps(
                        out, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
            except ValueError as e:
                logger.error(f"Error: {get_exception_traceback()}")
                out = {"error": {"message": str(e)}}
                yield b"data: " + orjson.dumps(
                    out, option=orjson.OPT_NON_STR_KEYS
                ) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=request_handler.create_abort_task(obj),
        )
    else:
        try:
            ret = await request_handler.generate_request(obj, request).__anext__()
            return ret
        except ValueError as e:
            return ORJSONResponse(
                {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
            )


app.post("/generate")(generate_request)
app.put("/generate")(generate_request)


# =============================================================================
# Engine Info
# =============================================================================

@dataclasses.dataclass
class EngineInfo:
    """Information about a running engine."""
    port_args: PortArgs
    model_path: str
    gpu_ids: List[int]
    model_name: str
    instance_idx: int
    memory_usage: MemoryUsage
    init_memory_pool_size: float
    on: bool = True


# =============================================================================
# Launch Functions
# =============================================================================

def launch_engine(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_ids: Optional[List[int]] = None,
    instance_idx: Optional[int] = 0,
    shared_cpu_models: Optional[Dict[Tuple[str, int], List[Any]]] = None,
    model_names_to_model_paths: Optional[Dict[str, str]] = None,
    engine_id: Optional[str] = None,
    input_queue=None,
    output_queue=None,
) -> EngineInfo:
    """
    Launch a single engine (Scheduler + Detokenizer).
    """
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Prepare model and tokenizer paths
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    # Launch scheduler processes
    scheduler_procs = []
    scheduler_pipe_readers = []
    tp_size_per_node = server_args.tp_size // server_args.nnodes
    tp_rank_range = range(
        tp_size_per_node * server_args.node_rank,
        tp_size_per_node * (server_args.node_rank + 1),
    )
    
    assert len(tp_rank_range) == len(gpu_ids)
    
    for tp_rank in tp_rank_range:
        reader, writer = mp.Pipe(duplex=False)
        gpu_id = gpu_ids[tp_rank % tp_size_per_node]
        proc = mp.Process(
            target=run_scheduler_process,
            args=(
                server_args,
                port_args,
                gpu_id,
                tp_rank,
                None,
                writer,
                shared_cpu_models,
                model_names_to_model_paths,
                engine_id,
                input_queue,
                output_queue,
            ),
        )
        proc.start()
        scheduler_procs.append(proc)
        scheduler_pipe_readers.append(reader)

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
            model_names_to_model_paths,
        ),
    )
    detoken_proc.start()

    # Wait for model to finish loading
    memory_usage = None
    for i in range(len(scheduler_pipe_readers)):
        memory_usage = scheduler_pipe_readers[i].recv()

    logger.info(
        f"Model {server_args.model_name} instance {instance_idx} loaded "
        f"in process {scheduler_procs[-1].pid}"
    )
    
    return EngineInfo(
        port_args=port_args,
        gpu_ids=gpu_ids,
        model_name=server_args.model_name,
        model_path=server_args.model_path,
        instance_idx=instance_idx,
        memory_usage=memory_usage,
        on=getattr(server_args, 'on', True),
        init_memory_pool_size=getattr(server_args, 'max_memory_pool_size', None),
    )


def launch_request_handler(
    server_args: MultiModelServerArgs,
    port_args_dict: Dict[str, PortArgs],
    request_handler_ipc_name: str,
    num_engines: int,
    gpu_id_to_model_instance: Dict[int, Dict[str, int]],
    pipe_finish_writer: Optional[mp.connection.Connection] = None,
    controller_ipc_name: Optional[str] = None,
):
    """
    Launch request handler for routing requests.
    """
    # Check Redis connection
    try:
        redis_client = RedisClient(
            server_args.redis_host, server_args.redis_port, server_args.redis_db
        )
        redis_client.client.ping()
    except Exception as e:
        logger.error(
            f"Redis server is not running at {server_args.redis_host}:{server_args.redis_port}. "
            "Please start the Redis server first."
        )
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(get_exception_traceback())
        kill_child_process(os.getpid(), including_parent=False)
        return

    # Clear the Redis queue
    redis_client.clear_queue()

    global request_handler
    
    if server_args.enable_worker_pool:
        from prism.multi_model.request_handler import RequestHandlerWorkerPool
        request_handler = RequestHandlerWorkerPool(
            server_args,
            port_args_dict,
            num_engines=num_engines,
            num_gpus=server_args.num_gpus,
            ipc_name=request_handler_ipc_name,
            controller_ipc_name=controller_ipc_name,
        )
    else:
        from prism.multi_model.request_handler import RequestHandler
        request_handler = RequestHandler(
            server_args,
            port_args_dict,
            num_engines=num_engines,
            ipc_name=request_handler_ipc_name,
            gpu_id_to_model_instance=gpu_id_to_model_instance,
            controller_ipc_name=controller_ipc_name,
        )


def launch_multi_model_server(
    multi_model_server_args: MultiModelServerArgs,
    pipe_finish_writer: Optional[mp.connection.Connection] = None,
):
    """
    Launch the Prism Multi-Model Server.

    The server consists of:
    1. HTTP server: FastAPI server for receiving requests
    2. Request handler: Routes requests to appropriate engines
    3. Multiple engines: Each handles one model instance on one GPU
    """
    torch.multiprocessing.set_start_method("spawn")

    configure_logger(multi_model_server_args)
    
    # Check port availability
    if not is_port_available(multi_model_server_args.port):
        raise RuntimeError(
            f"Port {multi_model_server_args.port} is not available. "
            "Please choose another port."
        )

    # Create IPC names
    request_handler_ipc_name = tempfile.NamedTemporaryFile(delete=False).name
    request_handler_to_controller_ipc_name = tempfile.NamedTemporaryFile(delete=False).name
    
    schedulers_to_controller_ipc_name = None
    if multi_model_server_args.enable_controller:
        schedulers_to_controller_ipc_name = tempfile.NamedTemporaryFile(delete=False).name

    # Get model configurations
    model_configs = multi_model_server_args.model_configs
    model_names_to_model_paths = {
        model_config.model_name: model_config.model_path
        for model_config in model_configs
    }
    
    global model_names
    model_names = list(model_names_to_model_paths.keys())

    # Load shared CPU models if enabled
    path_to_shared_cpu_models = {}
    if multi_model_server_args.enable_cpu_share_memory:
        path_to_shared_cpu_models = _load_shared_cpu_models(multi_model_server_args)

    # Launch engines
    if multi_model_server_args.enable_worker_pool:
        (
            engine_info_dict,
            port_args_dict,
            gpu_id_to_model_instance,
            num_engines,
            init_placements,
        ) = _launch_worker_pool_engines(
            multi_model_server_args,
            path_to_shared_cpu_models,
            model_names_to_model_paths,
            request_handler_ipc_name,
            schedulers_to_controller_ipc_name,
        )
    else:
        (
            engine_info_dict,
            port_args_dict,
            gpu_id_to_model_instance,
            num_engines,
            init_placements,
        ) = _launch_model_engines(
            multi_model_server_args,
            path_to_shared_cpu_models,
            model_names_to_model_paths,
            request_handler_ipc_name,
            schedulers_to_controller_ipc_name,
        )

    # Launch controller if enabled
    if multi_model_server_args.enable_controller:
        _launch_controller(
            multi_model_server_args,
            recv_from_request_handler_ipc_name=request_handler_to_controller_ipc_name,
            recv_from_schedulers_ipc_name=schedulers_to_controller_ipc_name,
            engine_info_dict=engine_info_dict,
            model_names_to_model_paths=model_names_to_model_paths,
            init_placements=init_placements,
        )

    # Launch GPU schedulers if enabled
    if multi_model_server_args.enable_gpu_scheduler:
        for gpu_id in gpu_id_to_model_instance.keys():
            _launch_gpu_scheduler(
                multi_model_server_args,
                model_names_to_model_paths,
                engine_info_dict,
                gpu_id,
                init_placements.get(gpu_id, []),
            )

    # Launch request handler
    launch_request_handler(
        multi_model_server_args,
        port_args_dict,
        request_handler_ipc_name,
        num_engines,
        gpu_id_to_model_instance,
        pipe_finish_writer,
        controller_ipc_name=request_handler_to_controller_ipc_name,
    )

    # Add API key middleware if configured
    if multi_model_server_args.api_key:
        add_api_key_middleware(app, multi_model_server_args.api_key)

    # Start warmup threads
    threads = []
    url = multi_model_server_args.url()
    
    if multi_model_server_args.enable_worker_pool:
        for gpu_id, model_list in init_placements.items():
            for model_name in model_list:
                t = threading.Thread(
                    target=_wait_and_warmup,
                    args=(url, model_name, pipe_finish_writer, os.getpid()),
                )
                t.start()
                threads.append(t)
    else:
        for model_name, engine_infos in engine_info_dict.items():
            for engine_info in engine_infos:
                if engine_info.on:
                    t = threading.Thread(
                        target=_wait_and_warmup,
                        args=(url, model_name, pipe_finish_writer, os.getpid()),
                    )
                    t.start()
                    threads.append(t)

    # Start HTTP server
    try:
        LOGGING_CONFIG["formatters"]["default"]["fmt"] = "[%(asctime)s] %(levelprefix)s %(message)s"
        LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
        LOGGING_CONFIG["formatters"]["access"]["fmt"] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
        LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
        
        uvicorn.run(
            app,
            host=multi_model_server_args.host,
            port=multi_model_server_args.port,
            log_level=multi_model_server_args.log_level_http or multi_model_server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    except Exception as e:
        kill_child_process(os.getpid(), including_parent=False)
    finally:
        for t in threads:
            t.join()


# =============================================================================
# Helper Functions
# =============================================================================

def _set_envs_and_config(server_args: ServerArgs):
    """Set environment variables and configurations."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    set_ulimit()
    mp.set_start_method("spawn", force=True)


def _load_shared_cpu_models(multi_model_server_args: MultiModelServerArgs) -> Dict:
    """Load shared CPU models for fast activation."""
    from prism.multi_model.utils.cpu_model_loader import (
        init_torch_distributed_tp_1,
        load_shared_cpu_model,
    )
    
    logger.info("Initializing torch distributed...")
    init_torch_distributed_tp_1(device="cpu")
    
    model_configs = multi_model_server_args.model_configs
    model_ids = set(
        (model_config.model_path, model_config.tp_size)
        for model_config in model_configs
    )
    
    model_server_args = [
        ServerArgs(model_path=model_path, tp_size=tp_size)
        for model_path, tp_size in model_ids
    ]
    
    logger.info(f"Loading {len(model_ids)} shared models to CPU...")
    tic = time.time()
    
    max_workers = len(model_ids)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        shared_cpu_models = list(executor.map(load_shared_cpu_model, model_server_args))
    
    logger.info(f"Shared models loaded in {time.time() - tic:.2f} seconds.")
    return dict(zip(model_ids, shared_cpu_models))


def _launch_model_engines(
    multi_model_server_args: MultiModelServerArgs,
    path_to_shared_cpu_models: Dict,
    model_names_to_model_paths: Dict[str, str],
    request_handler_ipc_name: str,
    schedulers_to_controller_ipc_name: Optional[str] = None,
) -> Tuple:
    """Launch model engines for standard mode."""
    model_configs = multi_model_server_args.model_configs
    engine_info_dict = defaultdict(list)
    port_args_dict = defaultdict(list)
    gpu_id_to_model_instance = defaultdict(dict)
    num_engines = 0

    def launch_wrapper(args):
        server_args, port_args, gpu_ids, instance_idx, shared_cpu_models, names_to_paths, engine_id = args
        return (
            server_args.model_name,
            launch_engine(
                server_args=server_args,
                port_args=port_args,
                gpu_ids=gpu_ids,
                instance_idx=instance_idx,
                shared_cpu_models=shared_cpu_models,
                model_names_to_model_paths=names_to_paths,
                engine_id=engine_id,
            ),
        )

    start_time = time.perf_counter()
    engine_launch_args = []
    start_port = multi_model_server_args.port

    for model_config in model_configs:
        instance_configs = model_config.get_instance_configs()
        for i, instance_config in enumerate(instance_configs):
            server_args = ServerArgs.from_multi_model_server_args(
                multi_model_server_args=multi_model_server_args,
                instance_config=instance_config,
            )
            gpu_ids = instance_config.gpu_ids
            
            logger.info(
                f"Preparing engine for {server_args.model_name} on GPU {gpu_ids}"
            )

            port_args = PortArgs.init_with_request_handler_ipc_name(
                start_port, request_handler_ipc_name, schedulers_to_controller_ipc_name
            )
            start_port = port_args.nccl_port
            engine_id = f"{model_config.model_name}_{i}"
            
            engine_launch_args.append((
                server_args,
                port_args,
                gpu_ids,
                i,
                path_to_shared_cpu_models,
                model_names_to_model_paths,
                engine_id,
            ))

    # Launch engines in parallel
    with ThreadPoolExecutor(max_workers=len(engine_launch_args)) as executor:
        all_results = list(executor.map(launch_wrapper, engine_launch_args))

    # Process results
    for model_name, engine_info in all_results:
        engine_info_dict[model_name].append(engine_info)
        port_args_dict[model_name].append(engine_info.port_args)
        num_engines += 1
        gpu_id = engine_info.gpu_ids[0]
        gpu_id_to_model_instance[gpu_id][model_name] = engine_info.instance_idx

    logger.info(
        f"All {num_engines} engines prepared in {time.perf_counter() - start_time:.2f} seconds."
    )
    return engine_info_dict, port_args_dict, gpu_id_to_model_instance, num_engines, None


def _launch_worker_pool_engines(
    multi_model_server_args: MultiModelServerArgs,
    path_to_shared_cpu_models: Dict,
    model_names_to_model_paths: Dict[str, str],
    request_handler_ipc_name: str,
    schedulers_to_controller_ipc_name: Optional[str] = None,
) -> Tuple:
    """Launch worker pool engines."""
    model_configs = multi_model_server_args.model_configs
    engine_info_dict = defaultdict(list)
    port_args_dict = defaultdict(list)
    gpu_id_to_model_instance = defaultdict(dict)
    num_engines = 0

    # Get initial placements
    init_placements = defaultdict(list)
    for model_config in model_configs:
        model_name = model_config.model_name
        instance_configs = model_config.get_instance_configs()
        for instance_config in instance_configs:
            gpu_id = instance_config.gpu_ids[0]
            if instance_config.on:
                init_placements[gpu_id].append(model_name)

    def launch_wrapper(args):
        server_args, port_args, gpu_ids, worker_id, shared_cpu_models, names_to_paths, engine_id = args
        return (
            gpu_ids,
            launch_engine(
                server_args=server_args,
                port_args=port_args,
                gpu_ids=gpu_ids,
                instance_idx=worker_id,
                shared_cpu_models=shared_cpu_models,
                model_names_to_model_paths=names_to_paths,
                engine_id=engine_id,
            ),
        )

    start_time = time.perf_counter()
    engine_launch_args = []
    start_port = multi_model_server_args.port
    workers_per_gpu = multi_model_server_args.workers_per_gpu
    num_gpus = multi_model_server_args.num_gpus

    for gpu_id in range(num_gpus):
        for worker_id in range(workers_per_gpu):
            engine_id = f"{gpu_id}_{worker_id}"
            server_args = ServerArgs.from_multi_model_server_args(
                multi_model_server_args=multi_model_server_args,
                worker_id=worker_id,
            )
            port_args = PortArgs.init_with_request_handler_ipc_name(
                start_port, request_handler_ipc_name, schedulers_to_controller_ipc_name
            )
            start_port = port_args.nccl_port
            
            engine_launch_args.append((
                server_args,
                port_args,
                [gpu_id],
                worker_id,
                path_to_shared_cpu_models,
                model_names_to_model_paths,
                engine_id,
            ))

    # Launch engines in parallel
    with ThreadPoolExecutor(max_workers=len(engine_launch_args)) as executor:
        all_results = list(executor.map(launch_wrapper, engine_launch_args))

    # Process results
    for gpu_ids, engine_info in all_results:
        gpu_id = gpu_ids[0]
        engine_info_dict[gpu_id].append(engine_info)
        port_args_dict[gpu_id].append(engine_info.port_args)
        num_engines += 1
        gpu_id_to_model_instance[gpu_id][gpu_id] = engine_info.instance_idx

    logger.info(
        f"All {num_engines} worker pool engines prepared in {time.perf_counter() - start_time:.2f} seconds."
    )
    return engine_info_dict, port_args_dict, gpu_id_to_model_instance, num_engines, init_placements


def _launch_controller(
    multi_model_server_args: MultiModelServerArgs,
    recv_from_request_handler_ipc_name: str,
    recv_from_schedulers_ipc_name: str,
    engine_info_dict: Dict,
    model_names_to_model_paths: Dict[str, str],
    init_placements: Optional[Dict] = None,
):
    """Launch the global controller process."""
    from prism.multi_model.scheduling.controller import run_controller_process
    
    controller_proc = mp.Process(
        target=run_controller_process,
        args=(
            multi_model_server_args,
            recv_from_request_handler_ipc_name,
            recv_from_schedulers_ipc_name,
            engine_info_dict,
            model_names_to_model_paths,
            init_placements,
        ),
    )
    controller_proc.start()
    logger.info("Controller process started.")


def _launch_gpu_scheduler(
    multi_model_server_args: MultiModelServerArgs,
    model_names_to_model_paths: Dict[str, str],
    engine_info_dict: Dict,
    gpu_id: int,
    init_model_names: List[str],
):
    """Launch a GPU scheduler process."""
    from prism.multi_model.scheduling.gpu.scheduler import run_gpu_scheduler_process
    
    reader, writer = mp.Pipe(duplex=False)
    gpu_scheduler_proc = mp.Process(
        target=run_gpu_scheduler_process,
        args=(
            multi_model_server_args,
            engine_info_dict,
            model_names_to_model_paths,
            gpu_id,
            init_model_names,
            writer,
        ),
    )
    gpu_scheduler_proc.start()
    reader.recv()
    logger.info(f"GPU scheduler process started for GPU {gpu_id}")


def _wait_and_warmup(url: str, model_name: str, pipe_finish_writer, pid: int):
    """Wait for server to be ready and send warmup request."""
    logger.info(f"Waiting for the server to be ready for model {model_name}...")

    # Wait until the server is launched
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(f"{url}/get_model_names", timeout=5)
            assert res.status_code == 200
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            pass

    if not success:
        logger.error(f"Server not ready for {model_name}")
        if pipe_finish_writer is not None:
            pipe_finish_writer.send("timeout")
        kill_child_process(pid, including_parent=False)
        return

    # Send a warmup request
    json_data = {
        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
        "model": model_name,
        "is_warmup": True,
        "text": "The capital city of France is",
    }

    try:
        res = requests.post(f"{url}/generate", json=json_data, timeout=30000)
        assert res.status_code == 200
    except Exception as e:
        logger.error(f"Warmup failed for {model_name}: {e}")
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(str(e))
        kill_child_process(pid, including_parent=False)
        return

    logger.info(f"Server for {model_name} is ready!")
    if pipe_finish_writer is not None:
        pipe_finish_writer.send("ready")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    import sys
    from prism.multi_model.server_args import prepare_server_args
    
    args = prepare_server_args(sys.argv[1:])
    launch_multi_model_server(args)


if __name__ == "__main__":
    main()
