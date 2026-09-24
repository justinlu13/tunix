# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests, including real distributed DispatchTask/PollResponses RPC.

Run directly with absl-py, cloudpickle, grpcio, fsspec and PyYAML installed.
The model stack (JAX/vLLM/MaxText) is not needed for these contract tests.
"""

import asyncio
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
RECIPE = ROOT / "tunix/experimental/examples/deepswe_dist"


def load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


eval_lib = load("deepswe_eval_under_test", RECIPE / "eval_deepswe.py")


class EvalTest(unittest.TestCase):

  def args(self, *extra):
    return eval_lib.parse_args(
        ["--model_absolute_path", "gs://models/0/items", *extra]
    )

  def entry(self):
    return {
        "instance_id": "repo/issue",
        "problem_statement": "fix bug",
        "docker_image": "image",
    }

  def test_checkpoint_is_required_and_never_dummy(self):
    with self.assertRaises(SystemExit):
      eval_lib.parse_args([])
    config = eval_lib.maxtext_config(self.args())
    self.assertEqual(config["load_parameters_path"], "gs://models/0/items")
    self.assertFalse(config["scan_layers"])
    self.assertTrue(config["checkpoint_storage_use_ocdbt"])
    self.assertFalse(config["checkpoint_storage_use_zarr3"])

  def test_request_limits_and_engine_seed(self):
    a = self.args("--seed", "0")
    first = eval_lib.request_fields(a, self.entry(), 0, 0)
    second = eval_lib.request_fields(a, self.entry(), 0, 1)
    self.assertNotIn("seed", first["generation_kwargs"])
    self.assertNotIn("seed", second["generation_kwargs"])
    self.assertEqual(eval_lib.model_profile(a)["seed"], 0)
    self.assertNotEqual(first["request_id"], second["request_id"])
    self.assertIsNone(first["max_response_length"])
    self.assertEqual(first["generation_kwargs"]["max_generation_steps"], 12288)
    with_limits = self.args(
        "--max_context_limit",
        "61440",
        "--top_k",
        "-1",
        "--batch_size",
        "16",
    )
    req = eval_lib.request_fields(with_limits, self.entry(), 0, 0)
    self.assertEqual(req["max_response_length"], 61440)
    self.assertIsNone(req["generation_kwargs"]["top_k"])
    self.assertEqual(with_limits.batch_size, 16)

  def test_limited_remote_dataset_streams_and_uses_image_identity(self):
    entry = self.entry()
    del entry["instance_id"]
    stream = mock.Mock()
    stream.take.return_value = iter([entry])
    loader = mock.Mock(return_value=stream)
    deepswe = types.SimpleNamespace(_jsonify_lists=dict)
    with mock.patch.dict(
        sys.modules,
        {
            "datasets": types.SimpleNamespace(load_dataset=loader),
            "tunix.experimental.examples.deepswe_dist": types.SimpleNamespace(
                deepswe=deepswe
            ),
        },
    ):
      a = self.args("--tasks_limit", "1")
      entries = eval_lib.load_entries(a)
    loader.assert_called_once_with(
        a.dataset_name, split=a.dataset_split, streaming=True
    )
    stream.take.assert_called_once_with(1)
    self.assertEqual(entries[0]["instance_id"], "image")

  def test_local_dataset_preserves_ids_and_rejects_duplicate_image_ids(self):
    entry = self.entry()
    loader = mock.Mock(return_value=[entry])
    deepswe = types.SimpleNamespace(load_deepswe_dataset=loader)
    with mock.patch.dict(
        sys.modules,
        {
            "tunix.experimental.examples.deepswe_dist": types.SimpleNamespace(
                deepswe=deepswe
            ),
        },
    ):
      a = self.args("--dataset_path", "/saved/tasks")
      self.assertEqual(eval_lib.load_entries(a), [entry])
      without_id = {k: v for k, v in entry.items() if k != "instance_id"}
      loader.return_value = [without_id, without_id]
      with self.assertRaisesRegex(ValueError, "unique instance_id"):
        eval_lib.load_entries(a)

  def test_missing_and_failed_attempts_are_not_dropped(self):
    rows = [
        dict(
            instance_id="a",
            attempt=0,
            reward=1,
            resolved=True,
            status="SUCCEEDED",
        ),
        dict(
            instance_id="a",
            attempt=1,
            reward=0,
            resolved=False,
            status="ERROR",
            error="pod failed",
        ),
        dict(
            instance_id="b",
            attempt=0,
            reward=0,
            resolved=False,
            status="MAX_STEPS_REACHED",
        ),
    ]
    summary = eval_lib.summarize(rows, ["a", "b"], 4)
    self.assertEqual(summary["expected_attempts"], 8)
    self.assertEqual(summary["missing_attempts"], 5)
    self.assertEqual(summary["error_attempts"], 1)
    self.assertEqual(summary["avg_at_k"], 1 / 8)
    self.assertEqual(summary["pass_at_k"], {"1": 1 / 8, "4": 1 / 2})
    self.assertFalse(summary["complete"])
    with self.assertRaisesRegex(ValueError, "duplicate"):
      eval_lib.summarize(rows + [rows[0]], ["a", "b"], 4)

  def test_reward_comes_from_trajectory_not_completion_status(self):
    response = types.SimpleNamespace(
        error=None,
        status="COMPLETED",
        payload=types.SimpleNamespace(
            traj={"trajectory_reward": 0, "status": "SUCCEEDED"}
        ),
    )
    self.assertFalse(eval_lib.compact_result(response)["resolved"])
    response.payload.traj["trajectory_reward"] = 1
    self.assertTrue(eval_lib.compact_result(response)["resolved"])
    response.error = "infrastructure failure"
    self.assertFalse(eval_lib.compact_result(response)["resolved"])

  def test_individual_results_are_persisted(self):
    with tempfile.TemporaryDirectory() as directory:
      writer = eval_lib.ResultWriter(directory)
      row = dict(request_id="eval_0_0", instance_id="../unsafe-id", reward=0)
      writer.record(row)
      self.assertEqual(
          json.loads((Path(directory) / "attempts/eval_0_0.json").read_text()),
          row,
      )


class RpcTest(unittest.IsolatedAsyncioTestCase):

  async def test_real_grpc_out_of_order_bounded_dispatch_and_failure(self):
    remote = load(
        "eval_remote_execution_under_test",
        ROOT / "tunix/experimental/worker/remote_execution.py",
    )

    class Worker:
      active = 0
      peak = 0

      async def evaluate(self, fields):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
          attempt = fields["group_index"]
          await asyncio.sleep(0.03 if attempt == 0 else 0.001)
          if attempt == 2:
            raise RuntimeError("sandbox failed")
          return dict(reward=1, resolved=True, status="SUCCEEDED", error=None)
        finally:
          self.active -= 1

    worker = Worker()
    server = remote.GrpcRemoteExecutionServer(worker)
    with socket.socket() as reservation:
      reservation.bind(("127.0.0.1", 0))
      port = reservation.getsockname()[1]
    await server.start_serving_async(port)
    handle = remote.ActorHandle.from_address(f"grpc://127.0.0.1:{port}")
    a = eval_lib.parse_args(["--model_absolute_path", "gs://models/0/items"])
    entry = {
        "instance_id": "a",
        "problem_statement": "fix",
        "docker_image": "img",
    }
    jobs = iter(eval_lib.request_fields(a, entry, 0, i) for i in range(4))
    written = []
    try:
      rows = await eval_lib.evaluate_worker(handle, jobs, 2, 10, written.append)
    finally:
      await handle.close()
      await server.stop_serving()
    self.assertEqual(len(rows), 4)
    self.assertEqual(len(written), 4)
    self.assertLessEqual(worker.peak, 2)
    self.assertEqual(rows[0]["attempt"], 1)
    summary = eval_lib.summarize(rows, ["a"], 4)
    self.assertEqual(summary["error_attempts"], 1)
    self.assertEqual(summary["avg_at_k"], 0.75)

  async def test_dispatch_error_and_unknown_response_fail_fast(self):
    a = eval_lib.parse_args(["--model_absolute_path", "gs://models/0/items"])
    entry = {
        "instance_id": "a",
        "problem_statement": "fix",
        "docker_image": "img",
    }
    fields = eval_lib.request_fields(a, entry, 0, 0)
    handle = types.SimpleNamespace(
        dispatch_task=mock.AsyncMock(return_value="wrong-ack"),
        poll_responses=mock.AsyncMock(),
    )
    with self.assertRaisesRegex(RuntimeError, "acknowledgement"):
      await eval_lib.evaluate_worker(
          handle, iter([fields]), 1, 10, lambda r: None
      )
    handle.dispatch_task.return_value = fields["request_id"]
    handle.poll_responses.return_value = types.SimpleNamespace(
        request_id="stale"
    )
    with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
      await eval_lib.evaluate_worker(
          handle, iter([fields]), 1, 10, lambda r: None
      )

  async def test_missing_reply_has_deadline(self):
    a = eval_lib.parse_args(["--model_absolute_path", "gs://models/0/items"])
    fields = eval_lib.request_fields(
        a, {"instance_id": "a", "problem_statement": "p"}, 0, 0
    )
    handle = types.SimpleNamespace(
        dispatch_task=mock.AsyncMock(return_value=fields["request_id"]),
        poll_responses=mock.AsyncMock(return_value=None),
    )
    with mock.patch.object(eval_lib.time, "monotonic", side_effect=[0, 20]):
      with self.assertRaises(TimeoutError):
        await eval_lib.evaluate_worker(
            handle, iter([fields]), 1, 10, lambda r: None
        )


mllog_lib = load("mllog_utils_under_test", ROOT / "tunix/utils/mllog_utils.py")


class OfflineEvalRcpTest(unittest.IsolatedAsyncioTestCase):

  def test_compute_val_start_step(self):
    self.assertEqual(mllog_lib.compute_val_start_step(256), 18)
    self.assertEqual(mllog_lib.compute_val_start_step(128), 33)
    self.assertEqual(mllog_lib.compute_val_start_step(256, 5), 5)
    with self.assertRaises(ValueError):
      mllog_lib.compute_val_start_step(0)

  def test_checkpoint_manifest_read_write_and_contiguity(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      manifest_path = str(Path(tmpdir) / "eval_checkpoints.jsonl")
      mllog_lib.append_checkpoint_manifest(
          manifest_path,
          {
              "step": 19,
              "samples_count": 4864,
              "checkpoint_timestamp_ms": 2000,
              "checkpoint_path": "gs://ckpt/19/model_params",
              "val_start_at": 18,
              "max_steps": 50,
          },
      )
      mllog_lib.append_checkpoint_manifest(
          manifest_path,
          {
              "step": 18,
              "samples_count": 4608,
              "checkpoint_timestamp_ms": 1000,
              "checkpoint_path": "gs://ckpt/18/model_params",
              "val_start_at": 18,
              "max_steps": 50,
          },
      )
      records = mllog_lib.load_checkpoint_manifest(
          manifest_path, check_contiguous=True
      )
      self.assertEqual([r["step"] for r in records], [18, 19])

      # Non-contiguous step should raise ValueError
      mllog_lib.append_checkpoint_manifest(
          manifest_path,
          {
              "step": 21,
              "samples_count": 5376,
              "checkpoint_timestamp_ms": 4000,
              "checkpoint_path": "gs://ckpt/21/model_params",
              "val_start_at": 18,
              "max_steps": 50,
          },
      )
      with self.assertRaisesRegex(ValueError, "Non-contiguous"):
        mllog_lib.load_checkpoint_manifest(
            manifest_path, check_contiguous=True
        )

  def test_summarize_includes_pass_at_4(self):
    rows = [
        {"instance_id": "inst_1", "attempt": i, "resolved": i == 0, "reward": 1.0 if i == 0 else 0.0, "status": "ok"}
        for i in range(4)
    ]
    summary = eval_lib.summarize(rows, ["inst_1"], 4)
    self.assertIn("4", summary["pass_at_k"])
    self.assertAlmostEqual(summary["pass_at_k"]["4"], 1.0)
    self.assertAlmostEqual(summary["mean_reward"], 0.25)

  def test_offline_eval_rcp_sequence_converged_and_aborted(self):
    fake_mllogger = mock.MagicMock()
    with (
        mock.patch.object(mllog_lib, "mllogger", fake_mllogger),
        mock.patch.object(mllog_lib, "_is_master_process", return_value=True),
        mock.patch.object(mllog_lib, "_flush_to_gcs_if_needed"),
    ):
      args = types.SimpleNamespace(
          batch_size=16, num_generations=16, max_steps=50
      )
      mllog_lib.train_stop(args, step=50, time_ms=5000)
      fake_mllogger.end.assert_called_once_with(
          key="block_stop",
          metadata={"step": 50, "samples_count": 12800},
          time_ms=5000,
      )
      fake_mllogger.reset_mock()

      # Step 18: accuracy 0.68 < 0.69 -> no run_stop
      passed_18 = mllog_lib.log_offline_eval_step(
          step=18,
          samples_count=4608,
          eval_accuracy=0.68,
          target_accuracy=0.69,
          checkpoint_timestamp_ms=1000,
          is_last_checkpoint=False,
          validation_time=12.5,
      )
      self.assertFalse(passed_18)
      end_keys_18 = [c.kwargs["key"] for c in fake_mllogger.end.call_args_list]
      self.assertEqual(end_keys_18, ["eval_stop"])
      fake_mllogger.reset_mock()

      # Step 19: accuracy 0.71 >= 0.69 -> backdated run_stop(time_ms=2000, status="success") + train_samples=4864
      passed_19 = mllog_lib.log_offline_eval_step(
          step=19,
          samples_count=4864,
          eval_accuracy=0.71,
          target_accuracy=0.69,
          checkpoint_timestamp_ms=2000,
          is_last_checkpoint=False,
          validation_time=14.0,
      )
      self.assertTrue(passed_19)
      fake_mllogger.end.assert_any_call(
          key="run_stop",
          metadata={"status": "success", "samples_count": 4864},
          time_ms=2000,
      )
      fake_mllogger.event.assert_any_call(
          key="train_samples",
          value=4864,
      )
      fake_mllogger.reset_mock()

      # Step 50 (final): accuracy 0.68 < 0.69 -> backdated run_stop(time_ms=5000, status="aborted")
      passed_50 = mllog_lib.log_offline_eval_step(
          step=50,
          samples_count=12800,
          eval_accuracy=0.68,
          target_accuracy=0.69,
          checkpoint_timestamp_ms=5000,
          is_last_checkpoint=True,
          validation_time=15.0,
      )
      self.assertFalse(passed_50)
      fake_mllogger.end.assert_any_call(
          key="run_stop",
          metadata={"status": "aborted", "samples_count": 12800},
          time_ms=5000,
      )

  async def test_controller_manifest_loop_stops_on_convergence(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      manifest_path = str(Path(tmpdir) / "eval_checkpoints.jsonl")
      for step, ts_ms in ((18, 1000), (19, 2000), (20, 3000)):
        mllog_lib.append_checkpoint_manifest(
            manifest_path,
            {
                "step": step,
                "samples_count": step * 256,
                "checkpoint_timestamp_ms": ts_ms,
                "checkpoint_path": f"gs://ckpt/{step}/model_params",
                "val_start_at": 18,
                "max_steps": 20,
            },
        )

      evaluated_steps = []

      async def fake_single_controller(step_args):
        evaluated_steps.append(step_args.checkpoint_step)
        reached = step_args.checkpoint_step == 19
        return {
            "checkpoint_step": step_args.checkpoint_step,
            "checkpoint_timestamp_ms": step_args.checkpoint_timestamp_ms,
            "samples_count": step_args.samples_count,
            "target_reached": reached,
            "rcp_logged": True,
        }

      a = eval_lib.parse_args([
          "--checkpoint_manifest_file",
          manifest_path,
          "--output_dir",
          str(Path(tmpdir) / "results"),
      ])
      with (
          mock.patch.dict(
              sys.modules,
              {"tunix.utils": types.SimpleNamespace(mllog_utils=mllog_lib)},
          ),
          mock.patch.object(
              eval_lib, "_run_single_controller", side_effect=fake_single_controller
          ),
      ):
        summary = await eval_lib.run_controller(a)

      self.assertEqual(evaluated_steps, [18, 19])
      self.assertTrue(summary["target_reached"])
      self.assertEqual(summary["checkpoint_step"], 19)
      self.assertEqual(summary["checkpoint_timestamp_ms"], 2000)


if __name__ == "__main__":
  unittest.main()
