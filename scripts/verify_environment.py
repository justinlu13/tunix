#!/usr/bin/env python3
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

"""Reusable Python environment and TPU hardware preflight verifier for Tunix.

Replaces inline `python3 -c "..."` snippets in `Dockerfile` and CI container
builds.
"""

import argparse
import importlib
import sys
import traceback


def check_module(module_name: str, check_tunix_version: bool = False) -> bool:
  """Imports a module and optionally asserts a non-dev Tunix version."""
  try:
    mod = importlib.import_module(module_name)
    version = getattr(mod, "__version__", "installed")
    print(f"  [OK] {module_name:<30} - v{version}")
    if check_tunix_version and module_name == "tunix":
      if version == "0.0.0.dev0":
        print(
            "  [FAIL] Tunix version is '0.0.0.dev0' (package metadata not set)."
        )
        return False
    return True
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"  [FAIL] {module_name:<30} - ERROR: {exc}")
    traceback.print_exc(file=sys.stdout)
    return False


def verify_jax_backend(
    require_tpu: bool, expected_tpu_devices: int | None
) -> bool:
  """Validates JAX runtime and optional TPU device count."""
  try:
    import jax  # pylint: disable=g-import-not-at-top

    backend = jax.default_backend()
    devices = jax.devices()
    print(f"\n  [OK] JAX version:   {jax.__version__}")
    print(f"  [OK] JAX backend:   '{backend}'")
    print(f"  [OK] JAX devices:   {len(devices)} device(s) ({devices})")

    if require_tpu:
      has_tpu = bool(devices) and all(d.platform == "tpu" for d in devices)
      if not has_tpu:
        print(
            "  [FAIL] Expected all JAX devices to be on 'tpu' platform, got:"
            f" {[d.platform for d in devices]}"
        )
        return False

    if (
        expected_tpu_devices is not None
        and len(devices) != expected_tpu_devices
    ):
      print(
          f"  [FAIL] Expected {expected_tpu_devices} TPU device(s), got"
          f" {len(devices)}: {devices}"
      )
      return False

    return True
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"\n  [FAIL] JAX backend check failed: {exc}")
    traceback.print_exc(file=sys.stdout)
    return False


def load_packages_from_file(file_path: str) -> list[str]:
  packages = []
  with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line and not line.startswith("#"):
        packages.append(line)
  return packages


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Verify Python module imports and JAX/TPU runtime state."
  )
  parser.add_argument(
      "--packages",
      nargs="*",
      default=[],
      help="List of Python modules to import and verify.",
  )
  parser.add_argument(
      "--packages-file",
      default="",
      help="Path to a newline-delimited file of module names to import.",
  )
  parser.add_argument(
      "--check-tunix-version",
      action="store_true",
      help="Assert that tunix.__version__ is not '0.0.0.dev0'.",
  )
  parser.add_argument(
      "--require-tpu",
      action="store_true",
      help="Assert that all discovered JAX devices have platform == 'tpu'.",
  )
  parser.add_argument(
      "--expect-tpu-devices",
      type=int,
      default=None,
      help="Assert the exact number of discovered JAX TPU devices (e.g. 8).",
  )
  args = parser.parse_args()

  packages = list(args.packages)
  if args.packages_file:
    packages.extend(load_packages_from_file(args.packages_file))

  print("\n========================================================")
  print(" Tunix Environment & Hardware Preflight Verification")
  print(f" Python runtime: {sys.version.split()[0]}")
  print("========================================================\n")

  failed = []
  for pkg in packages:
    if not check_module(pkg, check_tunix_version=args.check_tunix_version):
      failed.append(pkg)

  backend_ok = True
  if (
      "jax" in packages
      or args.require_tpu
      or args.expect_tpu_devices is not None
  ):
    backend_ok = verify_jax_backend(
        require_tpu=args.require_tpu,
        expected_tpu_devices=args.expect_tpu_devices,
    )

  if failed or not backend_ok:
    print(f"\nVerification FAILED (failed modules: {failed}).")
    sys.exit(1)

  print("\nAll requested modules and runtime checks passed.")


if __name__ == "__main__":
  main()
