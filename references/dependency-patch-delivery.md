# Dependency patch delivery

Read this reference when the executed path requires changes outside the main
project repository, especially in editable checkouts, Python `site-packages`,
framework forks, or accelerator extensions. The deliverable is a reproducible
delta, not a copy of the target machine.

## Classify each changed dependency

Record one row per library before editing:

| Kind | Preferred treatment | Do not publish |
|---|---|---|
| Open-source Git checkout | Pin URL and revision; create a scoped patch | Whole unrelated repository history |
| Installed Python package | Pin distribution/version and base-file hashes; patch a clean source copy or use an overlay | Complete `site-packages` tree |
| Project-owned vendored source | Patch inside the project when its license and ownership allow | Generated build/cache files |
| Binary extension or wheel | Pin official artifact/version/hash or provide a build recipe through an authorized channel | Repacked third-party binary without redistribution rights |
| CANN, driver, firmware, TorchNPU runtime | Use the official compatibility/install source and record the observed tuple | Runtime installation trees or device drivers |
| Model/data/cache artifact | Record local path, interface, size, and hash when available | Weights, datasets, caches, credentials, or signed URLs |

Check the upstream license before redistributing source excerpts, patches, or
binaries. If redistribution is unclear, keep only version/hash metadata and a
local reconstruction command; use an authorized private artifact channel when
the target has no access to the original source. This toolkit does not grant
permission to publish third-party material.

## Minimize the patch surface

Do not treat every static scanner match as a file to modify. First trace the
requested entrypoint, then prefer in this order:

1. configure an existing NPU or eager fallback;
2. add a project-owned adapter or overlay;
3. patch a small set of upstream source files;
4. maintain a library fork only when the patch cannot be expressed or applied
   reliably as a delta.

Never edit a live environment first and reconstruct the change from memory.
Create the change against a clean copy of the exact source/version, test it,
then apply the same patch to the target. If emergency diagnosis changed
`site-packages`, diff it immediately against a clean installation and discard
unrelated generated files.

## Registry contract

For a multi-library port, place `dependency-patches.json` beside `patches/`.
Use paths relative to the bundle root. A minimal registry is:

```json
{
  "schema_version": 1,
  "project": "example",
  "libraries": [
    {
      "name": "diffsynth",
      "source": "UPSTREAM_SOURCE_URL",
      "base_revision": "immutable-commit-or-package-version",
      "license_reference": "upstream LICENSE URL or SPDX identifier",
      "target_kind": "source-checkout",
      "base_files": [
        {
          "path": "package/device.py",
          "sha256": "<64-lowercase-hex>"
        }
      ],
      "patches": [
        {
          "path": "patches/diffsynth/0001-device-routing.patch",
          "size_bytes": 1234,
          "sha256": "<64-lowercase-hex>"
        }
      ],
      "apply": "git apply --check ... && git apply ...",
      "revert": "git apply --check -R ... && git apply -R ...",
      "validation_commands": [
        "python3 -m pytest tests/test_device_routing.py"
      ]
    }
  ]
}
```

Allowed `target_kind` values are `source-checkout`, `installed-python`, and
`overlay-only`. `apply`, `revert`, and `validation_commands` are documentation,
not commands executed by the validator.

Validate bundle files and, when available, pristine dependency sources:

```bash
python3 scripts/validate_patch_registry.py dependency-patches.json \
  --bundle-root .

python3 scripts/validate_patch_registry.py dependency-patches.json \
  --bundle-root . \
  --source diffsynth=/absolute/path/to/pristine/diffsynth \
  --require-base
```

The second command must emit `ASCEND_PATCH_REGISTRY_VALID` before applying a
patch to the named source. For multiple libraries, pass one `--source` per
library. The validator checks registry structure, patch size/hash, safe relative
paths, and optional pristine base-file hashes; it deliberately does not execute
the recorded shell commands.

## Apply and rollback protocol

For each library in registry order:

1. verify the installed distribution, source URL, immutable revision, and
   base-file hashes;
2. stop on a dirty checkout or hash mismatch instead of forcing the patch;
3. run the recorded dry-run command, such as `git apply --check`;
4. apply only that library's patch set;
5. run syntax/import and the narrow component gate before continuing;
6. record post-patch file hashes and the exact active import path;
7. prove rollback in a disposable copy, not by destructively resetting the
   target workspace.

After all libraries are patched, run an import-path probe from the real training
or serving environment. A successful patch against one checkout is irrelevant
if Python imports a different installed copy.

## GitHub versus private handoff

A public repository should normally contain only the skill, patch registry,
small text patches, project-owned overlays, launch/config files, tests, and
documentation. Exclude complete dependency trees, `.git` metadata, virtual
environments, build products, binaries, weights, datasets, caches, secrets, and
machine-specific logs.

When a disconnected target lacks the pristine dependency source, keep the
public patch bundle unchanged and deliver the allowed source or wheel through a
separate approved channel. Record its hash in the private handoff; do not make
the public repository the transport mechanism for restricted or large assets.
