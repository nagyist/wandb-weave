import datetime
import json
import logging
import os
import typing

import wandb
from wandb import errors as wandb_errors
from wandb.apis.public import Run
from wandb.sdk.artifacts.artifact_saver import ArtifactSaver
from wandb.sdk.interface.interface import InterfaceBase
from wandb.sdk.internal import file_stream
from wandb.sdk.internal.file_pusher import FilePusher
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.internal.sender import _manifest_json_from_proto
from wandb.sdk.lib import runid

from weave_query import engine_trace, errors, wandb_client_api

logger = logging.getLogger(__name__)


# We disable urllib warnings because they are noisy and not actionable.
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

WANDB_HIDDEN_JOB_TYPE = "sweep-controller"


class InMemoryLazyLiteRun:
    # ID
    _entity_name: str
    _project_name: str
    _run_name: str
    _step: int = 0

    # Optional
    _display_name: typing.Optional[str] = None
    _job_type: typing.Optional[str] = None
    _group: typing.Optional[str] = None

    # Property Cache
    _i_api: typing.Optional[InternalApi] = None
    _run: typing.Optional[Run] = None
    _stream: typing.Optional[file_stream.FileStreamApi] = None
    _pusher: typing.Optional[FilePusher] = None
    _use_async_file_stream: bool = False

    def __init__(
        self,
        entity_name: str,
        project_name: str,
        run_name: typing.Optional[str] = None,
        *,
        job_type: typing.Optional[str] = None,
        group: typing.Optional[str] = None,
        _hide_in_wb: bool = False,
        _use_async_file_stream: bool = False,
    ):
        wandb_client_api.assert_wandb_authenticated()

        # Technically, we could use the default entity and project here, but
        # heeding Shawn's advice, we should be explicit about what we're doing.
        # We can always move to the default later, but we can't go back.
        if entity_name is None or entity_name == "":
            raise ValueError("Must specify entity_name")
        elif project_name is None or project_name == "":
            raise ValueError("Must specify project_name")

        self._entity_name = entity_name
        self._project_name = project_name
        self._display_name = run_name
        self._run_name = run_name or runid.generate_id()
        self._job_type = job_type if not _hide_in_wb else WANDB_HIDDEN_JOB_TYPE
        if _hide_in_wb and group is None:
            group = "weave_hidden_runs"
        self._group = group

        self._use_async_file_stream = (
            _use_async_file_stream
            and os.getenv("WEAVE_DISABLE_ASYNC_FILE_STREAM") == None
        )

        self._project_created = False
        self._is_log_setup = False

    def ensure_run(self) -> Run:
        return self.run

    @property
    def i_api(self) -> InternalApi:
        if self._i_api is None:
            self._i_api = InternalApi(
                {"project": self._project_name, "entity": self._entity_name}
            )
        return self._i_api

    @property
    def run(self) -> Run:
        if self._run is None:
            try:
                # Ensure project exists
                self.upsert_project()

                # Produce a run
                run_res, _, _ = self.i_api.upsert_run(
                    name=self._run_name,
                    display_name=self._display_name,
                    job_type=self._job_type,
                    group=self._group,
                    project=self._project_name,
                    entity=self._entity_name,
                )

                self._run = Run(
                    wandb_client_api.wandb_public_api().client,
                    run_res["project"]["entity"]["name"],
                    run_res["project"]["name"],
                    run_res["name"],
                    {
                        "id": run_res["id"],
                        "config": "{}",
                        "systemMetrics": "{}",
                        "summaryMetrics": "{}",
                        "tags": [],
                        "description": None,
                        "notes": None,
                        "state": "running",
                    },
                )
            except wandb_errors.CommError as e:
                raise errors.WeaveWandbAuthenticationException()

            self.i_api.set_current_run_id(self._run.id)

        return self._run

    @property
    def stream(self) -> file_stream.FileStreamApi:
        if self._stream is None:
            # Setup the FileStream
            self._stream = file_stream.FileStreamApi(
                self.i_api, self.run.id, datetime.datetime.utcnow().timestamp()
            )
            if self._use_async_file_stream:
                self._stream._client.headers.update(
                    {"X-WANDB-USE-ASYNC-FILESTREAM": "true"}
                )
            self._stream.start()

        return self._stream

    @property
    def pusher(self) -> FilePusher:
        if self._pusher is None:
            self._pusher = FilePusher(self.i_api, self.stream)

        return self._pusher

    def setup_log_deps(self) -> None:
        if not self._is_log_setup:
            self._step = self.run.lastHistoryStep + 1
            self.stream.set_file_policy(
                "wandb-history.jsonl",
                file_stream.JsonlFilePolicy(start_chunk_id=self._step),
            )
            self._is_log_setup = True

    def log(self, row_dict: dict) -> None:
        self.setup_log_deps()
        stream = self.stream
        row_dict = {
            **row_dict,
            "_timestamp": datetime.datetime.utcnow().timestamp(),
        }
        if not self._use_async_file_stream:
            row_dict["_step"] = self._step
        self._step += 1
        stream.push("wandb-history.jsonl", json.dumps(row_dict))

    def log_artifact(
        self,
        artifact: wandb.Artifact,
        additional_aliases: list = [],
        artifact_collection_exists: bool = False,
    ) -> typing.Optional[typing.Dict]:
        tracer = engine_trace.tracer()  # type: ignore
        artifact_name = artifact.name
        artifact_type_name = artifact.type

        assert artifact_name is not None
        assert artifact_type_name is not None

        wandb_client_api.assert_wandb_authenticated()

        if not artifact_collection_exists:
            self.upsert_project()
            self.i_api.create_artifact_type(
                artifact_type_name=artifact_type_name,
                entity_name=self._entity_name,
                project_name=self._project_name,
            )

        # Finalize the artifact and construct the manifest.
        artifact.finalize()
        manifest_dict = _manifest_json_from_proto(
            InterfaceBase()._make_artifact(artifact).manifest  # type: ignore[abstract, attr-defined]
        )

        # Save the Artifact and the associated files
        with tracer.trace("Instantiate ArtifactSaver"):
            saver = ArtifactSaver(
                api=self.i_api,
                digest=artifact.digest,
                manifest_json=manifest_dict,
                file_pusher=self.pusher,
                is_user_created=False,
            )

        with tracer.trace("ArtifactSaver.save"):
            res = saver.save(
                entity=self._entity_name,
                project=self._project_name,
                type=artifact_type_name,
                name=artifact_name,
                client_id=artifact._client_id,
                sequence_client_id=artifact._sequence_client_id,
                metadata=artifact.metadata,
                description=artifact.description,
                aliases=["latest"] + additional_aliases,
                use_after_commit=False,
            )  # type: ignore
        return res

    def upsert_project(self) -> None:
        if not self._project_created:
            self.i_api.upsert_project(
                project=self._project_name, entity=self._entity_name
            )
            self._project_created = True

    def finish(self) -> None:
        if self._stream is not None:
            # Finalize the run
            self.stream.finish(0)

        if self._pusher is not None:
            # Wait for the FilePusher and FileStream to finish
            self.pusher.finish()
            self.pusher.join()

        # Reset fields
        self._stream = None
        self._pusher = None
        self._run = None
        self._i_api = None
        self._step = 0

    def __del__(self) -> None:
        self.finish()
