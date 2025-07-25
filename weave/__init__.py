"""The top-level functions and classes for working with Weave."""

from weave import version
from weave.trace.api import *

__version__ = version.VERSION


from weave.flow.agent import Agent as Agent
from weave.flow.agent import AgentState as AgentState
from weave.flow.annotation_spec import AnnotationSpec
from weave.flow.dataset import Dataset
from weave.flow.eval import Evaluation
from weave.flow.eval_imperative import EvaluationLogger
from weave.flow.model import Model
from weave.flow.monitor import Monitor
from weave.flow.obj import Object
from weave.flow.prompt.prompt import EasyPrompt, MessagesPrompt, Prompt, StringPrompt
from weave.flow.saved_view import SavedView
from weave.flow.scorer import Scorer
from weave.initialization import *
from weave.trace.util import Thread as Thread
from weave.trace.util import ThreadPoolExecutor as ThreadPoolExecutor
from weave.type_handlers.Audio.audio import Audio
from weave.type_handlers.File.file import File
from weave.type_handlers.Markdown.markdown import Markdown
from weave.type_wrappers import Content

# Alias for succinct code
P = EasyPrompt

# Special object informing doc generation tooling which symbols
# to document & to associate with this module.
__docspec__ = [
    # Re-exported from trace.api
    init,
    publish,
    ref,
    get,
    require_current_call,
    get_current_call,
    finish,
    op,
    attributes,
    thread,
    # Re-exported from flow module
    Object,
    Dataset,
    Model,
    Prompt,
    StringPrompt,
    MessagesPrompt,
    Evaluation,
    EvaluationLogger,
    Scorer,
    AnnotationSpec,
    File,
    Content,
    Markdown,
    Monitor,
    SavedView,
    Audio,
]

__all__ = [
    "Agent",
    "AgentState",
    "AnnotationSpec",
    "Audio",
    "Content",
    "Dataset",
    "EasyPrompt",
    "Evaluation",
    "EvaluationLogger",
    "File",
    "Markdown",
    "MessagesPrompt",
    "Model",
    "Monitor",
    "Object",
    "Prompt",
    "SavedView",
    "Scorer",
    "StringPrompt",
    "attributes",
    "finish",
    "get",
    "get_current_call",
    "init",
    "op",
    "publish",
    "ref",
    "require_current_call",
    "thread",
]
