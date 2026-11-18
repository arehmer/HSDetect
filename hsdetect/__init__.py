# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:21:29 2023

@author: Rehmer
"""

import ast
import enum
import importlib
import pkgutil
from functools import lru_cache


class Payload(enum.Enum):
    """What one engine hands the next in an InferenceEngine chain.

    The chain is linear and every engine declares what it takes and what it
    produces, so a chain that cannot work is rejected when it is built rather
    than failing, or quietly doing nothing, once frames are flowing.

    The frame members distinguish states of the same array, because the
    difference matters and is not visible from the shape alone:

    FRAME_RAW
        (H, W) absolute sensor values as the sensor delivers them, dead
        corners included -- around 3030..3055 dK where there is no optics.
    FRAME_MASKED
        (H, W) with those corners set to 0, i.e. after ZeroCorners. Anything
        estimating a level over the frame needs this and not FRAME_RAW, since
        the corners sit above the scene and would drag the estimate up.
    FRAME_STACKED
        (H, W, 2) float32: channel 0 the background-subtracted frame,
        channel 1 the frame that entered the subtraction. What a two-channel
        model is trained on.
    TENSOR
        The network's own output, (1, cells * anchors, 5 + n_classes).
    BOXES
        A DataFrame of detections.
    """

    FRAME_RAW = 'frame_raw'
    FRAME_MASKED = 'frame_masked'
    FRAME_STACKED = 'frame_stacked'
    TENSOR = 'tensor'
    BOXES = 'boxes'

    @property
    def is_frame(self) -> bool:
        return self in _FRAME_PAYLOADS

    def __str__(self):
        return self.value


_FRAME_PAYLOADS = frozenset({Payload.FRAME_RAW,
                             Payload.FRAME_MASKED,
                             Payload.FRAME_STACKED})


class _Same:
    """`emits = SAME`: this engine hands on what it was given.

    For engines that change values but not the kind of payload -- MedianSub
    corrects a channel, BoxRescale moves boxes -- so that they do not have to
    enumerate every payload they might be placed behind.
    """

    __slots__ = ()

    def __repr__(self):
        return 'SAME'


SAME = _Same()


class Feedback:
    """Declares that an engine is updated with the result of the step it took
    part in, i.e. that the chain has an edge pointing backwards.

    BackgroundSub is the reason this exists: its background estimate is
    blended from the frame it was handed and the boxes the model went on to
    predict from that same frame. That made InferenceEngine.update_engines()
    name three engines literally, which silently stopped being true whenever
    a chain was ordered differently.

    Parameters
    ----------
    method : str
        Name of the method to call on the engine after the forward pass.
        It is called positionally, frame first.
    frame : bool
        Pass the payload this engine itself received this step. Not the
        output of some other engine: being the engine's own input, it is
        whatever the chain in front of it produced, so the edge survives a
        reordering.
    boxes : bool
        Pass the chain's BOXES output, or the ground truth when the step was
        given one.
    """

    __slots__ = ('method', 'frame', 'boxes')

    def __init__(self, method: str, frame: bool = True, boxes: bool = True):
        if not isinstance(method, str) or not method:
            raise ValueError(f'Feedback needs a method name, got {method!r}.')
        self.method = method
        self.frame = bool(frame)
        self.boxes = bool(boxes)

    def __repr__(self):
        return (f'Feedback(method={self.method!r}, frame={self.frame}, '
                f'boxes={self.boxes})')


class Engine:
    """Base class for all engines.

    Carries the chain contract. All three are plain CLASS attributes, never
    properties, because save() serializes an engine by walking its properties
    -- a contract field that became a property would end up in every saved
    config file. An engine whose contract depends on how it was configured
    sets the attribute on the instance instead (DLEngine reads the channel
    count off its ONNX model and sets `accepts` accordingly).

    accepts : Payload, tuple of Payload, or None
        What this engine can be handed. None: anything, i.e. untyped, which
        is what every engine that has not been given a contract yet gets.
    emits : Payload, SAME, or None
        What it hands on. SAME means "whatever it accepted".
    feedback : Feedback or None
        See Feedback.
    path_params : tuple of str
        Names of this engine's parameters that hold a filesystem path.
        hsdetect.engine_io stores those relative to the parameter file and
        resolves them again on the way back, so that a model directory can be
        moved as a whole. Declared here, by the engine that owns the
        parameter, rather than in one list in engine_io that has to know about
        every engine.
    """

    name = None

    accepts = None
    emits = None
    feedback = None
    path_params = ()

    def accepted_payloads(self) -> tuple:
        """`accepts` as a tuple, empty when the engine is untyped.

        Reads the instance, so that an engine which sets `accepts` from its
        own configuration is answered correctly.
        """
        accepts = self.accepts

        if accepts is None:
            return ()
        if isinstance(accepts, Payload):
            return (accepts,)
        return tuple(accepts)

    def save(self) -> dict:
        """Serialize this engine to a parameter dict, by walking its
        properties.

        Properties are the configuration and state of an engine; plain
        attributes are not stored. That is the whole convention: a value that
        should survive a save() is a property, a value that should not -- the
        contract above, a C++ handle, a filter's frame buffer -- is not.

        A property whose value has a save() of its own is stored as the result
        of that call, so a chain of engines nests.
        """
        attr_dict = {}

        for attr_name in dir(self):
            # The attribute on the CLASS, so that reading it does not run the
            # getter unless it really is a property.
            attr = getattr(self.__class__, attr_name, None)

            if isinstance(attr, property):

                attr_val = getattr(self, attr_name)

                if hasattr(attr_val, 'save') and callable(attr_val.save):
                    attr_dict[attr_name] = attr_val.save()
                else:
                    attr_dict[attr_name] = attr_val

        attr_dict['name'] = self.name

        return attr_dict


import hsdetect.annot
import hsdetect.cv
import hsdetect.ipc


def _iter_module_infos():
    """
    Yield (fully_qualified_module_name, file_path) for every .py file
    in this package, recursing into subpackages.
    """
    for finder, module_name, is_pkg in pkgutil.walk_packages(
        __path__, prefix=f"{__name__}."
    ):
        if is_pkg:
            continue  # packages themselves aren't engine files; walk_packages recurses into them for us
        spec = finder.find_spec(module_name)
        if spec is None or spec.origin is None:
            continue
        yield module_name, spec.origin


def _engine_module_info(module_path):
    """
    For an engine module (one that sets IS_ENGINE_MODULE = True at top
    level): (name of its build_* function, names of its classes). None for
    any other module.

    Read from the source instead of importing the module, so that looking
    for engines imports nothing. A module is only imported once one of its
    engines is built (see build_engine()), so e.g. sklearn is only loaded
    when one of the classical classifiers is built.
    """
    with open(module_path, 'rb') as io:
        tree = ast.parse(io.read(), filename=module_path)

    is_engine_module = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "IS_ENGINE_MODULE"
                for t in node.targets)
        and isinstance(node.value, ast.Constant) and node.value.value is True
        for node in tree.body)
    if not is_engine_module:
        return None

    build_fns = [node.name for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name.startswith("build_")]
    if len(build_fns) != 1:
        raise RuntimeError(f"Expected exactly one build_* function in "
                           f"{module_path}, found {build_fns}")

    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    return build_fns[0], classes


@lru_cache(maxsize=None)
def _build_registry():
    """{engine class name: (module name, name of its build_* function)}"""
    registry = {}

    for module_name, module_path in _iter_module_infos():
        info = _engine_module_info(module_path)
        if info is None:
            continue
        build_fn, classes = info

        for class_name in classes:
            if class_name in registry and registry[class_name] != (module_name, build_fn):
                raise RuntimeError(
                    f"Duplicate class name '{class_name}' found in "
                    f"multiple engine modules"
                )
            registry[class_name] = (module_name, build_fn)

    return registry


def build_engine(**kwargs):
    name = kwargs.get("name")
    if name is None:
        raise ValueError("build_engine requires a 'name' kwarg")

    entry = _build_registry().get(name)
    if entry is None:
        raise Exception(f"Engine '{name}' not known in any module!")

    module_name, build_fn = entry
    module = importlib.import_module(module_name)
    return getattr(module, build_fn)(**kwargs)
