import custom_ast
import inspect
import textwrap
from types import FunctionType
from numba import njit, uint8, float32

SCALE = float32(10.0)
BIAS = float32(0.5)

@njit(inline="always", fastmath=True)
def _pulse_log(acc_view, loss_view, i, acc, loss):
    acc_view[i] = uint8(acc * SCALE + BIAS)
    loss_view[i] = uint8(loss * SCALE + BIAS)
    return i + uint8(1)  # your ring256 wrap rule, keep this

class _RewriteLoggerLog(ast.NodeTransformer):
    def __init__(self, logger_name: str):
        self.logger_name = logger_name
        super().__init__()

    def visit_Expr(self, node):
        # Rewrite bare expression statements like: logger.log(a, b)
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "log":
                v = call.func.value
                if isinstance(v, ast.Name) and v.id == self.logger_name:
                    # Replace: logger.log(a,b)
                    # With:    i = _pulse_log(acc_view, loss_view, i, a, b)
                    if len(call.args) != 2:
                        return node
                    return ast.Assign(
                        targets=[ast.Name(id="i", ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id="_pulse_log", ctx=ast.Load()),
                            args=[
                                ast.Name(id="acc_view", ctx=ast.Load()),
                                ast.Name(id="loss_view", ctx=ast.Load()),
                                ast.Name(id="i", ctx=ast.Load()),
                                call.args[0],
                                call.args[1],
                            ],
                            keywords=[],
                        ),
                    )
        return node

def build_compiled_runner(user_train_model: FunctionType):
    """
    Takes user's train_model(logger, max_iterations, alpha, beta) and returns a compiled runner:
      runner(acc_view, loss_view, i0, max_iterations, alpha, beta) -> uint8
    """
    src = inspect.getsource(user_train_model)
    src = textwrap.dedent(src)
    mod = ast.parse(src)

    fn = None
    for n in mod.body:
        if isinstance(n, ast.FunctionDef) and n.name == user_train_model.__name__:
            fn = n
            break
    if fn is None:
        raise RuntimeError("Could not find function definition in source")

    if len(fn.args.args) < 1:
        raise RuntimeError("Expected first arg to be logger")
    logger_name = fn.args.args[0].arg

    # Rewrite logger.log(...) calls
    fn.body = [_RewriteLoggerLog(logger_name).visit(b) for b in fn.body]
    ast.fix_missing_locations(mod)

    # Build a new function wrapper with a new signature
    # def __pulse_runner(acc_view, loss_view, i0, max_iterations, alpha, beta):
    #     i = i0
    #     <original body, with logger.log rewritten>
    #     return i
    runner_name = "__pulse_runner"
    runner_args = ast.arguments(
        posonlyargs=[],
        args=[
            ast.arg(arg="acc_view"),
            ast.arg(arg="loss_view"),
            ast.arg(arg="i0"),
            ast.arg(arg="max_iterations"),
            ast.arg(arg="alpha"),
            ast.arg(arg="beta"),
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )

    init_i = ast.Assign(
        targets=[ast.Name(id="i", ctx=ast.Store())],
        value=ast.Name(id="i0", ctx=ast.Load()),
    )
    ret_i = ast.Return(value=ast.Name(id="i", ctx=ast.Load()))

    runner_fn = ast.FunctionDef(
        name=runner_name,
        args=runner_args,
        body=[init_i] + fn.body + [ret_i],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )

    runner_mod = ast.Module(body=[runner_fn], type_ignores=[])
    ast.fix_missing_locations(runner_mod)

    ns = {
        "_pulse_log": _pulse_log,
        "uint8": uint8,
        "float32": float32,
        "SCALE": SCALE,
        "BIAS": BIAS,
    }
    code = compile(runner_mod, filename="<pulse_wrap>", mode="exec")
    exec(code, ns, ns)

    py_runner = ns[runner_name]

    compiled = njit(py_runner, cache=True, fastmath=True, nogil=True)
    return compiled
