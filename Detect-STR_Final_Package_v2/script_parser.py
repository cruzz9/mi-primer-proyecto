\
# script_parser.py
# Parser seguro y ejecutor de un mini-lenguaje para describir pipelines de
# procesamiento de imágenes con variables.
from __future__ import annotations
import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional, Union
import numpy as np
import cv2
import os

# ----- Representaciones internas -----
@dataclass
class VarRef:
    name: str

@dataclass
class CallSpec:
    func_name: str
    args: List[Any]
    kwargs: Dict[str, Any]

# Paso parseado: tipo en {'call','assign','assign_call'}
ParsedStep = Tuple[str, ...]

# ----- Parser / Evaluator -----
class SafeParseError(Exception):
    pass

class ScriptParser:
    \"\"\"Parser seguro que transforma un script en lista de pasos y puede ejecutarlos.\"\"\"
    def __init__(self):
        # registry: nombre -> callable(proj, *args, **kwargs) -> result
        # por defecto vacío; llenarlo con register_default_commands()
        self.registry: Dict[str, Callable] = {}
        self.register_default_commands()

    # --------------------------
    # Parsing (AST -> CallSpec / VarRef / literals)
    # --------------------------
    def parse(self, script_text: str) -> List[ParsedStep]:
        \"\"\"Parsea el script completo y devuelve lista de pasos...\"\"\"
        tree = ast.parse(script_text, mode='exec')
        steps: List[ParsedStep] = []
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                cs = self._parse_call(node.value)
                steps.append(('call', cs))
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    raise SafeParseError(\"Sólo se soportan asignaciones simples: nombre = expresión\")
                target = node.targets[0].id
                value_node = node.value
                parsed_val = self._parse_expr(value_node)
                if isinstance(parsed_val, CallSpec):
                    steps.append(('assign_call', target, parsed_val))
                else:
                    steps.append(('assign', target, parsed_val))
            elif isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                raise SafeParseError(\"Import, def o class no permitidos en scripts.\")
            elif isinstance(node, ast.Expr) and not isinstance(node.value, ast.Call):
                raise SafeParseError(\"Expresiones sueltas no permitidas excepto llamadas.\")
            else:
                raise SafeParseError(f\"Nodo no soportado: {type(node).__name__}\")
        return steps

    def _parse_call(self, node: ast.Call) -> CallSpec:
        if not isinstance(node.func, ast.Name):
            raise SafeParseError(\"Llamadas deben ser a identificadores simples (p.ej. Rotate(...)).\")
        fname = node.func.id
        args = [self._parse_expr(a) for a in node.args]
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise SafeParseError(\"kwargs con ** no permitido.\")
            kwargs[kw.arg] = self._parse_expr(kw.value)
        return CallSpec(func_name=fname, args=args, kwargs=kwargs)

    def _parse_expr(self, node: ast.AST) -> Any:
        # (contenido igual al provisto por el usuario)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return [self._parse_expr(elt) for elt in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._parse_expr(elt) for elt in node.elts)
        if isinstance(node, ast.Dict):
            keys = [self._parse_expr(k) for k in node.keys]
            vals = [self._parse_expr(v) for v in node.values]
            return dict(zip(keys, vals))
        if isinstance(node, ast.Name):
            return VarRef(node.id)
        if isinstance(node, ast.BinOp):
            left = self._parse_expr(node.left)
            right = self._parse_expr(node.right)
            op = node.op
            return ('binop', op, left, right)
        if isinstance(node, ast.UnaryOp):
            val = self._parse_expr(node.operand)
            return ('unaryop', node.op, val)
        if isinstance(node, ast.Call):
            return self._parse_call(node)
        raise SafeParseError(f\"Expresión no permitida: {type(node).__name__}\")

    def execute(self, steps: List[ParsedStep], proyecto, extra_context: Optional[Dict[str, Any]] = None,
                update_callback: Optional[Callable[[Any, str], None]] = None) -> Tuple[bool, Union[np.ndarray, str, Dict[str, Any]]]:
        vars: Dict[str, Any] = {}
        if extra_context:
            vars.update(extra_context)
        try:
            for step in steps:
                typ = step[0]
                if typ == 'call':
                    callspec: CallSpec = step[1]
                    res = self._exec_call(spec=callspec, vars=vars, proyecto=proyecto)
                    vars['LAST'] = res
                    if update_callback:
                        update_callback(res, f\"Executed {callspec.func_name}\")
                elif typ == 'assign':
                    _, varname, val = step
                    resolved = self._resolve_value(val, vars, proyecto)
                    vars[varname] = resolved
                elif typ == 'assign_call':
                    _, varname, callspec = step
                    res = self._exec_call(spec=callspec, vars=vars, proyecto=proyecto)
                    vars[varname] = res
                    if update_callback:
                        update_callback(res, f\"Assigned {varname} <- {callspec.func_name}\")
                else:
                    raise RuntimeError(\"Paso desconocido: \" + str(step))
            final = proyecto.get_processed()
            return True, {\"final_image\": final, \"variables\": vars}
        except Exception as e:
            return False, f\"Error en ejecución: {e}\"

    def _exec_call(self, spec: CallSpec, vars: Dict[str, Any], proyecto):
        name = spec.func_name
        if name not in self.registry:
            raise RuntimeError(f\"Comando no registrado: {name}\")
        resolved_args = [self._resolve_value(a, vars, proyecto) for a in spec.args]
        resolved_kwargs = {k: self._resolve_value(v, vars, proyecto) for k, v in spec.kwargs.items()}
        func = self.registry[name]
        return func(proyecto, *resolved_args, **resolved_kwargs)

    def _resolve_value(self, v, vars: Dict[str, Any], proyecto):
        if isinstance(v, VarRef):
            name = v.name
            if name in vars:
                return vars[name]
            if name.upper() in (\"CURRENT\", \"THIS\", \"SELF\"):
                return proyecto.get_processed()
            raise RuntimeError(f\"Variable no definida: {name}\")
        if isinstance(v, CallSpec):
            return self._exec_call(v, vars, proyecto)
        if isinstance(v, tuple) and v and v[0] == 'binop':
            _, opnode, left_raw, right_raw = v
            left = self._resolve_value(left_raw, vars, proyecto)
            right = self._resolve_value(right_raw, vars, proyecto)
            return self._apply_binop(opnode, left, right)
        if isinstance(v, tuple) and v and v[0] == 'unaryop':
            _, opnode, operand_raw = v
            operand = self._resolve_value(operand_raw, vars, proyecto)
            return self._apply_unaryop(opnode, operand)
        if isinstance(v, list):
            return [self._resolve_value(x, vars, proyecto) for x in v]
        if isinstance(v, tuple):
            return tuple(self._resolve_value(x, vars, proyecto) for x in v)
        if isinstance(v, dict):
            return {self._resolve_value(k, vars, proyecto): self._resolve_value(val, vars, proyecto) for k, val in v.items()}
        return v

    def _apply_binop(self, opnode, left, right):
        import operator
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        for t, fn in ops.items():
            if isinstance(opnode, t):
                return fn(left, right)
        raise SafeParseError(f\"Operador no soportado: {type(opnode).__name__}\")

    def _apply_unaryop(self, opnode, val):
        import operator
        if isinstance(opnode, ast.UAdd):
            return +val
        if isinstance(opnode, ast.USub):
            return -val
        if isinstance(opnode, ast.Not):
            return not val
        raise SafeParseError(f\"Unary op no soportado: {type(opnode).__name__}\")

    def register_command(self, name: str, fn: Callable):
        self.registry[name] = fn

    def register_default_commands(self):
        def _wrap_proj_method(method_name, ret_key=None):
            def wrapper(proj, *args, **kwargs):
                m = getattr(proj, method_name)
                out = m(*args, **kwargs)
                if isinstance(out, dict):
                    if ret_key is None:
                        for v in out.values():
                            return v
                        return out
                    else:
                        return out.get(ret_key, out)
                return out
            return wrapper

        self.register_command("Rotate", _wrap_proj_method("rotate", "rotatedImage"))
        self.register_command("Gaussian", _wrap_proj_method("gaussianFilter", "gaussian"))
        self.register_command("Equalize", _wrap_proj_method("equalizeImage", "equalizedImage"))
        self.register_command("Roberts", _wrap_proj_method("roberts"))
        self.register_command("Prewitt", _wrap_proj_method("prewitt"))
        self.register_command("Sobel", _wrap_proj_method("sobel"))
        self.register_command("BinaryInvert", _wrap_proj_method("binaryInvert"))
        self.register_command("Mirror", _wrap_proj_method("mirror", "espejo"))
        self.register_command("Erosion", _wrap_proj_method("ErodeImage", "erImage"))
        self.register_command("Dilate", _wrap_proj_method("DilateImage", "dilImage"))
        self.register_command("Add", _wrap_proj_method("AddToImage", "addImage"))
        self.register_command("Load", lambda proj, path: cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
        def cmd_Save(proj, image_or_var, path):
            p = str(path)
            img = image_or_var
            if img is None:
                raise RuntimeError("Save: imagen vacía")
            ok = cv2.imwrite(p, img)
            if not ok:
                raise RuntimeError("Save: fallo al escribir fichero")
            return p
        self.register_command("Save", cmd_Save)

    def available_commands(self) -> List[str]:
        return sorted(list(self.registry.keys()))
