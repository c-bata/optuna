import ast

from typing import Dict  # NOQA
from optuna import distributions  # NOQA


class Space:
    def __init__(self, trial_object_name):
        self.trial_object_name = trial_object_name
        self.distributions = []
        self.children = []

    def walk_ast_node(self, depth, node):
        # type: (int, ast.AST) -> None
        if isinstance(node, ast.FunctionDef):
            for n in ast.iter_child_nodes(node):
                self.walk_ast_node(depth, n)
        elif isinstance(node, ast.If):
            space_if = Space(trial_object_name=self.trial_object_name)
            for node_if in node.body:
                space_if.walk_ast_node(depth+1, node_if)
            self.children.append(space_if)

            # todo(c-bata): support elif

            space_orelse = Space(trial_object_name=self.trial_object_name)
            for node_orelse in node.orelse:
                space_orelse.walk_ast_node(depth+1, node_orelse)
            self.children.append(space_orelse)
        elif isinstance(node, ast.Assign):
            self.walk_ast_node(depth, node.value)
        elif isinstance(node, ast.Name):
            return
        elif isinstance(node, ast.Call):
            for a in node.args:
                self.walk_ast_node(depth, a)

            call_node = node
            if not isinstance(call_node.func, ast.Attribute):
                return

            node = node.func
            if isinstance(node.value, ast.Attribute):
                return
            if not isinstance(node.value, ast.Name):
                return

            object_name = node.value.id
            if object_name != self.trial_object_name:
                return

            method_name = node.attr  # type: str
            if not method_name.startswith("suggest_"):
                return

            self.distributions.append(call_node)


def dump_space(space, indent=0):
    # type: (Space, int) -> None
    print(" "*indent + "space:")

    indent_next = indent + 2
    for d in space.distributions:
        print(" "*indent_next + ast.dump(d))
    for s in space.children:
        dump_space(s, indent=indent_next)


def generate_search_space(a):
    # type: (ast.FunctionDef) -> Dict[str, distributions.BaseDistribution]

    # 1st argument is trial object.
    trial_object_name = a.args.args[0].arg

    space = Space(trial_object_name)
    space.walk_ast_node(0, a)

    # todo(c-bata): ASTをdistributionsオブジェクトに変換
    from astpretty import pprint
    breakpoint()

    return {}
