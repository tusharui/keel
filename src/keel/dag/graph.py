from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from keel.errors import CycleDetected, MissingDependency
from keel.reliability.retry import RetryPolicy

NodeFn = Callable[[dict[str, object]], Awaitable[object]]


@dataclass(slots=True)
class Node:
    name: str
    run: NodeFn
    depends_on: tuple[str, ...] = ()
    retries: int = 0
    retry_policy: RetryPolicy | None = None
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("node name cannot be empty")
        if self.retries < 0:
            raise ValueError("retries cannot be negative")


@dataclass(slots=True)
class DAG:
    """Directed acyclic graph of work, validated on construction.

    Validation is not optional and not deferred to run time. A cycle in a
    pipeline is a design error, and discovering it after the expensive upstream
    stages have already run is the worst possible moment to find out.
    """

    nodes: tuple[Node, ...] = ()
    _by_name: dict[str, Node] = field(default_factory=dict, init=False)
    _topological: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        self._by_name = {}
        for node in self.nodes:
            if node.name in self._by_name:
                raise ValueError(f"duplicate node name {node.name!r}")
            self._by_name[node.name] = node
        self._validate_deps()
        self._topological = self._kahn_order()

    def _validate_deps(self) -> None:
        for node in self.nodes:
            for dependency in node.depends_on:
                if dependency not in self._by_name:
                    raise MissingDependency(
                        f"node {node.name!r} depends on {dependency!r}, which is not in the graph"
                    )
                if dependency == node.name:
                    raise CycleDetected(f"node {node.name!r} depends on itself")

    def _kahn_order(self) -> tuple[str, ...]:
        """Topological order by repeatedly removing nodes whose deps are done.

        When no node is ready, whatever is left contains a cycle. The remaining
        nodes are then walked to name it, because 'a cycle exists somewhere'
        costs an afternoon and 'fetch -> rank -> summarise -> fetch' costs one
        glance.
        """
        remaining = {n.name: set(n.depends_on) for n in self.nodes}
        dependents: dict[str, list[str]] = {n.name: [] for n in self.nodes}
        for node in self.nodes:
            for dependency in node.depends_on:
                dependents[dependency].append(node.name)

        ready = sorted(name for name, deps in remaining.items() if not deps)
        order: list[str] = []
        while ready:
            name = ready.pop(0)
            order.append(name)
            for child in dependents[name]:
                remaining[child].discard(name)
                if not remaining[child]:
                    ready.append(child)
            ready.sort()

        if len(order) != len(self.nodes):
            stuck = sorted(set(remaining) - set(order))
            raise CycleDetected(f"cycle among {stuck}: {self._find_cycle(stuck[0])}")
        return tuple(order)

    def _find_cycle(self, start: str) -> str:
        path: list[str] = []
        seen: set[str] = set()

        def walk(name: str) -> list[str] | None:
            if name in seen:
                index = path.index(name)
                return [*path[index:], name]
            seen.add(name)
            path.append(name)
            for dependency in self._by_name[name].depends_on:
                found = walk(dependency)
                if found is not None:
                    return found
            path.pop()
            return None

        cycle = walk(start)
        return " -> ".join(cycle) if cycle else f"{start} -> {start}"

    # --- accessors -----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> Node:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"no such node: {name!r}") from None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)

    @property
    def topological_order(self) -> tuple[str, ...]:
        return self._topological

    def roots(self) -> tuple[str, ...]:
        return tuple(n.name for n in self.nodes if not n.depends_on)

    def leaves(self) -> tuple[str, ...]:
        depended_on = {d for n in self.nodes for d in n.depends_on}
        return tuple(n.name for n in self.nodes if n.name not in depended_on)

    def ancestors(self, name: str) -> set[str]:
        found: set[str] = set()
        stack = list(self[name].depends_on)
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            stack.extend(self[current].depends_on)
        return found

    def descendants(self, name: str) -> set[str]:
        found: set[str] = set()
        stack = [name]
        while stack:
            current = stack.pop()
            for node in self.nodes:
                if current in node.depends_on and node.name not in found:
                    found.add(node.name)
                    stack.append(node.name)
        return found

    def levels(self) -> list[list[str]]:
        """Nodes grouped so every node sits one level after its deepest parent.

        Lets an executor run a whole level concurrently without re-deriving
        readiness on every pass.
        """
        depth: dict[str, int] = {}
        for name in self._topological:
            deps = self[name].depends_on
            depth[name] = 0 if not deps else max(depth[d] for d in deps) + 1

        grouped: dict[int, list[str]] = {}
        for name, level in depth.items():
            grouped.setdefault(level, []).append(name)
        return [sorted(grouped[level]) for level in sorted(grouped)]


def from_nodes(nodes: Sequence[Node]) -> DAG:
    return DAG(nodes=tuple(nodes))
