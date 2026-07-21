"""Chronowireの公開例外階層を定義する。"""


class ChronowireError(Exception):
    """Chronowireが検出した契約違反の基底例外。"""


class GraphError(ChronowireError):
    """Logical Graphの構築契約違反。"""


class CompileError(ChronowireError):
    """ExecutionPlanを安全に生成できない場合の例外。"""


class DuplicateOutputError(CompileError):
    """同じPortが複数回compile outputへ指定された場合の例外。"""


class MissingConfigError(CompileError):
    """Kernelが宣言したConfig pathを解決できない場合の例外。"""


class KernelExecutionError(ChronowireError):
    """安全なfallbackを生成できないKernel実装失敗。"""


class SynchronizationError(ChronowireError):
    """runtimeの同期契約自体が破損した場合の例外。"""
