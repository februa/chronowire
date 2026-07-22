"""Chronowireの公開例外階層を定義する。"""


class ChronowireError(Exception):
    """Chronowireが検出した契約違反の基底例外。"""


class GraphError(ChronowireError):
    """Logical Graphの構築契約違反。"""


class CompileError(ChronowireError):
    """Planを安全に生成できない場合の例外。"""


class DuplicateOutputError(CompileError):
    """同じPortが複数回compile outputへ指定された場合の例外。"""


class MissingConfigError(CompileError):
    """Kernelが宣言したConfig pathを解決できない場合の例外。"""


class MissingImplementationError(CompileError):
    """選択BackendにOperation実装が登録されていない場合の例外。"""


class ShapeMismatchError(CompileError):
    """Operation入力shapeをcompile時にunifyできない場合の例外。"""


class DuplicateExtensionIdError(CompileError):
    """同じextension_idが一つのPlanへ複数指定された場合の例外。"""


class ExtensionBindingError(ChronowireError):
    """Extension bindingの不足、過剰、種別、ABI契約違反。"""


class ExtensionExecutionError(ChronowireError):
    """Extension handlerがFAIL policyで停止した場合の例外。"""


class KernelExecutionError(ChronowireError):
    """安全なfallbackを生成できないKernel実装失敗。"""


class SynchronizationError(ChronowireError):
    """runtimeの同期契約自体が破損した場合の例外。"""


class SessionError(ChronowireError):
    """Sessionの再利用またはlifecycle契約違反。"""


# v0.4公開名からの一時的な例外alias。
PlanSessionError = SessionError


class SourceExecutionError(ChronowireError):
    """Sourceの開始、受信、停止契約に起因する実行時例外。"""


class ExecutionBindingError(ChronowireError):
    """PortablePlanIRとprocess-local bindingの不一致。"""
