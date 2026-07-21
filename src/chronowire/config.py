"""不変なスコープ付き設定を提供する。"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Final

_MISSING: Final = object()


def _merge_mapping(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    """属性path単位で設定を再帰マージする。"""

    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key, _MISSING)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mapping(current, value)
        elif isinstance(current, Mapping) != isinstance(value, Mapping) and current is not _MISSING:
            raise TypeError(f"config path {key!r} changes mapping/scalar kind")
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_path(data: Mapping[str, object], path: str) -> object:
    current: object = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


class ConfigView:
    """Config内の一つの階層を読み取り専用の属性アクセスで公開する。

    このviewは値を変更せず、実行中データやKernelStateを保持しない。
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, object]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> object:
        try:
            value = self._data[name]
        except KeyError as error:
            raise AttributeError(name) from error
        if isinstance(value, Mapping):
            return ConfigView(value)
        return copy.deepcopy(value)


class Config:
    """Flow chainが参照する不変な階層設定を表す。

    親scopeを変更せずにleaf単位のoverrideを作成する。時変制御値、KernelState、
    Diagnosticは責務に含めず、それらはGraph Edgeまたはruntime objectで扱う。
    """

    __slots__ = ("_data", "_digest", "_parent_scope_id", "_scope_id")

    def __init__(self, **values: object) -> None:
        self._data = copy.deepcopy(values)
        self._parent_scope_id: str | None = None
        self._digest = self._calculate_digest(self._data)
        self._scope_id = self._digest[:16]

    @classmethod
    def _from_scope(cls, parent: Config, values: Mapping[str, object]) -> Config:
        instance = cls.__new__(cls)
        instance._data = _merge_mapping(parent._data, values)
        instance._parent_scope_id = parent.scope_id
        instance._digest = cls._calculate_digest(instance._data)
        instance._scope_id = instance._digest[:16]
        return instance

    @staticmethod
    def _calculate_digest(data: Mapping[str, object]) -> str:
        encoded = json.dumps(data, sort_keys=True, default=repr, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def scope_id(self) -> str:
        """解決済み設定内容を識別する安定digestを返す。"""

        return self._scope_id

    @property
    def parent_scope_id(self) -> str | None:
        """親scope IDを返し、root scopeではNoneを返す。"""

        return self._parent_scope_id

    @property
    def digest(self) -> str:
        """compile cacheとexportに利用できる完全digestを返す。"""

        return self._digest

    def __getattr__(self, name: str) -> object:
        try:
            value = self._data[name]
        except KeyError as error:
            raise AttributeError(name) from error
        if isinstance(value, Mapping):
            return ConfigView(value)
        return copy.deepcopy(value)

    def scope(self, **overrides: object) -> Config:
        """親を変更せず、指定leafを上書きした子scopeを返す。

        Raises:
            TypeError: 同じpathでmappingとscalarの種類が衝突した場合。
        """

        return self._from_scope(self, overrides)

    def get(self, path: str, default: object = None) -> object:
        """dot区切りpathの値を返し、欠落時はdefaultを返す。"""

        try:
            return copy.deepcopy(_read_path(self._data, path))
        except KeyError:
            return default

    def require(self, path: str) -> object:
        """必須pathの値を返す。

        Raises:
            KeyError: pathが存在しない場合。
        """

        return copy.deepcopy(_read_path(self._data, path))

    def has(self, path: str) -> bool:
        """dot区切りpathが存在するか返す。"""

        try:
            _read_path(self._data, path)
        except KeyError:
            return False
        return True

    def to_dict(self, *, resolved: bool = True) -> dict[str, object]:
        """設定を独立したdictとして返す。

        Args:
            resolved: v0.1では常に解決済み値を返すため、Falseは未対応。

        Raises:
            ValueError: resolved=Falseが指定された場合。
        """

        if not resolved:
            raise ValueError("v0.1 supports only resolved Config export")
        return copy.deepcopy(self._data)
