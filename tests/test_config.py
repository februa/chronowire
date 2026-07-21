"""Configの不変scope契約を検証する。"""

import pytest

import chronowire as cw


def test_scope_merges_leaf_without_mutating_parent() -> None:
    """子scopeの追加が親の同階層値を消さないことを確認する。"""

    base = cw.Config(system={"fs": 32_768})
    child = base.scope(system={"block_size": 1024})

    assert child.system.fs == 32_768  # type: ignore[union-attr]
    assert child.system.block_size == 1024  # type: ignore[union-attr]
    assert not base.has("system.block_size")
    assert child.parent_scope_id == base.scope_id


def test_scope_rejects_mapping_scalar_collision() -> None:
    """階層をscalarへ暗黙置換して設定を失うことを防ぐ。"""

    base = cw.Config(system={"fs": 32_768})
    with pytest.raises(TypeError):
        base.scope(system=1)


def test_compile_checks_declared_config_paths() -> None:
    """設定不足をrunより前のcompile境界で検出する。"""

    flow = cw.Flow([1], cw.Config())
    mapped = flow.map(lambda value: value, config_paths=("system.fs",))

    with pytest.raises(cw.MissingConfigError):
        cw.compile([mapped])


def test_callable_receives_scoped_config() -> None:
    """NodeがFlow handleに設定されたscopeを受け取ることを確認する。"""

    base = cw.Config(system={"gain": 2})
    flow = cw.Flow([3], base)

    def apply_gain(value: int, config: cw.Config) -> int:
        gain = config.require("system.gain")
        if not isinstance(gain, int):
            raise TypeError("system.gain must be int")
        return value * gain

    mapped = flow.map(apply_gain, config_paths=("system.gain",))
    result = cw.compile([cw.output(mapped, collector=cw.Latest())]).run()

    assert result.outputs[0].emissions[0].value == 6
